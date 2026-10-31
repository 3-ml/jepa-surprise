#!/usr/bin/env python3
"""Benchmark exact V-JEPA2-G inference with sequential CPU/GPU block offload."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

try:
    import intphys2_confirmatory_common as common
except ModuleNotFoundError:
    from scripts import intphys2_confirmatory_common as common


LAYERS = ("block_00", "block_10", "block_20", "block_30", "final")
NATIVE_LAYERS = (0, 10, 20, 30, 40)
CHECKPOINT_SHA256 = "67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e"
UPSTREAM_REVISION = "204698b45b3712590f06245fbfba32d3be539812"
SEED = 230826


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa2-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-benchmark-clips", type=int, default=20)
    return parser.parse_args()


def git_text(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def nvidia_row() -> dict[str, str]:
    fields = [
        "timestamp",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
    ]
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    values = subprocess.check_output(command, text=True).strip().split(", ")
    return dict(zip(fields, values))


def normalize_frames(frames: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).float()
    steps, height, width, channels = tensor.shape
    tensor = tensor.reshape(steps, height, width, channels).permute(0, 3, 1, 2)
    tensor = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
    tensor = tensor.reshape(1, steps, channels, 256, 256).permute(0, 2, 1, 3, 4)
    tensor = tensor.contiguous().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    return tensor.sub_(mean).div_(std)


def module_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    marker = prefix + "."
    return {key[len(marker) :]: value for key, value in state.items() if key.startswith(marker)}


class SequentialOffloadEncoder:
    def __init__(self, model: torch.nn.Module, state: dict[str, torch.Tensor]):
        self.model = model
        self.states = {
            "patch_embed": module_state(state, "patch_embed"),
            "norm": module_state(state, "norm"),
        }
        self.states.update(
            {f"blocks.{i}": module_state(state, f"blocks.{i}") for i in range(40)}
        )

    def _run_module(self, name: str, module: torch.nn.Module, *args, **kwargs):
        module.to("cuda")
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = module(*args, **kwargs)
            torch.cuda.synchronize()
            return output
        finally:
            module.load_state_dict(self.states[name], strict=True, assign=True)

    @staticmethod
    def _pool(tokens: torch.Tensor) -> np.ndarray:
        return tokens.detach().float().mean(dim=1).cpu().numpy()[0]

    def infer(self, normalized: torch.Tensor) -> tuple[np.ndarray, dict[str, float]]:
        process = psutil.Process()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rss_start = process.memory_info().rss
        started = time.perf_counter()
        captures: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            x = normalized.to("cuda")
            x = self._run_module("patch_embed", self.model.patch_embed, x)
            captures[0] = self._pool(x)
            for index, block in enumerate(self.model.blocks):
                x = self._run_module(
                    f"blocks.{index}",
                    block,
                    x,
                    mask=None,
                    attn_mask=None,
                    T=32,
                    H_patches=16,
                    W_patches=16,
                )
                if index + 1 in {10, 20, 30}:
                    captures[index + 1] = self._pool(x)
            x = self._run_module("norm", self.model.norm, x)
            captures[40] = self._pool(x)
            del x
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = {
            "seconds": elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "rss_start_bytes": int(rss_start),
            "rss_end_bytes": int(process.memory_info().rss),
        }
        return np.stack([captures[layer] for layer in NATIVE_LAYERS]), metrics


def load_exact_encoder(checkpoint: Path, root: Path):
    if common.sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA-256 mismatch")
    if git_text(root, "rev-parse", "HEAD") != UPSTREAM_REVISION:
        raise ValueError("Pinned V-JEPA2 revision mismatch")
    if git_text(root, "status", "--porcelain", "--untracked-files=all", "--", "src"):
        raise ValueError("Pinned V-JEPA2 source tree is dirty")
    sys.path.insert(0, str(root))
    from src.models import vision_transformer as vit

    started = time.perf_counter()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = {
        key.removeprefix("module.backbone."): value
        for key, value in payload["encoder"].items()
    }
    original_linspace = torch.linspace

    def cpu_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_linspace(*args, **kwargs)

    torch.linspace = cpu_linspace
    try:
        with torch.device("meta"):
            model = vit.vit_giant_xformers(
                img_size=256,
                num_frames=64,
                uniform_power=True,
                use_activation_checkpointing=False,
                use_rope=True,
            )
    finally:
        torch.linspace = original_linspace
    load_result = model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    provenance = {
        "load_seconds": time.perf_counter() - started,
        "strict_load": str(load_result),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "gradient_buffers": sum(parameter.grad is not None for parameter in model.parameters()),
        "checkpoint_keys": sorted(payload),
    }
    return model, state, payload, provenance


def load_reference(cache_path: Path) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    with np.load(cache_path, allow_pickle=False) as data:
        recolor_metadata = {}
        for index in np.flatnonzero(data["transform_family"] == "selective_recolor"):
            recolor_metadata[str(data["name"][index])] = {
                "selective_center_degrees": str(data["selective_center_degrees"][index]),
                "transform_feasible": str(data["transform_feasible"][index]),
            }
        clean = np.flatnonzero(data["appearance_condition"] == "clean")
        metadata = []
        references = {}
        for index in clean:
            row = {key: str(data[key][index]) for key in data["metadata_columns"]}
            row.update(recolor_metadata[row["name"]])
            metadata.append(row)
            references[row["name"]] = data["X"][index].astype(np.float32)
    metadata.sort(key=lambda row: (row["scene_id"], row["pair_id"], row["name"]))
    return metadata, references


def appearance_row(source: dict[str, str], family: str, value: float) -> dict[str, str]:
    row = dict(source)
    row["transform_family"] = family
    row["transform_value"] = str(value)
    if family != "selective_recolor":
        row["transform_feasible"] = "true"
    return row


def benchmark_jobs(metadata: list[dict[str, str]], count: int) -> list[tuple[dict, str, float]]:
    grid = [
        ("clean", 0.0),
        ("hue", 15.0),
        ("brightness", 0.85),
        ("selective_recolor", 45.0),
        ("hue", 90.0),
        ("brightness", 0.45),
        ("selective_recolor", 180.0),
        ("hue", 45.0),
        ("brightness", 0.65),
        ("selective_recolor", 90.0),
    ]
    jobs = []
    for index in range(count):
        jobs.append((metadata[index % len(metadata)], *grid[index % len(grid)]))
    return jobs


def compare(reference: np.ndarray, observed: np.ndarray) -> list[dict[str, float | str]]:
    output = []
    for layer, expected, actual in zip(LAYERS, reference, observed):
        denominator = np.linalg.norm(expected) * np.linalg.norm(actual)
        output.append(
            {
                "layer": layer,
                "cosine": float(np.dot(expected, actual) / denominator),
                "max_abs_diff": float(np.max(np.abs(expected - actual))),
                "mean_abs_diff": float(np.mean(np.abs(expected - actual))),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    metadata, references = load_reference(args.reference_cache)
    model, state, payload, load_provenance = load_exact_encoder(
        args.checkpoint.resolve(), args.vjepa2_root.resolve()
    )
    encoder = SequentialOffloadEncoder(model, state)
    shared_config = json.loads(Path("configs/intphys2_confirmatory_shared.json").read_text())
    jobs = benchmark_jobs(metadata, args.max_benchmark_clips)
    results = []
    validation = None
    runtime_log = args.output_dir / "hardware_runtime_log.csv"
    with runtime_log.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "job_index",
                "source_id",
                "family",
                "dose",
                "seconds",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "rss_start_bytes",
                "rss_end_bytes",
                "timestamp",
                "utilization.gpu",
                "memory.used",
                "memory.total",
                "temperature.gpu",
                "power.draw",
            ],
        )
        writer.writeheader()
        for job_index, (source, family, dose) in enumerate(jobs, 1):
            path = args.dataset_root / source["source_relative_path"]
            indices = json.loads(source["source_frame_indices_0based"])
            frames = common.decode_selected_frames(path, indices)
            if family != "clean":
                frames = common.apply_appearance(
                    frames, appearance_row(source, family, dose), shared_config
                )
            normalized = normalize_frames(frames)
            del frames
            pooled, metrics = encoder.infer(normalized)
            del normalized
            if job_index == 1:
                validation = compare(references[source["name"]], pooled)
                if min(item["cosine"] for item in validation) < 0.999:
                    raise RuntimeError(f"Clean-cache validation failed: {validation}")
            hardware = nvidia_row()
            row = {
                "job_index": job_index,
                "source_id": source["name"],
                "family": family,
                "dose": dose,
                **metrics,
                **hardware,
            }
            writer.writerow(row)
            handle.flush()
            results.append(row)
            print(json.dumps({"completed": job_index, **row}), flush=True)
            del pooled
            gc.collect()
    summary = {
        "status": "PASS",
        "strategy": "exact sequential transformer-block CPU/GPU offload",
        "load": load_provenance,
        "validation": validation,
        "clips": results,
        "seconds_median": float(np.median([row["seconds"] for row in results])),
        "seconds_p90": float(np.quantile([row["seconds"] for row in results], 0.9)),
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in results),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in results),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "seed": SEED,
    }
    temporary = args.output_dir / "benchmark_results.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2))
    os.replace(temporary, args.output_dir / "benchmark_results.json")
    print(json.dumps(summary, indent=2))
    del encoder, model, state, payload
    gc.collect()


if __name__ == "__main__":
    main()
