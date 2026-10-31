#!/usr/bin/env python3
"""Extract locked layer summaries from clean and appearance-edited IntPhys clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import VJEPA2Model, VJEPA2VideoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def sample_frame_paths(video_dir: Path, count: int) -> tuple[list[Path], list[int]]:
    frames = sorted((video_dir / "scene").glob("scene_*.png"))
    if not frames:
        raise FileNotFoundError(f"No scene frames under {video_dir}")
    indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[int(i)] for i in indices], indices.tolist()


def load_rgb_video(frame_paths: list[Path]) -> np.ndarray:
    frames = []
    for path in frame_paths:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")))
    video = np.stack(frames)
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Unexpected video shape: {video.shape}")
    return video


def build_records(config: dict[str, Any], max_pairs: int | None) -> list[dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[1]
    pairs_path = Path(config["pairs_manifest"])
    if not pairs_path.is_absolute():
        pairs_path = repo_root / pairs_path
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    selected = [
        pair
        for pair in pairs
        if pair["block"] == config["block"] and pair["scene"] in config["scenes"]
    ]
    if max_pairs is not None:
        selected = selected[:max_pairs]
    split_by_scene = {
        **{scene: "development" for scene in config["development_scenes"]},
        **{scene: "test" for scene in config["test_scenes"]},
    }
    dataset_root = Path(config["dataset_root"])
    appearance_root = Path(config["appearance_root"])
    records: list[dict[str, Any]] = []
    for pair in selected:
        scene = pair["scene"]
        for side in ("a", "b"):
            video = str(pair[f"video_{side}"])
            possible = bool(pair[f"video_{side}_possible"])
            for condition in config["conditions"]:
                if condition["transform"] == "clean":
                    video_dir = dataset_root / pair["block"] / scene / video
                else:
                    video_dir = (
                        appearance_root
                        / condition["transform"]
                        / condition["severity"]
                        / pair["block"]
                        / scene
                        / video
                    )
                records.append(
                    {
                        "sample_id": (
                            f"{pair['block']}-{scene}-pair{pair['pair']}-{side}-"
                            f"{condition['name']}"
                        ),
                        "base_video_id": f"{pair['block']}-{scene}-{video}",
                        "base_pair_id": f"{pair['block']}-{scene}-pair{pair['pair']}",
                        "block": pair["block"],
                        "scene": scene,
                        "pair": int(pair["pair"]),
                        "side": side,
                        "video": video,
                        "possible": possible,
                        "condition": condition["name"],
                        "transform": condition["transform"],
                        "severity": condition["severity"],
                        "split": split_by_scene[scene],
                        "video_dir": str(video_dir),
                    }
                )
    return records


class SummaryHooks:
    def __init__(self, model: VJEPA2Model, layers: list[int]):
        self.summaries: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.handles = []
        modules: dict[int, torch.nn.Module] = {0: model.encoder.embeddings}
        for depth in layers:
            if depth == 0:
                continue
            modules[depth] = (
                model.encoder.layernorm
                if depth == model.config.num_hidden_layers
                else model.encoder.layer[depth - 1]
            )
        for depth in layers:
            self.handles.append(modules[depth].register_forward_hook(self._hook(depth)))

    def _hook(self, depth: int):
        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.detach().float()
            mean = hidden.mean(dim=1).cpu().numpy()[0]
            std = hidden.std(dim=1, unbiased=False).cpu().numpy()[0]
            self.summaries[depth] = (mean, std)

        return capture

    def clear(self) -> None:
        self.summaries.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = (args.output_root or Path(config["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(config["model_source"]).resolve()
    actual_checkpoint_hash = sha256_file(checkpoint / "model.safetensors")
    if actual_checkpoint_hash != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 does not match the locked config")
    records = build_records(config, args.max_pairs)
    if not records:
        raise ValueError("No records selected")

    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(config["device"])
    dtype = torch.float16 if config["dtype"] == "float16" else torch.float32
    torch.manual_seed(config["probe"]["seeds"][0])
    np.random.seed(config["probe"]["seeds"][0])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "VRAM GiB: "
        f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}; batch_size=1"
    )
    processor = VJEPA2VideoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = VJEPA2Model.from_pretrained(
        checkpoint, local_files_only=True, dtype=dtype
    ).to(device)
    model.eval()
    layers = [int(layer) for layer in config["layers"]]
    hooks = SummaryHooks(model, layers)
    pooled_rows = []
    mean_std_rows = []
    sampled_indices: list[list[int]] = []
    elapsed_rows = []
    determinism_max_abs = None
    first_summaries = None
    torch.cuda.reset_peak_memory_stats()

    try:
        for index, record in enumerate(records):
            frame_paths, indices = sample_frame_paths(
                Path(record["video_dir"]), int(config["frames_per_clip"])
            )
            video = load_rgb_video(frame_paths)
            batch = processor(videos=video, return_tensors="pt")
            inputs = batch["pixel_values_videos"].to(device)
            hooks.clear()
            start = time.perf_counter()
            with torch.inference_mode():
                output = model(
                    pixel_values_videos=inputs,
                    skip_predictor=True,
                    output_hidden_states=False,
                )
            torch.cuda.synchronize()
            elapsed_rows.append(time.perf_counter() - start)
            if not torch.isfinite(output.last_hidden_state).all():
                raise ValueError(f"Non-finite model output for {record['sample_id']}")
            missing = sorted(set(layers) - set(hooks.summaries))
            if missing:
                raise RuntimeError(f"Missing hooked layers: {missing}")
            pooled = np.stack([hooks.summaries[layer][0] for layer in layers])
            mean_std = np.stack(
                [np.concatenate(hooks.summaries[layer]) for layer in layers]
            )
            if not np.isfinite(pooled).all() or not np.isfinite(mean_std).all():
                raise ValueError(f"Non-finite summary for {record['sample_id']}")
            pooled_rows.append(pooled)
            mean_std_rows.append(mean_std)
            sampled_indices.append(indices)

            if index == 0:
                first_summaries = (pooled.copy(), mean_std.copy())
                hooks.clear()
                with torch.inference_mode():
                    model(
                        pixel_values_videos=inputs,
                        skip_predictor=True,
                        output_hidden_states=False,
                    )
                rerun_pooled = np.stack([hooks.summaries[layer][0] for layer in layers])
                rerun_mean_std = np.stack(
                    [np.concatenate(hooks.summaries[layer]) for layer in layers]
                )
                determinism_max_abs = float(
                    max(
                        np.max(np.abs(first_summaries[0] - rerun_pooled)),
                        np.max(np.abs(first_summaries[1] - rerun_mean_std)),
                    )
                )
                if determinism_max_abs > config["gate"]["determinism_atol"]:
                    raise ValueError(
                        f"Determinism tolerance failed: {determinism_max_abs}"
                    )
            print(
                f"[{index + 1:03d}/{len(records):03d}] {record['sample_id']} "
                f"{elapsed_rows[-1]:.2f}s"
            )
    finally:
        hooks.close()

    features_path = output_root / "layer_features.npz"
    np.savez_compressed(
        features_path,
        pooled=np.stack(pooled_rows).astype(np.float32),
        token_mean_std=np.stack(mean_std_rows).astype(np.float32),
        layers=np.asarray(layers, dtype=np.int64),
        sampled_source_indices=np.asarray(sampled_indices, dtype=np.int64),
    )
    metadata_path = output_root / "samples.json"
    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "command": " ".join(["python", str(Path(__file__)), *map(str, vars(args).values())]),
        "git_revision": git_text("rev-parse", "HEAD"),
        "git_dirty": bool(git_text("status", "--porcelain")),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_origin": config["model_origin"],
        "checkpoint_sha256": actual_checkpoint_hash,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "record_count": len(records),
        "layers": layers,
        "frames_per_clip": config["frames_per_clip"],
        "batch_size": 1,
        "elapsed_seconds_total": float(sum(elapsed_rows)),
        "elapsed_seconds_median": float(np.median(elapsed_rows)),
        "determinism_max_abs": determinism_max_abs,
        "features_path": str(features_path),
        "features_sha256": sha256_file(features_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
