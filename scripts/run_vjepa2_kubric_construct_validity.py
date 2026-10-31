#!/usr/bin/env python3
"""Score the locked paired Kubric target interventions with V-JEPA2-G."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import run_intphys2_dense as core
except ModuleNotFoundError:
    from scripts import run_intphys2_dense as core

MODEL = "vjepa2-g"
CONTEXT = 12
CONTROL_NAMES = ("target_hold", "target_shuffle")
SHUFFLE_VERSION = "kubric-target-tubelet-shuffle-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa2-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rendered-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shuffle-seed", default="kubric-construct-validity-pilot-v1")
    parser.add_argument("--project-revision", required=True)
    parser.add_argument("--max-scenes", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_clip(path: Path) -> np.ndarray:
    files = sorted(path.glob("rgba_*.png"))
    if len(files) != 48:
        raise ValueError(f"Expected 48 frames in {path}, found {len(files)}")
    clip = np.stack([np.asarray(Image.open(file).convert("RGB")) for file in files])
    if clip.shape != (48, 256, 256, 3) or clip.dtype != np.uint8:
        raise ValueError(f"Unexpected clip {path}: {clip.shape} {clip.dtype}")
    return clip


def tubelet_permutation(seed: str, scene_id: str, tubelets: int = 18) -> np.ndarray:
    material = f"{SHUFFLE_VERSION}:{seed}:{scene_id}"
    value = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
    permutation = np.random.default_rng(value).permutation(tubelets)
    if np.array_equal(permutation, np.arange(tubelets)):
        permutation = np.roll(permutation, 1)
    return permutation


def target_shuffle(clip: np.ndarray, seed: str, scene_id: str) -> tuple[np.ndarray, list[int]]:
    permutation = tubelet_permutation(seed, scene_id)
    target = clip[CONTEXT:].reshape(18, 2, 256, 256, 3)
    output = clip.copy()
    output[CONTEXT:] = target[permutation].reshape(36, 256, 256, 3)
    return output, permutation.tolist()


def target_hold(clip: np.ndarray) -> np.ndarray:
    output = clip.copy()
    output[CONTEXT:] = clip[CONTEXT - 1]
    return output


def score_scene(scene: dict, frozen: dict, args, spec: dict, models) -> tuple[list[dict], np.ndarray, dict]:
    from src.masks.utils import apply_masks

    encoder, target_encoder, predictor = models
    conditions = frozen["conditions"]
    clips = []
    records = []
    clean = None
    for condition in conditions:
        clip = load_clip(
            args.rendered_root / scene["scene_id"] / condition["name"] / "frames"
        )
        if clean is None:
            if condition["name"] != "clean":
                raise ValueError("Clean condition must be first")
            clean = clip
        elif not np.array_equal(clip[:CONTEXT], clean[:CONTEXT]):
            raise ValueError(f"Context mismatch: {scene['scene_id']} {condition['name']}")
        clips.append(clip)
        records.append(
            {
                "condition": condition["name"],
                "family": condition["family"],
                "dose": condition["dose"],
                "dose_unit": condition.get("dose_unit", "none"),
                "control_metadata": None,
            }
        )
    held = target_hold(clean)
    shuffled, permutation = target_shuffle(clean, args.shuffle_seed, scene["scene_id"])
    clips.extend([held, shuffled])
    records.extend(
        [
            {
                "condition": "target_hold",
                "family": "control",
                "dose": 0.0,
                "dose_unit": "none",
                "control_metadata": {"held_source_index_0based": CONTEXT - 1},
            },
            {
                "condition": "target_shuffle",
                "family": "control",
                "dose": 0.0,
                "dose_unit": "none",
                "control_metadata": {
                    "version": SHUFFLE_VERSION,
                    "seed": args.shuffle_seed,
                    "target_tubelet_permutation": permutation,
                },
            },
        ]
    )

    torch.cuda.reset_peak_memory_stats()
    normalized = torch.stack(
        [core.normalize_video(torch.from_numpy(clip.copy()), spec["resolution"]) for clip in clips]
    )
    x = normalized.permute(0, 2, 1, 3, 4).contiguous().to("cuda")
    batch_size = len(records)
    enc_mask, pred_mask, full_mask = core.token_masks(CONTEXT, spec, batch_size)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        targets_all = target_encoder(x, full_mask)
        targets_raw = apply_masks(targets_all, [pred_mask], concat=False)[0]
        encoded = encoder(x, enc_mask)
        predictions = predictor(encoded, enc_mask, pred_mask)
        targets = F.layer_norm(targets_raw, (targets_raw.size(-1),))
        error = (predictions - targets).abs().float()
    elapsed = time.perf_counter() - started
    if not all(
        bool(torch.isfinite(tensor).all())
        for tensor in (targets_raw, targets, predictions, error)
    ):
        raise RuntimeError(f"Nonfinite model output for {scene['scene_id']}")
    clean_prediction = predictions[0]
    predictor_differences = (
        predictions - clean_prediction.unsqueeze(0)
    ).abs().float().flatten(1).max(dim=1).values
    if float(predictor_differences.max()) > 1e-4:
        raise RuntimeError(
            f"Same context changed predictions for {scene['scene_id']}: "
            f"{float(predictor_differences.max())}"
        )
    target_shift_l1 = (
        targets - targets[0].unsqueeze(0)
    ).abs().float().mean(dim=(1, 2))
    target_shift_l2 = (
        (targets - targets[0].unsqueeze(0)).float().pow(2).mean(dim=(1, 2)).sqrt()
    )
    spatial_size = spec["resolution"] // 16
    temporal_spatial = error.mean(-1).reshape(batch_size, 18, spatial_size, spatial_size)
    temporal = temporal_spatial.mean(dim=(-1, -2))
    scalar = error.mean(dim=(1, 2))

    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "scene_id": scene["scene_id"],
                "seed": scene["seed"],
                "target_object": scene["target"]["blender_object_name"],
                **record,
                "scalar_l1": float(scalar[index]),
                "temporal_tubelet_l1": json.dumps(temporal[index].cpu().tolist()),
                "target_shift_l1_layernorm": float(target_shift_l1[index]),
                "target_shift_l2_layernorm": float(target_shift_l2[index]),
                "predictor_max_abs_difference_from_clean": float(
                    predictor_differences[index]
                ),
            }
        )
    maps = temporal_spatial.to(torch.float16).cpu().numpy()
    runtime = {
        "scene_id": scene["scene_id"],
        "batch_size": batch_size,
        "inference_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "maximum_predictor_difference": float(predictor_differences.max()),
    }
    del targets_all, targets_raw, targets, encoded, predictions, error, x
    torch.cuda.empty_cache()
    return rows, maps, runtime


def main() -> None:
    args = parse_args()
    expected_checkpoint = "67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e"
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != expected_checkpoint:
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint_hash}")
    expected_revision = "204698b45b3712590f06245fbfba32d3be539812"
    observed_revision = git_revision(args.vjepa2_root)
    if observed_revision != expected_revision:
        raise ValueError(f"V-JEPA2 revision mismatch: {observed_revision}")
    frozen = json.loads(args.manifest.read_text())
    scenes = frozen["scenes"]
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise ValueError("--max-scenes must be positive")
        scenes = scenes[: args.max_scenes]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    spec = core.MODEL_SPECS[MODEL]
    model_args = SimpleNamespace(checkpoint=args.checkpoint, vjepa2_root=args.vjepa2_root)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    encoder, target_encoder, predictor, load_metadata = core.load_models(model_args, spec)
    load_seconds = time.perf_counter() - load_started
    all_rows = []
    maps = []
    runtimes = []
    for scene in scenes:
        rows, scene_maps, runtime = score_scene(
            scene, frozen, args, spec, (encoder, target_encoder, predictor)
        )
        all_rows.extend(rows)
        maps.append(scene_maps)
        runtimes.append(runtime)
        print(json.dumps(runtime), flush=True)
    fields = list(all_rows[0])
    with (args.output_dir / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    np.savez_compressed(
        args.output_dir / "temporal_spatial_maps.npz",
        maps=np.stack(maps),
        scene_ids=np.asarray([scene["scene_id"] for scene in scenes]),
        conditions=np.asarray([row["condition"] for row in all_rows[: len(frozen["conditions"]) + len(CONTROL_NAMES)]]),
    )
    provenance = {
        "status": "complete",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "core_script": str(Path(core.__file__).resolve()),
        "core_script_sha256": sha256_file(Path(core.__file__)),
        "project_revision": args.project_revision,
        "model": MODEL,
        "model_spec": spec,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_origin": "https://dl.fbaipublicfiles.com/vjepa2/vitg.pt",
        "vjepa2_root": str(args.vjepa2_root.resolve()),
        "vjepa2_revision": observed_revision,
        "frozen_manifest": str(args.manifest.resolve()),
        "frozen_manifest_sha256": sha256_file(args.manifest),
        "rendered_root": str(args.rendered_root.resolve()),
        "selected_scenes": len(scenes),
        "conditions": [condition["name"] for condition in frozen["conditions"]],
        "controls": list(CONTROL_NAMES),
        "context": CONTEXT,
        "target_tubelets": 18,
        "score": "mean absolute predictor minus layer-normalized EMA target tokens",
        "dtype": "bfloat16 autocast; float32 error reduction",
        "condition_batch_size": len(frozen["conditions"]) + len(CONTROL_NAMES),
        "load_seconds": load_seconds,
        "runtime": runtimes,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        **load_metadata,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    del encoder, target_encoder, predictor
    gc.collect()


if __name__ == "__main__":
    main()
