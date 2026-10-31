#!/usr/bin/env python3
"""Run dense V-JEPA 2-G surprise on original IntPhys PNG videos."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import run_intphys2_dense as core
except ModuleNotFoundError:  # Imported as scripts.run_intphys1_vjepa2_dense
    from scripts import run_intphys2_dense as core


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa2-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pairs-manifest", type=Path, required=True)
    parser.add_argument("--block", choices=("O1", "O2", "O3"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, action="append")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--window-stride", type=int, default=2)
    parser.add_argument("--max-pairs", type=int)
    return parser.parse_args()


def manifest_pairs(path: Path, block: str) -> list[dict]:
    payload = json.loads(path.read_text())
    pairs = payload["pairs"] if isinstance(payload, dict) else payload
    selected = [pair for pair in pairs if pair["block"] == block]
    identities = {(pair["scene"], int(pair["pair"])) for pair in selected}
    if len(identities) != len(selected):
        raise ValueError("Duplicate pair identities")
    return sorted(selected, key=lambda pair: (pair["scene"], int(pair["pair"])))


def read_video(path: Path, frame_step: int):
    paths = sorted((path / "scene").glob("*.png"))[:99:frame_step]
    if len(paths) < 48:
        raise ValueError(f"Too few sampled frames in {path}: {len(paths)}")
    arrays = [np.asarray(Image.open(frame).convert("RGB"), dtype=np.uint8) for frame in paths]
    source_numbers = np.asarray([int(frame.stem.rsplit("_", 1)[1]) for frame in paths])
    return core.normalize_video(torch.from_numpy(np.stack(arrays)), 256), source_numbers


def run_video(path: Path, identity: dict, contexts: tuple[int, ...], args, models):
    from src.masks.utils import apply_masks

    spec = core.MODEL_SPECS["vjepa2-g"]
    encoder, target_encoder, predictor = models
    frames, sampled = read_video(path, args.frame_step)
    starts = list(range(0, len(sampled) - 48 + 1, args.window_stride))
    maps = {str(context): [] for context in contexts}
    rows = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for window_index, start in enumerate(starts):
            indices = sampled[start : start + 48]
            x = frames[start : start + 48].permute(1, 0, 2, 3).unsqueeze(0).contiguous().cuda()
            full = core.token_masks(contexts[0], spec)[2]
            targets_all = target_encoder(x, full)
            for context in contexts:
                enc_mask, pred_mask, _ = core.token_masks(context, spec)
                targets = apply_masks(targets_all, [pred_mask], concat=False)[0]
                encoded = encoder(x, enc_mask)
                predictions = predictor(encoded, enc_mask, pred_mask)
                targets = F.layer_norm(targets, (targets.size(-1),))
                error = (predictions - targets).abs().float()
                spatial = error.mean(-1).reshape((48 - context) // 2, 16, 16)
                temporal = spatial.mean((-1, -2))
                if not bool(torch.isfinite(error).all()):
                    raise RuntimeError(f"Non-finite error in {path}, window {window_index}")
                pairs = core.target_frame_pairs(indices, context)
                rows.append({
                    "window_index": window_index,
                    "context_length": context,
                    "window_source_start_frame_1based": int(indices[0]),
                    "window_source_end_frame_1based": int(indices[-1]),
                    "context_source_frames_1based": indices[:context].tolist(),
                    "target_source_frames_1based": indices[context:].tolist(),
                    "target_tubelet_source_frame_pairs_1based": pairs,
                    "scalar_l1": float(error.mean().item()),
                    "temporal_tubelet_l1": temporal.cpu().tolist(),
                })
                maps[str(context)].append(spatial.to(torch.float16).cpu().numpy())
    return {
        "dataset": "IntPhys",
        "block": args.block,
        "scene": identity["scene"],
        "pair": int(identity["pair"]),
        "video": path.name,
        "possible": bool(identity["possible"]),
        "frame_step": args.frame_step,
        "frames_per_clip": 48,
        "window_stride_sampled_frames": args.window_stride,
        "sampled_source_frames_1based": sampled.tolist(),
        "windows": rows,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }, {key: np.stack(value) for key, value in maps.items()}


def consolidate(root: Path):
    fields = ["block", "scene", "pair", "video", "possible", "window_index", "context_length", "window_source_start_frame_1based", "window_source_end_frame_1based", "scalar_l1", "context_source_frames_1based", "target_source_frames_1based", "target_tubelet_source_frame_pairs_1based", "temporal_tubelet_l1"]
    with (root / "windows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for path in sorted((root / "videos").glob("*.json")):
            payload = json.loads(path.read_text())
            for row in payload["windows"]:
                writer.writerow({field: json.dumps(row[field], separators=(",", ":")) if isinstance(row.get(field), list) else row.get(field, payload.get(field)) for field in fields})


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    contexts = tuple(args.context or core.MODEL_SPECS["vjepa2-g"]["contexts"])
    pairs = manifest_pairs(args.pairs_manifest, args.block)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.output_dir / "videos"
    maps_dir = args.output_dir / "spatial_maps"
    videos_dir.mkdir(exist_ok=True); maps_dir.mkdir(exist_ok=True)
    sys.path.insert(0, str(args.vjepa2_root))
    encoder, target, predictor, load_info = core.load_models(
        argparse.Namespace(checkpoint=args.checkpoint, vjepa2_root=args.vjepa2_root),
        core.MODEL_SPECS["vjepa2-g"],
    )
    provenance = {
        "model": "vjepa2-g", "protocol": "first 99 source frames; frame_step 1; FPC 48; stride 2",
        "contexts": contexts, "pairs": len(pairs), "checkpoint_sha256": core.sha256_file(args.checkpoint),
        "checkpoint": str(args.checkpoint), "vjepa2_revision": core.revision(args.vjepa2_root),
        "data_root": str(args.data_root), "pairs_manifest": str(args.pairs_manifest),
        "block": args.block, "batch_size": 1, "dtype": "bfloat16 autocast", **load_info,
    }
    core.write_atomic_text(args.output_dir / "provenance.json", json.dumps(provenance, indent=2) + "\n")
    total = len(pairs) * 2
    done = 0
    for pair in pairs:
        for side in ("a", "b"):
            video = str(pair[f"video_{side}"])
            name = f"{args.block}-{pair['scene']}-pair{pair['pair']}-video{video}"
            json_path = videos_dir / f"{name}.json"; map_path = maps_dir / f"{name}.npz"
            if json_path.exists() and map_path.exists():
                done += 1; continue
            identity = {"scene": pair["scene"], "pair": pair["pair"], "possible": pair[f"video_{side}_possible"]}
            payload, maps = run_video(args.data_root / args.block / pair["scene"] / video, identity, contexts, args, (encoder, target, predictor))
            temporary = map_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **{f"context_{key}": value for key, value in maps.items()})
            temporary.replace(map_path)
            payload["temporal_spatial_l1_npz"] = str(map_path.relative_to(args.output_dir))
            payload["temporal_spatial_l1_shapes"] = {key: list(value.shape) for key, value in maps.items()}
            core.write_atomic_text(json_path, json.dumps(payload, separators=(",", ":")) + "\n")
            done += 1
            print(json.dumps({"done": done, "total": total, "name": name, "wall_seconds": payload["wall_seconds"]}), flush=True)
    consolidate(args.output_dir)
    core.write_atomic_text(args.output_dir / "status.json", json.dumps({"status": "complete", "pairs": len(pairs), "videos": total}, indent=2) + "\n")


if __name__ == "__main__":
    main()
