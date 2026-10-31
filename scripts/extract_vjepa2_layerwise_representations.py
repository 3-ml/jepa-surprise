#!/usr/bin/env python3
"""Extract layerwise frozen V-JEPA2 representations for workshop probes."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    import run_intphys2_construct_validity as construct
    import run_intphys2_dense as core
except ModuleNotFoundError:  # Imported as scripts.extract_vjepa2_layerwise_representations
    from scripts import run_intphys2_construct_validity as construct
    from scripts import run_intphys2_dense as core

APPEARANCE_CONDITIONS = (
    "original",
    "grayscale",
    "brightness_contrast",
    "hue_shift",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("vjepa2-g",), default="vjepa2-g")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa2-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("Debug", "Main"), default="Main")
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--appearance-condition", choices=APPEARANCE_CONDITIONS, action="append")
    parser.add_argument("--frame-step", type=int)
    parser.add_argument("--window-batch-size", type=int, default=1)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--max-windows-per-video", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_encoder(args: argparse.Namespace, spec: dict) -> tuple[torch.nn.Module, dict]:
    sys.path.insert(0, str(args.vjepa2_root))
    from src.models import vision_transformer as vit

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    common = {
        "img_size": spec["resolution"],
        "num_frames": spec["frames_per_clip"],
        "uniform_power": True,
        "use_activation_checkpointing": True,
        "use_rope": spec["use_rope"],
    }
    architecture = getattr(vit, spec["architecture"])
    encoder, encoder_result = core.load_component(
        lambda: architecture(**common),
        checkpoint["encoder"],
        allow_pos_resize=not spec["use_rope"],
    )
    checkpoint_keys = sorted(checkpoint)
    del checkpoint
    gc.collect()
    return encoder, {
        "checkpoint_keys": checkpoint_keys,
        "strict_load": {"encoder": encoder_result},
    }


def _resolve_dotted_attribute(obj, dotted_path: str):
    current = obj
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def find_encoder_blocks(encoder: torch.nn.Module) -> tuple[list[torch.nn.Module], str]:
    for path in ("blocks", "module.blocks", "backbone.blocks", "encoder.blocks"):
        candidate = _resolve_dotted_attribute(encoder, path)
        if candidate is None:
            continue
        blocks = list(candidate)
        if blocks:
            return blocks, path
    raise ValueError("Could not find an iterable transformer block stack on the encoder")


def _first_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported layer output type: {type(output)!r}")


class LayerCapture:
    def __init__(self, blocks: list[torch.nn.Module]):
        self.layer_names = [f"block_{index:02d}" for index in range(len(blocks))]
        self._captures: dict[str, torch.Tensor] = {}
        self._handles = [
            block.register_forward_hook(self._hook(name))
            for block, name in zip(blocks, self.layer_names)
        ]

    def _hook(self, name: str):
        def capture(_module, _inputs, output) -> None:
            tensor = _first_tensor(output)
            self._captures[name] = tensor.detach().float().mean(dim=1).cpu()

        return capture

    def clear(self) -> None:
        self._captures.clear()

    def features(self, encoded) -> np.ndarray:
        missing = [name for name in self.layer_names if name not in self._captures]
        if missing:
            raise RuntimeError(f"Missing layer captures: {missing[:5]}")
        pooled = [self._captures[name].numpy() for name in self.layer_names]
        pooled.append(_first_tensor(encoded).detach().float().mean(dim=1).cpu().numpy())
        return np.stack(pooled, axis=1)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def transform_clip(frames: np.ndarray, condition: str) -> np.ndarray:
    if condition == "original":
        return frames
    if condition == "grayscale":
        gray = np.rint(
            0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
        ).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=-1)
    if condition == "brightness_contrast":
        return np.clip(frames.astype(np.float32) * 0.72 + 20.0, 0, 255).round().astype(
            np.uint8
        )
    if condition == "hue_shift":
        shifted = []
        hue_shift = int(round(40.0 * 256.0 / 360.0))
        for frame in frames:
            hsv = core.rgb_to_hsv(frame).copy()
            hsv[..., 0] = (hsv[..., 0].astype(np.uint16) + hue_shift) % 256
            shifted.append(core.hsv_to_rgb(hsv))
        return np.stack(shifted)
    raise ValueError(f"Unknown appearance condition: {condition}")


def infer_frame_step(rows: list[dict], requested: int | None) -> int:
    observed: set[int] = set()
    for row in rows:
        for field in ("context_source_frames_0based", "target_source_frames_0based"):
            values = row.get(field) or []
            if len(values) >= 2:
                observed.add(int(values[1]) - int(values[0]))
    if requested is not None:
        if requested <= 0:
            raise ValueError("--frame-step must be positive")
        if observed and observed != {requested}:
            raise ValueError(
                f"--frame-step {requested} does not match window manifest steps {sorted(observed)}"
            )
        return requested
    if len(observed) != 1:
        raise ValueError(
            "Could not infer a unique frame step from the window manifest; pass --frame-step"
        )
    return observed.pop()


def select_window_rows(
    path: Path,
    records: list[dict],
    *,
    max_windows_per_video: int | None,
) -> dict[str, list[dict]]:
    rows = read_jsonl(path)
    names = {str(record["name"]) for record in records}
    rows = [row for row in rows if str(row.get("name")) in names]
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_video[str(row["name"])].append(row)
    if set(by_video) != names:
        missing = sorted(names - set(by_video))
        raise ValueError(f"Window manifest is missing selected videos: {missing}")
    for name, video_rows in by_video.items():
        ordered = sorted(video_rows, key=lambda row: int(row["window_index"]))
        if max_windows_per_video is not None:
            if max_windows_per_video <= 0:
                raise ValueError("--max-windows-per-video must be positive")
            ordered = ordered[:max_windows_per_video]
        by_video[name] = ordered
    return by_video


def decode_sampled_rgb(video_path: str, frame_step: int) -> np.ndarray:
    try:
        from decord import VideoReader, cpu
    except ModuleNotFoundError:
        return decode_sampled_rgb_cv2(video_path, frame_step)

    reader = VideoReader(video_path, ctx=cpu(0))
    sampled = np.arange(0, len(reader), frame_step)
    return reader.get_batch(sampled).asnumpy()


def decode_sampled_rgb_cv2(video_path: str, frame_step: int) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0:
        sampled = np.arange(0, frame_count, frame_step)
        frames = []
        for index in sampled:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {int(index)} from {video_path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        frames = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % frame_step == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return np.stack(frames)


def extract_video_features(
    record: dict,
    window_rows: list[dict],
    *,
    frame_step: int,
    appearance_conditions: tuple[str, ...],
    spec: dict,
    encoder: torch.nn.Module,
    capture: LayerCapture,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("--window-batch-size must be positive")
    decoded = decode_sampled_rgb(record["video_path"], frame_step)
    frames_per_clip = spec["frames_per_clip"]
    result: dict[str, np.ndarray] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for appearance in appearance_conditions:
            transformed = transform_clip(decoded, appearance)
            summed: np.ndarray | None = None
            count = 0
            for batch_start in range(0, len(window_rows), batch_size):
                batch_rows = window_rows[batch_start : batch_start + batch_size]
                clips = []
                for row in batch_rows:
                    start = int(row["window_sampled_start"])
                    stop = start + frames_per_clip
                    if stop > len(transformed):
                        raise ValueError(f"Window exceeds decoded frames for {row['window_id']}")
                    clips.append(
                        core.normalize_video(
                            torch.from_numpy(transformed[start:stop]),
                            spec["resolution"],
                        )
                    )
                normalized = torch.stack(clips)
                x = normalized.permute(0, 2, 1, 3, 4).contiguous().to("cuda")
                _, _, full_mask = core.token_masks(frames_per_clip, spec, len(batch_rows))
                capture.clear()
                encoded = encoder(x, full_mask)
                batch_features = capture.features(encoded)
                if not np.isfinite(batch_features).all():
                    raise RuntimeError(f"Non-finite features for {record['name']} {appearance}")
                summed = batch_features.sum(axis=0) if summed is None else summed + batch_features.sum(axis=0)
                count += batch_features.shape[0]
                del x, normalized, encoded, batch_features
            if count == 0:
                raise ValueError(f"No windows selected for {record['name']}")
            result[appearance] = (summed / count).astype(np.float32)
    return result


def metadata_row(record: dict, window_count: int, appearance: str) -> dict[str, object]:
    pair_id, label = construct.pair_identity(record)
    return {
        "sample_id": f"{record['name']}|{appearance}",
        "name": str(record["name"]),
        "file_name": str(record.get("file_name", "")),
        "pair_id": pair_id,
        "pair_label": label,
        "pair_label_value": 1 if label == "impossible" else 0,
        "appearance_condition": appearance,
        "condition": str(record.get("condition", "")),
        "difficulty": str(record.get("Difficulty", "")),
        "camera": str(record.get("Camera", "")),
        "window_count": int(window_count),
    }


def write_samples_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V-JEPA2 representation extraction")
    if args.output_dir.exists() and not args.overwrite:
        representations = args.output_dir / "representations.npz"
        if representations.exists():
            raise FileExistsError(f"{representations} exists; pass --overwrite to replace")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, selection = construct.selected_records(args)
    windows_by_video = select_window_rows(
        args.window_manifest,
        records,
        max_windows_per_video=args.max_windows_per_video,
    )
    frame_step = infer_frame_step(
        [row for rows in windows_by_video.values() for row in rows],
        args.frame_step,
    )
    appearance_conditions = tuple(args.appearance_condition or APPEARANCE_CONDITIONS)
    spec = core.MODEL_SPECS[args.model]
    load_started = time.perf_counter()
    encoder, load_info = load_encoder(args, spec)
    blocks, block_path = find_encoder_blocks(encoder)
    capture = LayerCapture(blocks)

    features: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    try:
        for index, record in enumerate(records):
            name = str(record["name"])
            video_features = extract_video_features(
                record,
                windows_by_video[name],
                frame_step=frame_step,
                appearance_conditions=appearance_conditions,
                spec=spec,
                encoder=encoder,
                capture=capture,
                batch_size=args.window_batch_size,
            )
            for appearance in appearance_conditions:
                features.append(video_features[appearance])
                rows.append(metadata_row(record, len(windows_by_video[name]), appearance))
            print(
                json.dumps(
                    {
                        "done": index + 1,
                        "total": len(records),
                        "name": name,
                        "windows": len(windows_by_video[name]),
                    }
                ),
                flush=True,
            )
    finally:
        capture.close()

    x = np.stack(features).astype(np.float32)
    layer_names = np.array([*capture.layer_names, "final"])
    metadata_columns = list(rows[0])
    arrays = {
        "X": x,
        "layer_names": layer_names,
        "metadata_columns": np.array(metadata_columns),
    }
    for column in metadata_columns:
        values = [row[column] for row in rows]
        if column in {"pair_label_value", "window_count"}:
            arrays[column] = np.asarray(values, dtype=np.int64)
        else:
            arrays[column] = np.asarray(values, dtype=str)
    temporary_npz = args.output_dir / "representations.npz.tmp"
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary_npz.replace(args.output_dir / "representations.npz")
    write_samples_csv(args.output_dir / "samples.csv", rows)

    provenance = {
        "schema_version": 1,
        "script": "scripts/extract_vjepa2_layerwise_representations.py",
        "script_sha256": core.sha256_file(Path(__file__)),
        "core_script_sha256": core.sha256_file(Path(core.__file__)),
        "construct_script_sha256": core.sha256_file(Path(construct.__file__)),
        "project_revision": project_revision(),
        "model": args.model,
        "model_spec": spec,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": core.sha256_file(args.checkpoint),
        "vjepa2_root": str(args.vjepa2_root),
        "vjepa2_revision": core.revision(args.vjepa2_root),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "selection_manifest": str(args.selection_manifest),
        "selection_manifest_sha256": core.sha256_file(args.selection_manifest),
        "window_manifest": str(args.window_manifest),
        "window_manifest_sha256": core.sha256_file(args.window_manifest),
        "selected_pairs": len(records) // 2,
        "selected_videos": len(records),
        "selection_pair_preserving_smoke_limit": args.max_pairs,
        "appearance_conditions": list(appearance_conditions),
        "frame_step": frame_step,
        "max_windows_per_video": args.max_windows_per_video,
        "window_batch_size": args.window_batch_size,
        "feature_pooling": "mean encoder tokens per layer, then mean selected windows per video/appearance",
        "encoder_block_path": block_path,
        "layer_names": layer_names.tolist(),
        "representations_shape": list(x.shape),
        "dtype": "bfloat16 autocast; stored float32",
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "seed": args.seed,
        "wall_seconds": time.perf_counter() - load_started,
        **load_info,
    }
    write_atomic_text(args.output_dir / "provenance.json", json.dumps(provenance, indent=2) + "\n")
    write_atomic_text(
        args.output_dir / "status.json",
        json.dumps(
            {
                "status": "complete",
                "samples": int(x.shape[0]),
                "layers": int(x.shape[1]),
                "feature_dim": int(x.shape[2]),
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
