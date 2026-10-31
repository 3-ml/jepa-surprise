from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sample_indices(num_frames: int, frames_per_clip: int) -> np.ndarray:
    if num_frames <= 0:
        return np.arange(frames_per_clip)
    if num_frames >= frames_per_clip:
        return np.linspace(0, num_frames - 1, frames_per_clip).round().astype(int)
    base = np.arange(num_frames)
    pad = np.full(frames_per_clip - num_frames, num_frames - 1, dtype=int)
    return np.concatenate([base, pad])


def read_video(path: Path, frames_per_clip: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    try:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_indices(num_frames, frames_per_clip)
        frames = []
        last = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                if last is None:
                    raise RuntimeError(f"Could not read frame {idx} from {path}")
                frame = last
            last = frame
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(frames, axis=0)
    finally:
        cap.release()


def row_key(row: dict[str, str]) -> str:
    return f"{row['split']}|{row['rgb_mp4']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pooled frozen V-JEPA2 features for Physion++ videos.")
    parser.add_argument("--manifest", default="data/physionpp/mass_only/manifest_mass_debug20.csv")
    parser.add_argument("--root", default="data/physionpp/mass_only")
    parser.add_argument("--output", default="data/physionpp/mass_only/features_vjepa2_debug20.npz")
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root = Path(args.root)
    rows = read_manifest(manifest_path)
    if args.max_videos is not None:
        rows = rows[: args.max_videos]

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    processor = AutoVideoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()

    keys, features, labels, splits, scenarios, pair_keys, rgb_paths = [], [], [], [], [], [], []
    with torch.no_grad():
        for i, row in enumerate(rows, start=1):
            video_path = root / row["rgb_mp4"]
            video = read_video(video_path, args.frames)
            inputs = processor(video, return_tensors="pt")
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            outputs = model(**inputs, skip_predictor=True)
            pooled = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().float().cpu().numpy()

            keys.append(row_key(row))
            features.append(pooled)
            labels.append(int(row["label"]))
            splits.append(row["split"])
            scenarios.append(row["scenario"])
            pair_keys.append(row.get("pair_key", ""))
            rgb_paths.append(row["rgb_mp4"])
            print(f"[{i}/{len(rows)}] {row['split']} {row['scenario']} {row['rgb_mp4']}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        row_key=np.array(keys),
        X=np.stack(features).astype(np.float32),
        y=np.array(labels, dtype=np.int64),
        split=np.array(splits),
        scenario=np.array(scenarios),
        pair_key=np.array(pair_keys),
        rgb_mp4=np.array(rgb_paths),
        model=np.array(args.model),
        frames=np.array(args.frames),
        manifest=np.array(str(manifest_path)),
    )
    print(f"Wrote {output} with shape {np.stack(features).shape}")


if __name__ == "__main__":
    main()
