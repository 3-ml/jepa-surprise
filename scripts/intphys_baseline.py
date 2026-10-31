#!/usr/bin/env python3
"""Audit IntPhys data and reproduce released V-JEPA surprise metrics."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BLOCKS = ("O1", "O2", "O3")
PAIRINGS = (
    (("1", "2"), ("3", "4")),
    (("1", "3"), ("2", "4")),
    (("1", "4"), ("2", "3")),
)


def _label(video_dir: Path) -> bool:
    with (video_dir / "status.json").open() as handle:
        return bool(json.load(handle)["header"]["is_possible"])


def _frames(video_dir: Path) -> list[Path]:
    return sorted((video_dir / "scene").glob("*.png"))


def _same_image(left: Path, right: Path) -> bool:
    if left.read_bytes() == right.read_bytes():
        return True
    with Image.open(left) as left_image, Image.open(right) as right_image:
        return np.array_equal(np.asarray(left_image), np.asarray(right_image))


def first_divergence(
    left: Path,
    right: Path,
    *,
    frame_step: int = 1,
    frame_count: int | None = None,
) -> int:
    stop = None if frame_count is None else frame_step * frame_count
    left_frames = _frames(left)[:stop:frame_step]
    right_frames = _frames(right)[:stop:frame_step]
    if [path.name for path in left_frames] != [path.name for path in right_frames]:
        raise ValueError(f"Frame names differ: {left} and {right}")
    for index, (left_frame, right_frame) in enumerate(zip(left_frames, right_frames)):
        if not _same_image(left_frame, right_frame):
            return index
    return len(left_frames)


def audit_dataset(
    root: Path,
    *,
    expected_scenes: int = 30,
    expected_frames: int = 100,
    evaluation_frame_step: int = 2,
    evaluation_frame_count: int = 49,
) -> tuple[dict, list[dict]]:
    manifest: list[dict] = []
    report = {"root": str(root.resolve()), "blocks": {}}

    for block in BLOCKS:
        block_dir = root / block
        scenes = sorted(path for path in block_dir.iterdir() if path.is_dir())
        if len(scenes) != expected_scenes:
            raise ValueError(f"{block}: expected {expected_scenes} scenes, found {len(scenes)}")

        for scene in scenes:
            videos = {name: scene / name for name in ("1", "2", "3", "4")}
            if not all(path.is_dir() for path in videos.values()):
                raise ValueError(f"{block}/{scene.name}: expected videos 1, 2, 3, and 4")
            labels = {name: _label(path) for name, path in videos.items()}
            if sum(labels.values()) != 2:
                raise ValueError(f"{block}/{scene.name}: expected two possible videos")
            for name, path in videos.items():
                frame_count = len(_frames(path))
                if frame_count != expected_frames:
                    raise ValueError(
                        f"{block}/{scene.name}/{name}: expected {expected_frames} frames, "
                        f"found {frame_count}"
                    )

            breaking_points = [
                first_divergence(
                    videos["1"],
                    videos[name],
                    frame_step=evaluation_frame_step,
                    frame_count=evaluation_frame_count,
                )
                for name in ("2", "3", "4")
            ]
            pairing_index = int(np.argmax(breaking_points))
            for pair_index, (left, right) in enumerate(PAIRINGS[pairing_index]):
                if labels[left] == labels[right]:
                    raise ValueError(
                        f"{block}/{scene.name}: inferred pair {left}-{right} has equal labels"
                    )
                manifest.append(
                    {
                        "block": block,
                        "scene": scene.name,
                        "pair": pair_index,
                        "video_a": left,
                        "video_a_possible": labels[left],
                        "video_b": right,
                        "video_b_possible": labels[right],
                        "breaking_points_vs_video_1": breaking_points,
                    }
                )

        report["blocks"][block] = {
            "scenes": len(scenes),
            "videos": len(scenes) * 4,
            "pairs": len(scenes) * 2,
            "frames_per_video": expected_frames,
        }

    report["totals"] = {
        "scenes": sum(value["scenes"] for value in report["blocks"].values()),
        "videos": sum(value["videos"] for value in report["blocks"].values()),
        "pairs": len(manifest),
    }
    report["evaluation_sampling"] = {
        "frame_step": evaluation_frame_step,
        "frame_count": evaluation_frame_count,
    }
    return report, manifest


def compute_official_pairwise_metrics(
    losses: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    """Reproduce the two pairwise metrics from upstream commit 302a2e8."""
    real = losses[labels == 1]
    impossible = losses[labels == 0]
    if real.shape != impossible.shape:
        raise ValueError("Possible and impossible losses are not paired")

    return {
        "Relative Accuracy (avg)": float(
            (real.mean(1) < impossible.mean(1)).float().mean() * 100
        ),
        "Relative Accuracy (max)": float(
            (real.max(1).values < impossible.max(1).values).float().mean() * 100
        ),
    }


def _load_archive_member(archive: tarfile.TarFile, suffix: str) -> bytes:
    matches = [member for member in archive.getmembers() if member.name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one archive member ending in {suffix!r}, found {len(matches)}")
    extracted = archive.extractfile(matches[0])
    if extracted is None:
        raise ValueError(f"Could not read {matches[0].name}")
    return extracted.read()


def verify_released_archive(archive_path: Path, model: str) -> dict:
    base = f"data_intphys/{model}/intphys"
    with tarfile.open(archive_path, "r:gz") as archive:
        expected_text = _load_archive_member(archive, f"{base}/performance.csv").decode()
        expected_rows = list(csv.DictReader(io.StringIO(expected_text), delimiter=";"))
        expected = {
            (row["Block"], row["Context length(s)"], int(row["Frame skip"])): row
            for row in expected_rows
        }

        actual: dict[tuple[str, str, int], dict[str, float]] = {}
        for block in BLOCKS:
            payload = _load_archive_member(
                archive, f"{base}/raw_surprises/{block}_16frames.pth"
            )
            data = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
            losses = data["losses"]
            labels = data["labels"]
            for index, context in enumerate(data["context_lengths"]):
                actual[(block, str(context), data["frame_step"])] = (
                    compute_official_pairwise_metrics(losses[:, index], labels)
                )
            actual[(block, "Filtered", data["frame_step"])] = (
                compute_official_pairwise_metrics(losses.min(1).values, labels)
            )

    failures = []
    metric_names = list(next(iter(actual.values())).keys())
    for key, metrics in actual.items():
        if key not in expected:
            failures.append({"row": key, "error": "missing expected row"})
            continue
        for metric_name in metric_names:
            difference = abs(metrics[metric_name] - float(expected[key][metric_name]))
            if difference > 5.1e-5:
                failures.append(
                    {
                        "row": key,
                        "metric": metric_name,
                        "expected": float(expected[key][metric_name]),
                        "actual": metrics[metric_name],
                        "difference": difference,
                    }
                )

    best_average = {}
    for block in BLOCKS:
        rows = [
            metrics["Relative Accuracy (avg)"]
            for (row_block, context, _), metrics in actual.items()
            if row_block == block and context != "Filtered"
        ]
        best_average[block] = max(rows)

    return {
        "archive": str(archive_path.resolve()),
        "model": model,
        "rows_checked": len(actual),
        "metrics_per_row": len(metric_names),
        "failures": failures,
        "best_pairwise_average_accuracy": best_average,
        "mean_best_pairwise_average_accuracy": sum(best_average.values()) / len(best_average),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("audit-data")
    data_parser.add_argument("--dataset-root", type=Path, required=True)
    data_parser.add_argument("--manifest", type=Path)
    data_parser.add_argument("--expected-scenes", type=int, default=30)
    data_parser.add_argument("--expected-frames", type=int, default=100)
    data_parser.add_argument("--evaluation-frame-step", type=int, default=2)
    data_parser.add_argument("--evaluation-frame-count", type=int, default=49)

    archive_parser = subparsers.add_parser("verify-archive")
    archive_parser.add_argument("--archive", type=Path, required=True)
    archive_parser.add_argument("--model", default="vit-h-rope-vm2m")

    args = parser.parse_args()
    if args.command == "audit-data":
        report, manifest = audit_dataset(
            args.dataset_root,
            expected_scenes=args.expected_scenes,
            expected_frames=args.expected_frames,
            evaluation_frame_step=args.evaluation_frame_step,
            evaluation_frame_count=args.evaluation_frame_count,
        )
        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
            report["manifest"] = str(args.manifest.resolve())
    else:
        report = verify_released_archive(args.archive, args.model)
        if report["failures"]:
            print(json.dumps(report, indent=2))
            raise SystemExit(1)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
