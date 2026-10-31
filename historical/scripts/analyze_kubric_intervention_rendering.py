from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rendered-root", type=Path, required=True)
    parser.add_argument("--blend-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_clip(path: Path) -> np.ndarray:
    files = sorted(path.glob("rgba_*.png"))
    if len(files) != 48:
        raise ValueError(f"Expected 48 RGBA frames in {path}, found {len(files)}")
    arrays = [np.asarray(Image.open(file)) for file in files]
    clip = np.stack(arrays)
    if clip.shape != (48, 256, 256, 4) or clip.dtype != np.uint8:
        raise ValueError(f"Unexpected clip shape/type: {clip.shape} {clip.dtype}")
    if not np.all(clip[..., 3] == 255):
        raise ValueError(f"Nonopaque frame in {path}")
    return clip[..., :3]


def effect_metrics(clean: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    difference = np.abs(observed.astype(np.int16) - clean.astype(np.int16)) / 255.0
    pixel_l1 = difference.mean(axis=-1)
    patches = pixel_l1.reshape(36, 16, 16, 16, 16).mean(axis=(2, 4))
    event = difference[0]
    return {
        "target_rgb_l1_01": float(difference.mean()),
        "target_rgb_rmse_01": float(np.sqrt(np.mean(difference**2))),
        "target_changed_pixel_fraction": float(np.any(difference > 0, axis=-1).mean()),
        "target_max_patch_rgb_l1_01": float(patches.max()),
        "event_rgb_l1_01": float(event.mean()),
        "event_rgb_rmse_01": float(np.sqrt(np.mean(event**2))),
        "event_changed_pixel_fraction": float(np.any(event > 0, axis=-1).mean()),
    }


def main() -> None:
    args = parse_args()
    frozen = json.loads(args.manifest.read_text())
    blend_validation = json.loads(args.blend_validation.read_text())
    if not blend_validation.get("all_passed"):
        raise ValueError("Blend state validation did not pass")
    blend_rows = {
        (row["scene_id"], row["condition"]): row for row in blend_validation["rows"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    exact_context = True
    maximum_pre_event_difference = 0
    for scene in frozen["scenes"]:
        scene_root = args.rendered_root / scene["scene_id"]
        clean = load_clip(scene_root / "clean" / "frames")
        clean_context_hashes = [
            sha256_file(scene_root / "clean" / "frames" / f"rgba_{index:05d}.png")
            for index in range(frozen["context_frames"])
        ]
        target_id = int(scene["target"]["instance_id"])
        segmentation_areas = []
        for index in range(48):
            segmentation = np.asarray(
                Image.open(Path(scene["base_output"]) / f"segmentation_{index:05d}.png")
            )
            segmentation_areas.append(int(np.count_nonzero(segmentation == target_id)))
        for condition in frozen["conditions"]:
            condition_root = scene_root / condition["name"]
            observed = load_clip(condition_root / "frames")
            context_hashes = [
                sha256_file(condition_root / "frames" / f"rgba_{index:05d}.png")
                for index in range(frozen["context_frames"])
            ]
            context_equal = context_hashes == clean_context_hashes and np.array_equal(
                observed[: frozen["context_frames"]], clean[: frozen["context_frames"]]
            )
            exact_context &= context_equal
            pre_event_difference = int(
                np.abs(
                    observed[: frozen["context_frames"]].astype(np.int16)
                    - clean[: frozen["context_frames"]].astype(np.int16)
                ).max()
            )
            maximum_pre_event_difference = max(
                maximum_pre_event_difference, pre_event_difference
            )
            metrics = effect_metrics(
                clean[frozen["context_frames"] :],
                observed[frozen["context_frames"] :],
            )
            state_row = blend_rows.get((scene["scene_id"], condition["name"]))
            if condition["name"] == "clean":
                state_passed = True
                event_projection = scene["target"]["centroid_frame_11_xy_01"]
            else:
                state_passed = bool(state_row["passed"])
                event_projection = state_row["target_event_projection_xyz"][:2]
            row = {
                "scene_id": scene["scene_id"],
                "seed": scene["seed"],
                "condition": condition["name"],
                "family": condition["family"],
                "dose": condition["dose"],
                "dose_unit": condition.get("dose_unit", "none"),
                "target_object": scene["target"]["blender_object_name"],
                "context_exact": context_equal,
                "state_validation_passed": state_passed,
                "target_clean_segmentation_area_event": segmentation_areas[
                    frozen["context_frames"]
                ],
                "target_clean_segmentation_area_min_target": min(
                    segmentation_areas[frozen["context_frames"] :]
                ),
                "target_event_projection_x": event_projection[0],
                "target_event_projection_y": event_projection[1],
                **metrics,
            }
            rows.append(row)
    if not exact_context or maximum_pre_event_difference != 0:
        raise RuntimeError("Paired contexts are not exactly identical")
    if not all(row["state_validation_passed"] for row in rows):
        raise RuntimeError("At least one condition failed state validation")
    fields = list(rows[0])
    with (args.output_dir / "render_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete",
        "scenes": len(frozen["scenes"]),
        "conditions_per_scene": len(frozen["conditions"]),
        "condition_outputs": len(rows),
        "frames_validated": len(rows) * 48,
        "exact_context_across_conditions": exact_context,
        "maximum_pre_event_uint8_difference": maximum_pre_event_difference,
        "all_blend_state_validations_passed": True,
        "all_values_finite": all(
            math.isfinite(value)
            for row in rows
            for key, value in row.items()
            if isinstance(value, float)
        ),
        "frozen_manifest_sha256": sha256_file(args.manifest),
        "blend_validation_sha256": sha256_file(args.blend_validation),
    }
    (args.output_dir / "render_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
