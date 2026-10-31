#!/usr/bin/env python3
"""Generate diagnostic O2 brightening and mask-localized recoloring variants."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.appearance_interventions import (
    DEFAULT_SEVERITIES,
    _condition_directory,
    _copy_video_metadata,
    _hue_shift,
    _pair_videos,
    _rgb_array,
    _validate_source_pair,
    create_pair_contact_sheet,
    frame_paths,
    load_pairs,
    parameter_id,
    sha256_file,
)

HARDER_LEVELS: dict[str, dict[str, dict[str, float]]] = {
    "brightening": {
        "mild": {"factor": 1.15},
        "medium": {"factor": 1.35},
        "strong": {"factor": 1.60},
    },
    "foreground_recolor": {
        "mild": {"shift_degrees": 45.0},
        "medium": {"shift_degrees": 90.0},
        "strong": {"shift_degrees": 180.0},
    },
    "background_recolor": {
        "mild": {"shift_degrees": 45.0},
        "medium": {"shift_degrees": 90.0},
        "strong": {"shift_degrees": 180.0},
    },
}
HARDER_TRANSFORMS = tuple(HARDER_LEVELS)


def foreground_mask(instance_mask: np.ndarray) -> np.ndarray:
    """Select instance labels absent from every image border."""
    if instance_mask.ndim != 2:
        raise ValueError(f"Expected a grayscale instance mask, found {instance_mask.shape}")
    border = np.concatenate(
        (
            instance_mask[0],
            instance_mask[-1],
            instance_mask[1:-1, 0],
            instance_mask[1:-1, -1],
        )
    )
    return ~np.isin(instance_mask, np.unique(border))


def apply_harder_transform(
    rgb: np.ndarray,
    transform: str,
    parameters: Mapping[str, float],
    instance_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed RGB and the spatial region targeted by the transform."""
    if transform == "brightening":
        output = np.clip(
            np.rint(rgb.astype(np.float32) * float(parameters["factor"])), 0, 255
        ).astype(np.uint8)
        return output, np.ones(rgb.shape[:2], dtype=bool)

    if instance_mask is None or instance_mask.shape != rgb.shape[:2]:
        raise ValueError("Localized recoloring requires one aligned grayscale mask")
    foreground = foreground_mask(instance_mask)
    region = foreground if transform == "foreground_recolor" else ~foreground
    if transform not in ("foreground_recolor", "background_recolor"):
        raise ValueError(f"Unsupported transform: {transform}")
    recolored = _hue_shift(rgb, float(parameters["shift_degrees"]))
    output = rgb.copy()
    output[region] = recolored[region]
    return output, region


def _matching_mask(source: Path, scene_frame: Path) -> Path:
    suffix = scene_frame.stem.removeprefix("scene_")
    mask = source / "masks" / f"masks_{suffix}.png"
    if not mask.is_file():
        raise FileNotFoundError(mask)
    return mask


def transform_video(
    source: Path,
    destination: Path,
    transform: str,
    parameters: Mapping[str, float],
) -> dict[str, Any]:
    source_frames = frame_paths(source)
    _copy_video_metadata(source, destination)
    output_scene = destination / "scene"
    output_scene.mkdir()

    changed_frames = 0
    frames_with_target_region = 0
    changed_target_pixels = 0
    target_pixels = 0
    absolute_delta_sum = 0.0
    for source_frame in source_frames:
        with Image.open(source_frame) as image:
            source_rgb = _rgb_array(image, source_frame)
        instance_mask = None
        if transform != "brightening":
            mask_path = _matching_mask(source, source_frame)
            with Image.open(mask_path) as image:
                instance_mask = np.asarray(image)
        transformed, region = apply_harder_transform(
            source_rgb, transform, parameters, instance_mask
        )
        changed = np.any(transformed != source_rgb, axis=2)
        changed_frames += int(changed.any())
        frames_with_target_region += int(region.any())
        changed_target_pixels += int((changed & region).sum())
        target_pixels += int(region.sum())
        absolute_delta_sum += float(
            np.abs(source_rgb.astype(np.int16) - transformed.astype(np.int16)).mean()
        )
        Image.fromarray(transformed).save(
            output_scene / source_frame.name,
            format="PNG",
            optimize=False,
            compress_level=6,
        )

    output_frames = frame_paths(destination)
    if [path.name for path in output_frames] != [path.name for path in source_frames]:
        raise AssertionError(f"Frame names changed for {source}")
    status_source = source / "status.json"
    status_output = destination / "status.json"
    status_unchanged = (
        status_source.is_file()
        and status_output.is_file()
        and sha256_file(status_source) == sha256_file(status_output)
    )
    return {
        "source": str(source.resolve()),
        "output": str(destination.resolve()),
        "frame_count": len(source_frames),
        "frame_names_unchanged": True,
        "frame_shape": list(source_rgb.shape),
        "changed_frames": changed_frames,
        "frames_with_target_region": frames_with_target_region,
        "target_pixels": target_pixels,
        "changed_target_pixels": changed_target_pixels,
        "mean_absolute_pixel_delta": absolute_delta_sum / len(source_frames),
        "status_json_unchanged": status_unchanged,
        "parameter_id": parameter_id(parameters),
    }


def generate_harder_o2(
    dataset_root: Path,
    pairs_manifest: Path,
    output_root: Path,
    *,
    pair_limit: int = 60,
    contact_sheets: bool = True,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(pairs_manifest, pair_limit=pair_limit, blocks={"O2"})
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset_root": str(dataset_root.resolve()),
        "pairs_manifest": str(pairs_manifest.resolve()),
        "output_root": str(output_root.resolve()),
        "block": "O2",
        "pair_count": len(pairs),
        "transforms": list(HARDER_TRANSFORMS),
        "severities": list(DEFAULT_SEVERITIES),
        "conditions_per_pair": 9,
        "foreground_definition": "instance-mask labels absent from all image borders",
        "pairs": [],
    }
    total_videos = total_frames = total_changed_frames = 0
    for pair in pairs:
        _validate_source_pair(dataset_root, pair)
        pair_report: dict[str, Any] = {**pair, "parameters": {}, "conditions": []}
        for transform in HARDER_TRANSFORMS:
            pair_report["parameters"][transform] = HARDER_LEVELS[transform]
            for severity in DEFAULT_SEVERITIES:
                parameters = HARDER_LEVELS[transform][severity]
                fixed_id = parameter_id(parameters)
                video_reports = []
                for video_name, _ in _pair_videos(pair):
                    source = dataset_root / "O2" / str(pair["scene"]) / video_name
                    destination = _condition_directory(
                        output_root, transform, severity, pair, video_name
                    )
                    current = transform_video(
                        source, destination, transform, parameters
                    )
                    video_reports.append(current)
                    total_videos += 1
                    total_frames += int(current["frame_count"])
                    total_changed_frames += int(current["changed_frames"])
                    if not current["status_json_unchanged"]:
                        raise AssertionError(f"status.json changed for {source}")
                if {row["parameter_id"] for row in video_reports} != {fixed_id}:
                    raise AssertionError("Matched videos did not share one parameter set")
                pair_report["conditions"].append(
                    {
                        "transform": transform,
                        "severity": severity,
                        "parameters": parameters,
                        "parameter_id": fixed_id,
                        "videos": video_reports,
                    }
                )
        if contact_sheets:
            pair_report["contact_sheet"] = str(
                create_pair_contact_sheet(
                    dataset_root,
                    output_root,
                    pair,
                    HARDER_TRANSFORMS,
                    DEFAULT_SEVERITIES,
                ).resolve()
            )
        report["pairs"].append(pair_report)

    report["totals"] = {
        "transformed_videos": total_videos,
        "transformed_frames": total_frames,
        "changed_frames": total_changed_frames,
        "contact_sheets": len(pairs) if contact_sheets else 0,
    }
    report["all_integrity_checks_passed"] = True
    (output_root / "appearance_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pairs-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pair-limit", type=int, default=60)
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()
    report = generate_harder_o2(
        args.dataset_root,
        args.pairs_manifest,
        args.output_root,
        pair_limit=args.pair_limit,
        contact_sheets=not args.no_contact_sheets,
    )
    print(json.dumps({key: report[key] for key in ("output_root", "pair_count", "totals", "all_integrity_checks_passed")}, indent=2))


if __name__ == "__main__":
    main()
