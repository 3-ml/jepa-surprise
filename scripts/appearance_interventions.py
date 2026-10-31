#!/usr/bin/env python3
"""Create deterministic appearance-only counterfactuals for matched IntPhys pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_LEVELS: dict[str, dict[str, dict[str, float | int]]] = {
    "hue": {
        "mild": {"shift_degrees": 15.0},
        "medium": {"shift_degrees": 45.0},
        "strong": {"shift_degrees": 90.0},
    },
    "brightness": {
        "mild": {"factor": 0.85},
        "medium": {"factor": 0.65},
        "strong": {"factor": 0.45},
    },
    "selective_recolor": {
        "mild": {"shift_degrees": 45.0},
        "medium": {"shift_degrees": 90.0},
        "strong": {"shift_degrees": 180.0},
    },
}
DEFAULT_TRANSFORMS = tuple(DEFAULT_LEVELS)
DEFAULT_SEVERITIES = ("mild", "medium", "strong")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def parameter_id(parameters: Mapping[str, Any]) -> str:
    """Return a compact stable identifier for one fixed parameter set."""
    return hashlib.sha256(_canonical_json(parameters).encode()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_paths(video_dir: Path) -> list[Path]:
    frames = sorted((video_dir / "scene").glob("*.png"))
    if not frames:
        raise ValueError(f"No PNG frames found under {video_dir / 'scene'}")
    return frames


def _rgb_array(image: Image.Image, source: Path | None = None) -> np.ndarray:
    if image.mode != "RGB":
        location = f" in {source}" if source else ""
        raise ValueError(f"Expected RGB input{location}, found mode {image.mode!r}")
    return np.asarray(image, dtype=np.uint8)


def _hsv_array(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.uint8)


def _rgb_from_hsv(hsv: np.ndarray) -> np.ndarray:
    height, width = hsv.shape[:2]
    hsv_image = Image.frombytes("HSV", (width, height), hsv.tobytes())
    return np.asarray(hsv_image.convert("RGB"), dtype=np.uint8)


def _hue_shift(rgb: np.ndarray, shift_degrees: float) -> np.ndarray:
    hsv = _hsv_array(rgb).copy()
    shift = int(round(shift_degrees * 256.0 / 360.0))
    hsv[..., 0] = (hsv[..., 0].astype(np.uint16) + shift) % 256
    return _rgb_from_hsv(hsv)


def _brightness(rgb: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(np.rint(rgb.astype(np.float32) * factor), 0, 255).astype(np.uint8)


def derive_selective_recolor_base(first_frames: Sequence[Path]) -> dict[str, float | int]:
    """Infer one shared saturated hue band from both videos' first frames."""
    hue_samples: list[np.ndarray] = []
    minimum_saturation = 64
    for frame in first_frames:
        with Image.open(frame) as image:
            hsv = _hsv_array(_rgb_array(image, frame))
        selected = hsv[..., 0][hsv[..., 1] >= minimum_saturation]
        if selected.size:
            hue_samples.append(selected)

    if not hue_samples:
        raise ValueError(
            "Selective recoloring is infeasible: the paired anchor frames contain no "
            "sufficiently saturated pixels"
        )

    hues = np.concatenate(hue_samples).astype(np.int32)
    bin_count = 36
    bins = np.minimum(hues * bin_count // 256, bin_count - 1)
    dominant_bin = int(np.argmax(np.bincount(bins, minlength=bin_count)))
    return {
        "source_hue_center_degrees": (dominant_bin + 0.5) * 360.0 / bin_count,
        "hue_half_width_degrees": 20.0,
        "minimum_saturation": minimum_saturation,
    }


def _selective_recolor(
    rgb: np.ndarray,
    *,
    source_hue_center_degrees: float,
    hue_half_width_degrees: float,
    minimum_saturation: int,
    shift_degrees: float,
) -> np.ndarray:
    hsv = _hsv_array(rgb).copy()
    hue_degrees = hsv[..., 0].astype(np.float32) * (360.0 / 256.0)
    hue_distance = np.abs(
        (hue_degrees - source_hue_center_degrees + 180.0) % 360.0 - 180.0
    )
    mask = (hue_distance <= hue_half_width_degrees) & (
        hsv[..., 1] >= minimum_saturation
    )
    shift = int(round(shift_degrees * 256.0 / 360.0))
    shifted = (hsv[..., 0].astype(np.uint16) + shift) % 256
    hsv[..., 0][mask] = shifted[mask]
    return _rgb_from_hsv(hsv)


def apply_transform(
    rgb: np.ndarray, transform: str, parameters: Mapping[str, float | int]
) -> np.ndarray:
    if transform == "hue":
        return _hue_shift(rgb, float(parameters["shift_degrees"]))
    if transform == "brightness":
        return _brightness(rgb, float(parameters["factor"]))
    if transform == "selective_recolor":
        return _selective_recolor(
            rgb,
            source_hue_center_degrees=float(parameters["source_hue_center_degrees"]),
            hue_half_width_degrees=float(parameters["hue_half_width_degrees"]),
            minimum_saturation=int(parameters["minimum_saturation"]),
            shift_degrees=float(parameters["shift_degrees"]),
        )
    raise ValueError(f"Unsupported transform: {transform}")


def _copy_video_metadata(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for child in source.iterdir():
        if child.name == "scene":
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def transform_video(
    source: Path,
    destination: Path,
    transform: str,
    parameters: Mapping[str, float | int],
) -> dict[str, Any]:
    source_frames = frame_paths(source)
    _copy_video_metadata(source, destination)
    output_scene = destination / "scene"
    output_scene.mkdir()

    changed_frames = 0
    absolute_delta_sum = 0.0
    source_shape: tuple[int, int, int] | None = None
    for source_frame in source_frames:
        with Image.open(source_frame) as image:
            source_rgb = _rgb_array(image, source_frame)
            if source_shape is None:
                source_shape = source_rgb.shape
            elif source_rgb.shape != source_shape:
                raise ValueError(f"Frame shape changed within {source}")
            transformed = apply_transform(source_rgb, transform, parameters)

        if transformed.shape != source_rgb.shape:
            raise AssertionError("A photometric transform changed frame geometry")
        changed_frames += int(not np.array_equal(source_rgb, transformed))
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
    if len(output_frames) != len(source_frames):
        raise AssertionError(f"Frame count changed for {source}")

    status_source = source / "status.json"
    status_output = destination / "status.json"
    status_unchanged = status_source.exists() and status_output.exists()
    if status_unchanged:
        status_unchanged = sha256_file(status_source) == sha256_file(status_output)

    return {
        "source": str(source.resolve()),
        "output": str(destination.resolve()),
        "frame_count": len(source_frames),
        "frame_names_unchanged": True,
        "frame_shape": list(source_shape or ()),
        "changed_frames": changed_frames,
        "mean_absolute_pixel_delta": absolute_delta_sum / len(source_frames),
        "status_json_unchanged": status_unchanged,
        "parameter_id": parameter_id(parameters),
    }


def load_pairs(
    manifest_path: Path,
    *,
    pair_limit: int,
    blocks: set[str] | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Pair manifest must be a JSON list")

    required = {
        "block",
        "scene",
        "pair",
        "video_a",
        "video_a_possible",
        "video_b",
        "video_b_possible",
    }
    selected: list[dict[str, Any]] = []
    identities: set[tuple[str, str, int]] = set()
    for raw in payload:
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise ValueError(f"Malformed pair-manifest row: {raw!r}")
        if blocks is not None and raw["block"] not in blocks:
            continue
        if bool(raw["video_a_possible"]) == bool(raw["video_b_possible"]):
            raise ValueError(f"Pair does not contain one possible and one impossible video: {raw}")
        identity = (str(raw["block"]), str(raw["scene"]), int(raw["pair"]))
        if identity in identities:
            raise ValueError(f"Duplicate pair identity: {identity}")
        identities.add(identity)
        selected.append(dict(raw))
        if len(selected) == pair_limit:
            break

    if len(selected) != pair_limit:
        raise ValueError(f"Requested {pair_limit} pairs, found {len(selected)}")
    return selected


def _pair_videos(pair: Mapping[str, Any]) -> tuple[tuple[str, bool], tuple[str, bool]]:
    return (
        (str(pair["video_a"]), bool(pair["video_a_possible"])),
        (str(pair["video_b"]), bool(pair["video_b_possible"])),
    )


def _validate_source_pair(dataset_root: Path, pair: Mapping[str, Any]) -> None:
    left, right = _pair_videos(pair)
    left_dir = dataset_root / str(pair["block"]) / str(pair["scene"]) / left[0]
    right_dir = dataset_root / str(pair["block"]) / str(pair["scene"]) / right[0]
    left_frames = frame_paths(left_dir)
    right_frames = frame_paths(right_dir)
    if [path.name for path in left_frames] != [path.name for path in right_frames]:
        raise ValueError(f"Paired frame names differ: {left_dir} and {right_dir}")
    for video_dir, expected_possible in ((left_dir, left[1]), (right_dir, right[1])):
        status_path = video_dir / "status.json"
        if not status_path.is_file():
            raise ValueError(f"Missing status.json: {status_path}")
        actual_possible = bool(json.loads(status_path.read_text())["header"]["is_possible"])
        if actual_possible != expected_possible:
            raise ValueError(f"Manifest label disagrees with {status_path}")


def _parameters_for_pair(
    dataset_root: Path,
    pair: Mapping[str, Any],
    transforms: Sequence[str],
    severities: Sequence[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    parameters: dict[str, dict[str, dict[str, float | int]]] = {}
    recolor_base: dict[str, float | int] | None = None
    if "selective_recolor" in transforms:
        videos = _pair_videos(pair)
        first_frames = [
            frame_paths(
                dataset_root
                / str(pair["block"])
                / str(pair["scene"])
                / video_name
            )[0]
            for video_name, _ in videos
        ]
        recolor_base = derive_selective_recolor_base(first_frames)

    for transform in transforms:
        parameters[transform] = {}
        for severity in severities:
            current = dict(DEFAULT_LEVELS[transform][severity])
            if transform == "selective_recolor":
                assert recolor_base is not None
                current = {**recolor_base, **current}
            parameters[transform][severity] = current
    return parameters


def _condition_directory(
    output_root: Path,
    transform: str,
    severity: str,
    pair: Mapping[str, Any],
    video_name: str,
) -> Path:
    return (
        output_root
        / transform
        / severity
        / str(pair["block"])
        / str(pair["scene"])
        / video_name
    )


def _paste_thumbnail(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    thumbnail = image.copy()
    thumbnail.thumbnail((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    x = x0 + (x1 - x0 - thumbnail.width) // 2
    y = y0 + (y1 - y0 - thumbnail.height) // 2
    canvas.paste(thumbnail, (x, y))


def create_pair_contact_sheet(
    dataset_root: Path,
    output_root: Path,
    pair: Mapping[str, Any],
    transforms: Sequence[str],
    severities: Sequence[str],
) -> Path:
    conditions: list[tuple[str, Path | None]] = [("clean", None)]
    conditions.extend(
        (f"{transform}-{severity}", output_root / transform / severity)
        for transform in transforms
        for severity in severities
    )
    videos = sorted(_pair_videos(pair), key=lambda item: not item[1])
    first_video = (
        dataset_root / str(pair["block"]) / str(pair["scene"]) / videos[0][0]
    )
    names = [path.name for path in frame_paths(first_video)]
    indices = sorted({0, len(names) // 2, len(names) - 1})

    label_width, cell_width, cell_height, header_height = 150, 180, 125, 42
    width = label_width + cell_width * len(conditions)
    height = header_height + cell_height * len(videos) * len(indices)
    canvas = Image.new("RGB", (width, height), "#202020")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for column, (label, _) in enumerate(conditions):
        draw.text((label_width + column * cell_width + 5, 12), label, fill="white", font=font)

    row = 0
    for video_name, possible in videos:
        for index in indices:
            y0 = header_height + row * cell_height
            label = f"{'possible' if possible else 'impossible'} t={index:03d}"
            draw.text((5, y0 + cell_height // 2), label, fill="white", font=font)
            for column, (_, condition_root) in enumerate(conditions):
                root = dataset_root if condition_root is None else condition_root
                frame = (
                    root
                    / str(pair["block"])
                    / str(pair["scene"])
                    / video_name
                    / "scene"
                    / names[index]
                )
                with Image.open(frame) as image:
                    rgb = image.convert("RGB")
                    _paste_thumbnail(
                        canvas,
                        rgb,
                        (
                            label_width + column * cell_width,
                            y0,
                            label_width + (column + 1) * cell_width,
                            y0 + cell_height,
                        ),
                    )
            row += 1

    verification_dir = output_root / "verification"
    verification_dir.mkdir(exist_ok=True)
    destination = verification_dir / (
        f"{pair['block']}-{pair['scene']}-pair-{int(pair['pair'])}.png"
    )
    canvas.save(destination, format="PNG", optimize=False, compress_level=6)
    return destination


def create_appearance_pilot(
    dataset_root: Path,
    pairs_manifest: Path,
    output_root: Path,
    *,
    pair_limit: int = 20,
    blocks: Iterable[str] | None = None,
    transforms: Sequence[str] = DEFAULT_TRANSFORMS,
    severities: Sequence[str] = DEFAULT_SEVERITIES,
    contact_sheets: bool = True,
) -> dict[str, Any]:
    if pair_limit < 1:
        raise ValueError("pair_limit must be at least 1")
    unknown_transforms = set(transforms) - set(DEFAULT_LEVELS)
    if unknown_transforms:
        raise ValueError(f"Unknown transforms: {sorted(unknown_transforms)}")
    unknown_severities = set(severities) - set(DEFAULT_SEVERITIES)
    if unknown_severities:
        raise ValueError(f"Unknown severities: {sorted(unknown_severities)}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    selected = load_pairs(
        pairs_manifest,
        pair_limit=pair_limit,
        blocks=None if blocks is None else set(blocks),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset_root": str(dataset_root.resolve()),
        "pairs_manifest": str(pairs_manifest.resolve()),
        "output_root": str(output_root.resolve()),
        "pair_count": len(selected),
        "transforms": list(transforms),
        "severities": list(severities),
        "conditions_per_pair": len(transforms) * len(severities),
        "pairs": [],
    }

    total_videos = 0
    total_frames = 0
    all_integrity_checks_passed = True
    for pair in selected:
        _validate_source_pair(dataset_root, pair)
        pair_parameters = _parameters_for_pair(
            dataset_root, pair, transforms, severities
        )
        pair_report: dict[str, Any] = {
            "block": pair["block"],
            "scene": pair["scene"],
            "pair": pair["pair"],
            "video_a": pair["video_a"],
            "video_a_possible": pair["video_a_possible"],
            "video_b": pair["video_b"],
            "video_b_possible": pair["video_b_possible"],
            "parameters": pair_parameters,
            "conditions": [],
        }
        for transform in transforms:
            for severity in severities:
                parameters = pair_parameters[transform][severity]
                fixed_parameter_id = parameter_id(parameters)
                video_reports = []
                for video_name, _ in _pair_videos(pair):
                    source = (
                        dataset_root
                        / str(pair["block"])
                        / str(pair["scene"])
                        / video_name
                    )
                    destination = _condition_directory(
                        output_root, transform, severity, pair, video_name
                    )
                    video_report = transform_video(
                        source, destination, transform, parameters
                    )
                    video_reports.append(video_report)
                    total_videos += 1
                    total_frames += int(video_report["frame_count"])
                    all_integrity_checks_passed &= bool(
                        video_report["frame_names_unchanged"]
                        and video_report["status_json_unchanged"]
                        and video_report["parameter_id"] == fixed_parameter_id
                    )

                if {row["parameter_id"] for row in video_reports} != {fixed_parameter_id}:
                    raise AssertionError("Matched videos did not share one parameter set")
                pair_report["conditions"].append(
                    {
                        "transform": transform,
                        "severity": severity,
                        "parameters": parameters,
                        "parameter_id": fixed_parameter_id,
                        "videos": video_reports,
                    }
                )

        if contact_sheets:
            pair_report["contact_sheet"] = str(
                create_pair_contact_sheet(
                    dataset_root,
                    output_root,
                    pair,
                    transforms,
                    severities,
                ).resolve()
            )
        report["pairs"].append(pair_report)

    report["totals"] = {
        "transformed_videos": total_videos,
        "transformed_frames": total_frames,
        "contact_sheets": len(selected) if contact_sheets else 0,
    }
    report["all_integrity_checks_passed"] = all_integrity_checks_passed
    manifest_path = output_root / "appearance_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic, temporally consistent appearance transformations "
            "for matched IntPhys possible/impossible pairs."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pairs-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pair-limit", type=int, default=20)
    parser.add_argument("--blocks", nargs="+")
    parser.add_argument(
        "--transforms",
        nargs="+",
        choices=DEFAULT_TRANSFORMS,
        default=list(DEFAULT_TRANSFORMS),
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        choices=DEFAULT_SEVERITIES,
        default=list(DEFAULT_SEVERITIES),
    )
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()

    report = create_appearance_pilot(
        args.dataset_root,
        args.pairs_manifest,
        args.output_root,
        pair_limit=args.pair_limit,
        blocks=args.blocks,
        transforms=args.transforms,
        severities=args.severities,
        contact_sheets=not args.no_contact_sheets,
    )
    print(
        json.dumps(
            {
                "output_root": report["output_root"],
                "pair_count": report["pair_count"],
                "totals": report["totals"],
                "all_integrity_checks_passed": report[
                    "all_integrity_checks_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
