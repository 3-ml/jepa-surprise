from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

EVENT_SOURCE_INDEX = 12
EVENT_BLENDER_FRAME = 13
CONDITIONS = [
    {"name": "clean", "family": "clean", "renderer_condition": "clean", "dose": 0.0},
    *[
        {
            "name": f"hue-{degrees:03d}",
            "family": "appearance",
            "renderer_condition": "hue",
            "dose": float(degrees),
            "dose_unit": "degrees",
        }
        for degrees in (30, 60, 120, 180)
    ],
    *[
        {
            "name": f"physical-z-{str(dose).replace('.', 'p')}",
            "family": "physical",
            "renderer_condition": "physical-z",
            "dose": dose,
            "dose_unit": "object_diameters",
        }
        for dose in (0.25, 0.5, 1.0, 1.5)
    ],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-output",
        type=Path,
        action="append",
        required=True,
        help="Official MOVi worker output directory; repeat for each scene.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def blender_object_name(instance: dict) -> str:
    fields = (
        instance.get("size_label"),
        instance.get("color_label"),
        instance.get("material"),
        instance.get("shape") or instance.get("asset_id"),
    )
    if not all(isinstance(value, str) and value for value in fields):
        raise ValueError(f"Cannot derive Blender object name from instance: {fields}")
    return " ".join(fields)


def load_segmentation(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.shape != (256, 256) or array.dtype != np.uint8:
        raise ValueError(f"Unexpected segmentation {path}: {array.shape} {array.dtype}")
    return array


def validate_passes(base: Path) -> dict:
    patterns = {
        "rgba": "rgba_*.png",
        "segmentation": "segmentation_*.png",
        "depth": "depth_*.tiff",
        "forward_flow": "forward_flow_*.png",
        "backward_flow": "backward_flow_*.png",
    }
    expected = {name: 48 for name in patterns}
    observed = {
        name: len(list(base.glob(pattern))) for name, pattern in patterns.items()
    }
    if observed != expected:
        raise ValueError(f"Incomplete render passes in {base}: {observed}")
    rgba = np.asarray(Image.open(base / "rgba_00000.png"))
    if rgba.shape != (256, 256, 4) or rgba.dtype != np.uint8:
        raise ValueError(
            f"Unexpected RGBA shape/type in {base}: {rgba.shape} {rgba.dtype}"
        )
    depth = np.asarray(Image.open(base / "depth_00000.tiff"))
    if (
        depth.shape != (256, 256)
        or not np.isfinite(depth).all()
        or float(depth.min()) < 0
    ):
        raise ValueError(f"Invalid depth pass in {base}")
    for required in (
        "scene.blend",
        "scene.bullet",
        "metadata.json",
        "events.json",
        "data_ranges.json",
    ):
        if not (base / required).is_file():
            raise ValueError(f"Missing {required} in {base}")
    return observed


def select_target(base: Path, metadata: dict) -> tuple[int, dict, dict]:
    segmentations = {
        index: load_segmentation(base / f"segmentation_{index:05d}.png")
        for index in (10, 11, 12)
    }
    candidates = []
    for instance_id, instance in enumerate(metadata["instances"], start=1):
        areas = {
            index: int(np.count_nonzero(array == instance_id))
            for index, array in segmentations.items()
        }
        centroid = np.asarray(instance["image_positions"][11], dtype=np.float64)
        qualifies = (
            all(area >= 655 for area in areas.values())
            and bool(np.all((centroid >= 0.1) & (centroid <= 0.9)))
            and areas[12] > 0
        )
        if qualifies:
            candidates.append((areas[11], instance_id, instance, areas, centroid))
    if not candidates:
        raise ValueError(f"No target satisfies the locked visibility rule in {base}")
    _, instance_id, instance, areas, centroid = max(
        candidates, key=lambda item: item[0]
    )
    target = {
        "instance_id": instance_id,
        "asset_id": instance["asset_id"],
        "blender_object_name": blender_object_name(instance),
        "segmentation_area": {str(key): value for key, value in areas.items()},
        "centroid_frame_11_xy_01": centroid.tolist(),
        "size_label": instance["size_label"],
        "color_label": instance["color_label"],
        "material": instance["material"],
        "shape": instance["shape"],
    }
    return instance_id, instance, target


def prepare_scene(base: Path) -> dict:
    base = base.resolve()
    observed_passes = validate_passes(base)
    metadata = json.loads((base / "metadata.json").read_text())
    scene_metadata = metadata["metadata"]
    if scene_metadata["num_frames"] != 48 or scene_metadata["resolution"] != [256, 256]:
        raise ValueError(f"Unexpected scene dimensions in {base}: {scene_metadata}")
    if scene_metadata["num_instances"] != 3:
        raise ValueError(f"Expected three visible instances in {base}")
    _, _, target = select_target(base, metadata)
    return {
        "scene_id": f"seed-{int(scene_metadata['seed'])}",
        "seed": int(scene_metadata["seed"]),
        "base_output": str(base),
        "source_blend": str(base / "scene.blend"),
        "source_blend_sha256": sha256_file(base / "scene.blend"),
        "metadata_sha256": sha256_file(base / "metadata.json"),
        "events_sha256": sha256_file(base / "events.json"),
        "observed_passes": observed_passes,
        "target": target,
    }


def main() -> None:
    args = parse_args()
    scenes = [prepare_scene(path) for path in args.base_output]
    if len({scene["seed"] for scene in scenes}) != len(scenes):
        raise ValueError("Base scenes contain duplicate seeds")
    payload = {
        "schema_version": 1,
        "status": "targets_frozen_before_intervention_rendering",
        "kubric_revision": "61f2422c84bab75006df33c6989e0b483db3ccfe",
        "variant": "MOVi-A",
        "frames": 48,
        "resolution": [256, 256],
        "context_frames": 12,
        "event_source_index_0based": EVENT_SOURCE_INDEX,
        "event_blender_frame_1based": EVENT_BLENDER_FRAME,
        "paired_samples_per_pixel": 64,
        "paired_cycles_device": "optix",
        "paired_denoiser": "none",
        "conditions": CONDITIONS,
        "scenes": scenes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "scenes": len(scenes)}))


if __name__ == "__main__":
    main()
