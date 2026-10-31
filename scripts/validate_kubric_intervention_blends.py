#!/usr/bin/env python3
"""Validate paired Kubric condition .blend state invariants with bpy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rendered-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def principled(obj):
    if obj.type != "MESH" or not obj.data.materials:
        return None
    material = obj.data.materials[0]
    if not material.use_nodes or material.node_tree is None:
        return None
    nodes = [
        node
        for node in material.node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    ]
    return None if len(nodes) != 1 else nodes[0]


def snapshot(path: Path, frames: int = 48) -> dict:
    bpy.ops.wm.open_mainfile(filepath=str(path.resolve()))
    scene = bpy.context.scene
    object_names = sorted(
        obj.name for obj in bpy.data.objects if obj.type in {"MESH", "CAMERA", "LIGHT"}
    )
    transforms = {name: [] for name in object_names}
    base_colors = {name: [] for name in object_names}
    image_positions = {name: [] for name in object_names}
    material_other = {}
    for frame in range(1, frames + 1):
        scene.frame_set(frame)
        for name in object_names:
            obj = bpy.data.objects[name]
            transforms[name].append(
                [
                    *[float(value) for value in obj.location],
                    *[float(value) for value in obj.rotation_quaternion],
                    *[float(value) for value in obj.scale],
                    *[float(value) for value in obj.dimensions],
                ]
            )
            projected = world_to_camera_view(scene, scene.camera, obj.matrix_world.translation)
            image_positions[name].append([float(projected.x), float(projected.y), float(projected.z)])
            node = principled(obj)
            if node is None:
                base_colors[name].append(None)
            else:
                base_colors[name].append(
                    [float(value) for value in node.inputs["Base Color"].default_value]
                )
                if name not in material_other:
                    material_other[name] = {
                        socket.name: (
                            [float(value) for value in socket.default_value]
                            if hasattr(socket.default_value, "__len__")
                            else float(socket.default_value)
                        )
                        for socket in node.inputs
                        if socket.name != "Base Color"
                        and hasattr(socket, "default_value")
                    }
    return {
        "objects": object_names,
        "transforms": {
            name: np.asarray(rows, dtype=np.float64) for name, rows in transforms.items()
        },
        "base_colors": base_colors,
        "image_positions": image_positions,
        "material_other": material_other,
    }


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def main() -> None:
    args = parse_args()
    frozen = json.loads(args.manifest.read_text())
    event = int(frozen["event_blender_frame_1based"])
    rows = []
    for scene in frozen["scenes"]:
        scene_root = args.rendered_root / scene["scene_id"]
        clean = snapshot(scene_root / "clean" / "condition.blend")
        target = scene["target"]["blender_object_name"]
        for condition in frozen["conditions"]:
            if condition["name"] == "clean":
                continue
            observed = snapshot(scene_root / condition["name"] / "condition.blend")
            if observed["objects"] != clean["objects"]:
                raise ValueError(f"Object mismatch in {scene['scene_id']} {condition['name']}")
            non_target_transform_max = max(
                max_abs(clean["transforms"][name], observed["transforms"][name])
                for name in clean["objects"]
                if name != target
            )
            target_before_max = max_abs(
                clean["transforms"][target][: event - 1],
                observed["transforms"][target][: event - 1],
            )
            target_after_delta = (
                observed["transforms"][target][event - 1 :]
                - clean["transforms"][target][event - 1 :]
            )
            other_material_equal = observed["material_other"] == clean["material_other"]
            non_target_color_equal = all(
                observed["base_colors"][name] == clean["base_colors"][name]
                for name in clean["objects"]
                if name != target
            )
            target_color_before_equal = (
                observed["base_colors"][target][: event - 1]
                == clean["base_colors"][target][: event - 1]
            )
            event_projection = observed["image_positions"][target][event - 1]
            event_center_in_frame = (
                0.0 <= event_projection[0] <= 1.0
                and 0.0 <= event_projection[1] <= 1.0
                and event_projection[2] > 0.0
            )
            if condition["family"] == "appearance":
                target_transform_max = max_abs(
                    clean["transforms"][target], observed["transforms"][target]
                )
                passed = (
                    non_target_transform_max == 0
                    and target_transform_max == 0
                    and other_material_equal
                    and non_target_color_equal
                    and target_color_before_equal
                    and event_center_in_frame
                )
                detail = {"target_transform_max_abs": target_transform_max}
            else:
                expected = float(condition["dose"])
                manifest = json.loads(
                    (scene_root / condition["name"] / "manifest.json").read_text()
                )
                displacement = float(manifest["intervention"]["z_displacement_world"])
                expected_delta = np.zeros_like(target_after_delta)
                expected_delta[:, 2] = displacement
                target_after_max = max_abs(target_after_delta, expected_delta)
                target_color_equal = (
                    observed["base_colors"][target] == clean["base_colors"][target]
                )
                passed = (
                    non_target_transform_max == 0
                    and target_before_max == 0
                    and target_after_max < 1e-6
                    and other_material_equal
                    and non_target_color_equal
                    and target_color_equal
                    and event_center_in_frame
                    and abs(
                        displacement
                        / float(manifest["intervention"]["object_diameter_world"])
                        - expected
                    )
                    < 1e-6
                )
                detail = {
                    "target_after_declared_delta_max_abs": target_after_max,
                    "z_displacement_world": displacement,
                }
            row = {
                "scene_id": scene["scene_id"],
                "condition": condition["name"],
                "family": condition["family"],
                "passed": bool(passed),
                "non_target_transform_max_abs": non_target_transform_max,
                "target_pre_event_transform_max_abs": target_before_max,
                "other_material_fields_equal": other_material_equal,
                "non_target_base_colors_equal": non_target_color_equal,
                "target_pre_event_base_color_equal": target_color_before_equal,
                "target_event_projection_xyz": event_projection,
                "target_event_center_in_frame": event_center_in_frame,
                **detail,
            }
            rows.append(row)
            if not passed:
                raise RuntimeError(f"State validation failed: {row}")
            print(f"validated {scene['scene_id']} {condition['name']}", flush=True)
    payload = {
        "status": "complete",
        "conditions_validated": len(rows),
        "all_passed": all(row["passed"] for row in rows),
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
