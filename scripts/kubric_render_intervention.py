#!/usr/bin/env python3
"""Render one state or appearance intervention from a keyed Kubric .blend file.

Run inside the pinned Kubric/Blender container; this script imports bpy directly.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--condition", choices=["clean", "physical-z", "hue"], required=True
    )
    parser.add_argument("--target-object", required=True)
    parser.add_argument("--event-frame", type=int, default=24)
    parser.add_argument("--dose", type=float, default=0.0)
    parser.add_argument("--samples-per-pixel", type=int, default=16)
    parser.add_argument(
        "--cycles-device", choices=["cpu", "cuda", "optix"], default="cpu"
    )
    parser.add_argument(
        "--denoiser", choices=["openimage", "optix", "none"], default="openimage"
    )
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def object_trajectory(obj, frame_start: int, frame_end: int):
    scene = bpy.context.scene
    rows = []
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        rows.append(
            {
                "frame": frame,
                "location": [float(value) for value in obj.location],
                "rotation_quaternion": [float(value) for value in obj.rotation_quaternion],
                "dimensions": [float(value) for value in obj.dimensions],
            }
        )
    return rows


def scene_state_hash(frame_start: int, frame_end: int) -> str:
    scene = bpy.context.scene
    rows = []
    objects = sorted(
        [obj for obj in bpy.data.objects if obj.type == "MESH"], key=lambda obj: obj.name
    )
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        rows.append(
            [
                frame,
                [
                    [
                        obj.name,
                        [float(value) for value in obj.location],
                        [float(value) for value in obj.rotation_quaternion],
                        [float(value) for value in obj.scale],
                    ]
                    for obj in objects
                ],
            ]
        )
    payload = json.dumps(rows, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def principled_base_color_socket(obj):
    if not hasattr(obj.data, "materials") or not obj.data.materials:
        raise ValueError(f"Target {obj.name!r} has no material")
    material = obj.data.materials[0]
    if not material.use_nodes or material.node_tree is None:
        raise ValueError(f"Target material {material.name!r} does not use nodes")
    nodes = [
        node
        for node in material.node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    ]
    if len(nodes) != 1:
        raise ValueError(f"Expected one Principled BSDF node, found {len(nodes)}")
    return material, nodes[0].inputs["Base Color"]


def rotate_hue(rgb, degrees: float):
    hue, saturation, value = colorsys.rgb_to_hsv(*rgb)
    hue = (hue + degrees / 360.0) % 1.0
    return colorsys.hsv_to_rgb(hue, saturation, value)


def apply_hue(obj, event_frame: int, degrees: float, frame_end: int):
    material, socket = principled_base_color_socket(obj)
    original = tuple(float(value) for value in socket.default_value)
    transformed_rgb = rotate_hue(original[:3], degrees)
    transformed = (*transformed_rgb, original[3])
    socket.default_value = original
    socket.keyframe_insert("default_value", frame=event_frame - 1)
    socket.default_value = transformed
    socket.keyframe_insert("default_value", frame=event_frame)
    socket.keyframe_insert("default_value", frame=frame_end)
    action = material.node_tree.animation_data.action
    for fcurve in action.fcurves:
        for point in fcurve.keyframe_points:
            point.interpolation = "CONSTANT"
    return {
        "material": material.name,
        "original_rgba": original,
        "transformed_rgba": transformed,
        "hue_degrees": degrees,
    }


def apply_physical_z(obj, event_frame: int, diameters: float):
    scene = bpy.context.scene
    scene.frame_set(event_frame - 1)
    diameter = max(float(value) for value in obj.dimensions)
    displacement = diameters * diameter
    if obj.animation_data is None or obj.animation_data.action is None:
        raise ValueError(f"Target {obj.name!r} has no keyed action")
    changed = 0
    for fcurve in obj.animation_data.action.fcurves:
        if fcurve.data_path == "location" and fcurve.array_index == 2:
            for point in fcurve.keyframe_points:
                if point.co[0] >= event_frame:
                    point.co[1] += displacement
                    point.handle_left[1] += displacement
                    point.handle_right[1] += displacement
                    changed += 1
    if changed == 0:
        raise ValueError("No post-event z-location keyframes were changed")
    return {
        "object_diameter_world": diameter,
        "dose_object_diameters": diameters,
        "z_displacement_world": displacement,
        "changed_keyframes": changed,
        "semantics": "keyframed displacement persisted over the remaining base trajectory; no post-intervention PyBullet simulation",
    }


def configure_cycles_device(scene, requested: str):
    if requested == "cpu":
        scene.cycles.device = "CPU"
        return [{"name": "CPU", "type": "CPU", "use": True}]
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = requested.upper()
    preferences.get_devices()
    enabled = []
    for device in preferences.devices:
        use = device.type == requested.upper()
        device.use = use
        enabled.append({"name": device.name, "type": device.type, "use": bool(use)})
    if not any(item["use"] for item in enabled):
        raise RuntimeError(f"No {requested.upper()} Cycles device is available: {enabled}")
    scene.cycles.device = "GPU"
    return enabled


def disable_auxiliary_outputs(scene):
    changed = {"muted_file_nodes": [], "disabled_view_layers": []}
    if scene.use_nodes and scene.node_tree is not None:
        for node in scene.node_tree.nodes:
            if node.bl_idname == "CompositorNodeOutputFile":
                node.mute = True
                changed["muted_file_nodes"].append(node.name)
    for view_layer in scene.view_layers:
        if view_layer.name != "ViewLayer":
            view_layer.use = False
            changed["disabled_view_layers"].append(view_layer.name)
    return changed


def render(
    scene,
    output_dir: Path,
    frame_start: int,
    frame_end: int,
    samples: int,
    denoiser: str,
):
    output_dir.mkdir(parents=True, exist_ok=False)
    scene.cycles.samples = samples
    scene.cycles.use_denoising = denoiser != "none"
    if denoiser != "none":
        scene.cycles.denoiser = {
            "openimage": "OPENIMAGEDENOISE",
            "optix": "OPTIX",
        }[denoiser]
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    disabled_outputs = disable_auxiliary_outputs(scene)
    checksums = []
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        path = output_dir / f"rgba_{frame - frame_start:05d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        checksums.append(
            {"frame": frame, "path": path.name, "sha256": sha256_file(path)}
        )
    return checksums, disabled_outputs


def main() -> None:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    scene = bpy.context.scene
    frame_start = scene.frame_start if args.frame_start is None else args.frame_start
    frame_end = scene.frame_end if args.frame_end is None else args.frame_end
    if not frame_start <= args.event_frame <= frame_end:
        raise ValueError("event-frame must lie inside rendered frame range")
    if args.target_object not in bpy.data.objects:
        raise ValueError(f"Unknown object {args.target_object!r}")
    target = bpy.data.objects[args.target_object]

    before_hash = scene_state_hash(frame_start, frame_end)
    before_target = object_trajectory(target, frame_start, frame_end)
    intervention = {"condition": args.condition, "dose": args.dose}
    if args.condition == "hue":
        intervention.update(apply_hue(target, args.event_frame, args.dose, frame_end))
    elif args.condition == "physical-z":
        intervention.update(apply_physical_z(target, args.event_frame, args.dose))
    after_hash = scene_state_hash(frame_start, frame_end)
    after_target = object_trajectory(target, frame_start, frame_end)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = args.output_dir / "frames"
    cycles_devices = configure_cycles_device(scene, args.cycles_device)
    frame_checksums, disabled_outputs = render(
        scene,
        frames_dir,
        frame_start,
        frame_end,
        args.samples_per_pixel,
        args.denoiser,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_dir / "condition.blend"))
    manifest = {
        "status": "complete",
        "source_blend": str(args.blend.resolve()),
        "source_blend_sha256": sha256_file(args.blend),
        "condition": args.condition,
        "dose": args.dose,
        "target_object": args.target_object,
        "event_frame": args.event_frame,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "samples_per_pixel": args.samples_per_pixel,
        "cycles_device_requested": args.cycles_device,
        "cycles_devices": cycles_devices,
        "denoiser": args.denoiser,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "blender_version": bpy.app.version_string,
        "state_hash_before": before_hash,
        "state_hash_after": after_hash,
        "state_unchanged": before_hash == after_hash,
        "target_trajectory_before": before_target,
        "target_trajectory_after": after_target,
        "intervention": intervention,
        "disabled_auxiliary_outputs": disabled_outputs,
        "frame_checksums": frame_checksums,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ["status", "condition", "dose", "state_unchanged"]}))


if __name__ == "__main__":
    main()
