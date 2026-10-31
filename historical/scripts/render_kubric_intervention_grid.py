from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender-python", type=Path, required=True)
    parser.add_argument("--renderer-script", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def valid_existing(path: Path, scene: dict, condition: dict, settings: dict) -> bool:
    try:
        payload = json.loads((path / "manifest.json").read_text())
        expected = {
            "status": "complete",
            "source_blend_sha256": scene["source_blend_sha256"],
            "condition": condition["renderer_condition"],
            "dose": condition["dose"],
            "target_object": scene["target"]["blender_object_name"],
            "event_frame": settings["event_blender_frame_1based"],
            "frame_start": 1,
            "frame_end": settings["frames"],
            "samples_per_pixel": settings["paired_samples_per_pixel"],
            "cycles_device_requested": settings["paired_cycles_device"],
            "denoiser": settings["paired_denoiser"],
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            return False
        checksums = payload["frame_checksums"]
        if len(checksums) != settings["frames"]:
            return False
        for row in checksums:
            frame = path / "frames" / row["path"]
            if not frame.is_file() or sha256_file(frame) != row["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    frozen = json.loads(args.manifest.read_text())
    settings = {
        key: frozen[key]
        for key in (
            "frames",
            "event_blender_frame_1based",
            "paired_samples_per_pixel",
            "paired_cycles_device",
            "paired_denoiser",
        )
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for scene in frozen["scenes"]:
        for condition in frozen["conditions"]:
            output = args.output_root / scene["scene_id"] / condition["name"]
            if valid_existing(output, scene, condition, settings):
                runs.append(
                    {
                        "scene_id": scene["scene_id"],
                        "condition": condition["name"],
                        "status": "validated_existing",
                    }
                )
                print(f"validated {scene['scene_id']} {condition['name']}", flush=True)
                continue
            if output.exists():
                raise ValueError(
                    f"Invalid pre-existing output requires diagnosis: {output}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(args.blender_python),
                str(args.renderer_script),
                "--blend",
                scene["source_blend"],
                "--output-dir",
                str(output),
                "--condition",
                condition["renderer_condition"],
                "--dose",
                str(condition["dose"]),
                "--target-object",
                scene["target"]["blender_object_name"],
                "--event-frame",
                str(settings["event_blender_frame_1based"]),
                "--samples-per-pixel",
                str(settings["paired_samples_per_pixel"]),
                "--cycles-device",
                settings["paired_cycles_device"],
                "--denoiser",
                settings["paired_denoiser"],
                "--frame-start",
                "1",
                "--frame-end",
                str(settings["frames"]),
            ]
            started = time.perf_counter()
            log_path = output.parent / f"{condition['name']}.log"
            with log_path.open("w") as log:
                subprocess.run(
                    command, check=True, stdout=log, stderr=subprocess.STDOUT
                )
            elapsed = time.perf_counter() - started
            if not valid_existing(output, scene, condition, settings):
                raise RuntimeError(f"New output failed validation: {output}")
            (output.parent / f"{condition['name']}.command.txt").write_text(
                " ".join(command) + "\n"
            )
            runs.append(
                {
                    "scene_id": scene["scene_id"],
                    "condition": condition["name"],
                    "status": "rendered_and_validated",
                    "wall_seconds": elapsed,
                }
            )
            print(
                f"rendered {scene['scene_id']} {condition['name']} {elapsed:.2f}s",
                flush=True,
            )
    payload = {
        "status": "complete",
        "frozen_manifest": str(args.manifest.resolve()),
        "frozen_manifest_sha256": sha256_file(args.manifest),
        "renderer_script": str(args.renderer_script.resolve()),
        "renderer_script_sha256": sha256_file(args.renderer_script),
        "settings": settings,
        "expected_outputs": len(frozen["scenes"]) * len(frozen["conditions"]),
        "runs": runs,
    }
    (args.output_root / "grid_run.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
