from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rendered-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    frozen = json.loads(args.manifest.read_text())
    context = int(frozen["context_frames"])
    records = []
    for scene in frozen["scenes"]:
        scene_root = args.rendered_root / scene["scene_id"]
        clean_root = scene_root / "clean"
        clean_manifest = json.loads((clean_root / "manifest.json").read_text())
        clean_hashes = {
            row["path"]: row["sha256"] for row in clean_manifest["frame_checksums"]
        }
        for condition in frozen["conditions"]:
            if condition["name"] == "clean":
                continue
            condition_root = scene_root / condition["name"]
            manifest_path = condition_root / "manifest.json"
            payload = json.loads(manifest_path.read_text())
            before = []
            for index in range(context):
                name = f"rgba_{index:05d}.png"
                source = clean_root / "frames" / name
                target = condition_root / "frames" / name
                before.append(sha256_file(target))
                shutil.copyfile(source, target)
                if sha256_file(target) != clean_hashes[name]:
                    raise RuntimeError(f"Context copy verification failed: {target}")
            for row in payload["frame_checksums"]:
                if int(row["frame"]) - int(payload["frame_start"]) < context:
                    row["sha256"] = clean_hashes[row["path"]]
            payload["context_canonicalization"] = {
                "status": "complete",
                "context_source_condition": "clean",
                "context_frames": context,
                "source_frame_sha256": [
                    clean_hashes[f"rgba_{index:05d}.png"] for index in range(context)
                ],
                "original_condition_frame_sha256": before,
                "reason": "Interventions begin after context; canonical copying removes measured sub-LSB OptiX rerender variation from predictor inputs.",
            }
            manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
            records.append(
                {"scene_id": scene["scene_id"], "condition": condition["name"]}
            )
    output = {
        "status": "complete",
        "frozen_manifest_sha256": sha256_file(args.manifest),
        "context_frames": context,
        "canonicalized_outputs": len(records),
        "records": records,
    }
    (args.rendered_root / "context_canonicalization.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps({key: output[key] for key in ("status", "canonicalized_outputs")}))


if __name__ == "__main__":
    main()
