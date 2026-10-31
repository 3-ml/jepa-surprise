'Build the extractor input table for the eight-scene Kubric pilot frames (exploratory regression fixture).\n\nOne row per rendered clip: clip_id is the frame folder relative to the frames root, split is marked\n`development` only because the extractor needs a split label; these scenes are not fresh data.\n\nUsage: python a3_make_pilot8_table.py --frames-root ${JEPA_PILOT8_FRAMES} --out table.csv\n'
import argparse
import csv
from pathlib import Path

from a3_extract_fresh_scenes import clip_input_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.frames_root)
    rows = []
    for scene in sorted(p for p in root.iterdir() if p.is_dir()):
        for cond in sorted(p for p in scene.iterdir() if p.is_dir()):
            frames = sorted((cond / "frames").glob("*.png"))
            if len(frames) != 48:
                raise SystemExit(f"{cond}: {len(frames)} frames")
            rows.append({"clip_id": f"{scene.name}/{cond.name}/frames", "scene_id": scene.name,
                         "condition": cond.name, "split": "development",
                         "input_sha256": clip_input_sha256(frames)})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "clips")


if __name__ == "__main__":
    main()
