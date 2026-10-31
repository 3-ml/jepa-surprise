"""Build the A3 extractor input table from Vasi's development24 render bundle.

One CSV row per clip (288 clips for 24 configurations). The table is the input of
`a3_extract_fresh_scenes.plan(rows, frames_root=<bundle root>, split="development")`.

Layout assumed (checked against scripts/wmpai2026/vasi/run_development24.py and
score_development24.py on origin/wmpai/vasi-experiments; the scorer reads the same paths):
  <root>/<configuration_id>/<world>/manifest.json          world in solid, gap
  manifest["clips"][k]["source_frame_paths"]               48 paths relative to <root>/<configuration_id>/<world>/
  manifest["clips"][k]["source_frame_sha256"]              per-frame SHA-256, same order as the paths
  manifest["clips"][k]["physical_condition"]               lawful or inconsistent
  manifest["clips"][k]["appearance_condition"]             clean, persistent_hue or abrupt_event
  clip frame folder = the parent folder of the listed frames (model-inputs/<branch>/<appearance>/frames
  after canonicalization); clip_id = its path relative to --root, forward slashes.
Trajectory: solid lawful = stop, solid inconsistent = fall; gap lawful = fall, gap inconsistent = stop.
Factors from the configurations file: radius = sphere_radius, fall_time = nominal_contact_time,
camera_x = camera_position[0].

Only manifest.json, the listed frame PNGs and the configurations file are read. Below the bundle root, a
read is refused (PermissionError) when a path component starts with "score", equals "scores", or contains
"all-scores" (case-insensitive). The bundle root's own name is not tested, and files outside the root,
such as the configurations file, are not restricted by this rule. The bundle is not walked.

Usage:
  python a3_make_dev24_table.py --root <bundle root> --configs development24-configs.json --out table.csv

Writes table.csv (good clips only) and problems.json next to it. Exit code 1 if any problem was found.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("a3_extract_fresh_scenes", HERE / "a3_extract_fresh_scenes.py")
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)

FRAMES = 48
WORLDS = ("solid", "gap")
BRANCHES = ("lawful", "inconsistent")
APPEARANCES = ("clean", "persistent_hue", "abrupt_event")
CLIPS_PER_CONFIG = len(WORLDS) * len(BRANCHES) * len(APPEARANCES)
TRAJECTORY = {("solid", "lawful"): "stop", ("solid", "inconsistent"): "fall",
              ("gap", "lawful"): "fall", ("gap", "inconsistent"): "stop"}
MANIFEST_NAME = "manifest.json"
PATHS_KEY = "source_frame_paths"
HASHES_KEY = "source_frame_sha256"
BRANCH_KEY = "physical_condition"
APPEARANCE_KEY = "appearance_condition"
COLUMNS = ["clip_id", "scene_id", "configuration_id", "world", "physical_condition", "trajectory",
           "appearance_condition", "split", "input_sha256", "n_frames", "radius", "fall_time", "camera_x"]

READ_LOG = []  # every path this script has read; tests inspect it
_BUNDLE_ROOT = None  # absolute bundle root while build() runs


def check_readable(path):
    """Refuse score-related reads below the bundle root. Every read goes through here.

    Only the part of the path relative to the bundle root is tested (lower-cased): a component
    starting with "score", equal to "scores", or containing "all-scores" is refused. The root's
    own name is not tested, and paths outside the root are not restricted by this rule.
    """
    p = Path(os.path.abspath(path))
    if _BUNDLE_ROOT is not None:
        try:
            rel = p.relative_to(_BUNDLE_ROOT)
        except ValueError:
            rel = None
        if rel is not None:
            for part in rel.parts:
                low = part.lower()
                if low.startswith("score") or low == "scores" or "all-scores" in low:
                    raise PermissionError(f"refusing to open a score-related path: {rel.as_posix()}")
    return p


def guarded_read_bytes(path):
    p = check_readable(path)
    READ_LOG.append(p)
    return p.read_bytes()


def factors(cfg):
    return {"radius": cfg["sphere_radius"], "fall_time": cfg["nominal_contact_time"],
            "camera_x": cfg["camera_position"][0]}


def clip_row(root, world_dir, cfg, world, clip):
    """One table row for one manifest clip. Raises ValueError with a readable reason on any problem."""
    cid = cfg["scene_id"]
    branch, appearance = clip.get(BRANCH_KEY), clip.get(APPEARANCE_KEY)
    if (world, branch) not in TRAJECTORY:
        raise ValueError(f"unknown physical_condition {branch!r}")
    if appearance not in APPEARANCES:
        raise ValueError(f"unknown appearance_condition {appearance!r}")
    if clip.get("scene_id", cid) != cid:
        raise ValueError(f"manifest scene_id {clip.get('scene_id')!r} differs from configuration {cid}")
    rel_paths = clip.get(PATHS_KEY)
    if not isinstance(rel_paths, list) or len(rel_paths) != FRAMES:
        n = len(rel_paths) if isinstance(rel_paths, list) else 0
        raise ValueError(f"{n} frames listed, expected {FRAMES}")
    hashes = clip.get(HASHES_KEY)
    if hashes is not None and len(hashes) != FRAMES:
        raise ValueError(f"{len(hashes)} frame hashes listed, expected {FRAMES}")
    for rel in rel_paths:
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError(f"frame path escapes the world folder: {rel}")
    folder_rel = {Path(rel).parent.as_posix() for rel in rel_paths}
    if len(folder_rel) != 1:
        raise ValueError("listed frames are not all in one folder")
    folder_rel = folder_rel.pop()
    files = [world_dir / rel for rel in rel_paths]
    for rel, f in zip(rel_paths, files):
        check_readable(f)
        if not f.is_file():
            raise ValueError(f"missing frame {rel}")
    listed_names = {f.name for f in files}
    extra = sorted(p.name for p in (world_dir / folder_rel).glob("*.png") if p.name not in listed_names)
    if extra:
        raise ValueError(f"{len(extra)} PNG(s) in the frame folder are not listed, e.g. {extra[0]}")
    if hashes is not None:
        for rel, f, want in zip(rel_paths, files, hashes):
            if hashlib.sha256(guarded_read_bytes(f)).hexdigest() != want:
                raise ValueError(f"frame hash mismatch: {rel}")
    input_sha = ex.clip_input_sha256(files)  # reads the listed files again; all are checked above
    return {"clip_id": f"{cid}/{world}/{folder_rel}", "scene_id": cid, "configuration_id": cid,
            "world": world, "physical_condition": branch, "trajectory": TRAJECTORY[(world, branch)],
            "appearance_condition": appearance, "split": "development", "input_sha256": input_sha,
            "n_frames": len(files), **factors(cfg)}


def world_rows(root, cfg, world, problems):
    """Rows for one configuration and world, and the number of clips its manifest lists."""
    cid = cfg["scene_id"]
    world_dir = root / cid / world
    try:
        manifest = json.loads(guarded_read_bytes(world_dir / MANIFEST_NAME).decode("utf-8"))
        clips = manifest["clips"]
        if not isinstance(clips, list):
            raise ValueError("'clips' is not a list")
    except (OSError, ValueError, KeyError, TypeError) as e:
        problems.append(f"{cid}/{world}: cannot read manifest ({e})")
        return [], 0
    rows = []
    for k, clip in enumerate(clips):
        try:
            rows.append(clip_row(root, world_dir, cfg, world, clip))
        except (OSError, ValueError, TypeError) as e:
            problems.append(f"{cid}/{world}/clip[{k}]: {e}")
    return rows, len(clips)


def build(root, configs_path, out_csv):
    global _BUNDLE_ROOT
    _BUNDLE_ROOT = Path(os.path.abspath(root))
    try:
        return _build(root, configs_path, out_csv)
    finally:
        _BUNDLE_ROOT = None


def _build(root, configs_path, out_csv):
    root = Path(root)
    doc = json.loads(guarded_read_bytes(configs_path).decode("utf-8"))
    problems, rows = [], []
    for cfg in doc["scenes"]:
        cid = cfg["scene_id"]
        listed = 0
        for world in WORLDS:
            r, n = world_rows(root, cfg, world, problems)
            rows.extend(r)
            listed += n
        if listed != CLIPS_PER_CONFIG:
            problems.append(f"{cid}: {listed} clips in the manifests, expected {CLIPS_PER_CONFIG}")
    out_csv = Path(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(out_csv.with_name("problems.json"), "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)
    return rows, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="development24 bundle root")
    ap.add_argument("--configs", required=True, help="protocol/vasi/development24-configs.json")
    ap.add_argument("--out", required=True, help="output CSV")
    a = ap.parse_args(argv)
    rows, problems = build(a.root, a.configs, a.out)
    for p in problems:
        print("PROBLEM", p)
    print(f"{len(rows)} clips written to {a.out}; {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
