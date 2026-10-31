import csv
import hashlib
import importlib.util
import json
import os
import pathlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1] / "scripts/wmpai2026/alex"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mk = _load("mk_dev24", "a3_make_dev24_table.py")
ex = _load("ex_dev24", "a3_extract_fresh_scenes.py")

IDS = ["dev24-001", "dev24-002"]
FRAMES = 48
WORLDS = ("solid", "gap")
BRANCHES = ("lawful", "inconsistent")
APPEARANCES = ("clean", "persistent_hue", "abrupt_event")


def write_configs(path, ids=IDS):
    scenes = [{"scene_id": cid, "sphere_radius": 0.5 + 0.1 * k, "nominal_contact_time": 1.1 + 0.05 * k,
               "camera_position": [-8.0 + k, -18.0, 5.0], "split": "development"}
              for k, cid in enumerate(ids)]
    path.write_text(json.dumps({"schema": 1, "scenes": scenes}))
    return path


def build_bundle(root, ids=IDS):
    """Tiny 8x8 PNG bundle: <root>/<cid>/<world>/manifest.json and 48-frame clips per branch and appearance."""
    counter = 0
    for cid in ids:
        for world in WORLDS:
            wdir = root / cid / world
            wdir.mkdir(parents=True)
            clips = []
            for branch in BRANCHES:
                for ap in APPEARANCES:
                    folder = Path("model-inputs") / branch / ap / "frames"
                    (wdir / folder).mkdir(parents=True)
                    rels, hashes = [], []
                    for i in range(FRAMES):
                        rel = (folder / f"rgba_{i:05d}.png").as_posix()
                        p = wdir / rel
                        arr = np.random.default_rng(counter).integers(0, 256, (8, 8, 3), dtype=np.uint8)
                        counter += 1
                        Image.fromarray(arr).save(p)
                        rels.append(rel)
                        hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
                    clips.append({"scene_id": cid, "physical_condition": branch, "appearance_condition": ap,
                                  "split": "development", "frames": FRAMES,
                                  "source_frame_paths": rels, "source_frame_sha256": hashes})
            (wdir / "manifest.json").write_text(json.dumps({"status": "pass", "clips": clips}))
    return root


def edit_manifest(root, cid, world, fn):
    p = root / cid / world / "manifest.json"
    doc = json.loads(p.read_text())
    fn(doc)
    p.write_text(json.dumps(doc))


def run(tmp_path, root, configs):
    out = tmp_path / "table.csv"
    rc = mk.main(["--root", str(root), "--configs", str(configs), "--out", str(out)])
    rows = list(csv.DictReader(open(out, newline="")))
    problems = json.loads((tmp_path / "problems.json").read_text())
    return rc, rows, problems, out


@pytest.fixture
def good(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    configs = write_configs(tmp_path / "configs.json")
    return tmp_path, root, configs


def test_full_table_has_24_rows_and_columns(good):
    tmp_path, root, configs = good
    rc, rows, problems, out = run(tmp_path, root, configs)
    assert rc == 0 and problems == []
    assert len(rows) == 24
    assert list(rows[0].keys()) == mk.COLUMNS
    assert all(r["split"] == "development" and r["n_frames"] == "48" for r in rows)
    assert all(r["scene_id"] == r["configuration_id"] for r in rows)


def test_plan_accepts_every_clip_of_the_produced_table(good):
    tmp_path, root, configs = good
    _, rows, _, out = run(tmp_path, root, configs)
    todo, problems = ex.plan(rows, root, "development")
    assert len(todo) == 24 and problems == []


def test_trajectory_mapping_for_both_worlds(good):
    tmp_path, root, configs = good
    _, rows, _, _ = run(tmp_path, root, configs)
    expect = {("solid", "lawful"): "stop", ("solid", "inconsistent"): "fall",
              ("gap", "lawful"): "fall", ("gap", "inconsistent"): "stop"}
    for r in rows:
        assert r["trajectory"] == expect[(r["world"], r["physical_condition"])]
    assert {r["world"] for r in rows} == {"solid", "gap"}


def test_factor_columns_match_configs_and_clip_ids_are_frame_folders(good):
    tmp_path, root, configs = good
    _, rows, _, _ = run(tmp_path, root, configs)
    cfgs = {s["scene_id"]: s for s in json.loads(configs.read_text())["scenes"]}
    for r in rows:
        c = cfgs[r["configuration_id"]]
        assert float(r["radius"]) == c["sphere_radius"]
        assert float(r["fall_time"]) == c["nominal_contact_time"]
        assert float(r["camera_x"]) == c["camera_position"][0]
        assert r["clip_id"] == f"{r['scene_id']}/{r['world']}/model-inputs/{r['physical_condition']}/" \
                               f"{r['appearance_condition']}/frames"


def test_corrupted_frame_byte_is_a_problem(good):
    tmp_path, root, configs = good
    p = root / "dev24-001" / "gap" / "model-inputs/lawful/clean/frames/rgba_00007.png"
    b = bytearray(p.read_bytes())
    b[-1] ^= 1
    p.write_bytes(bytes(b))
    rc, rows, problems, _ = run(tmp_path, root, configs)
    assert rc == 1 and len(rows) == 23
    assert len(problems) == 1 and "hash mismatch" in problems[0]


def test_missing_frame_is_a_problem(good):
    tmp_path, root, configs = good
    (root / "dev24-002" / "solid" / "model-inputs/inconsistent/abrupt_event/frames/rgba_00000.png").unlink()
    rc, rows, problems, _ = run(tmp_path, root, configs)
    assert rc == 1 and len(rows) == 23
    assert len(problems) == 1 and "missing frame" in problems[0]


def test_47_frame_clip_is_a_problem(good):
    tmp_path, root, configs = good

    def drop_last(doc):
        clip = doc["clips"][0]
        clip["source_frame_paths"].pop()
        clip["source_frame_sha256"].pop()

    edit_manifest(root, "dev24-001", "solid", drop_last)
    (root / "dev24-001" / "solid" / "model-inputs/lawful/clean/frames/rgba_00047.png").unlink()
    rc, rows, problems, _ = run(tmp_path, root, configs)
    assert rc == 1 and len(rows) == 23
    assert len(problems) == 1 and "47 frames listed" in problems[0]


def test_configuration_with_11_clips_is_a_problem(good):
    tmp_path, root, configs = good
    edit_manifest(root, "dev24-002", "gap", lambda doc: doc["clips"].pop())
    rc, rows, problems, _ = run(tmp_path, root, configs)
    assert rc == 1 and len(rows) == 23
    assert problems == ["dev24-002: 11 clips in the manifests, expected 12"]


def test_poisoned_file_is_never_opened(good, monkeypatch):
    tmp_path, root, configs = good
    poisoned = root / "scores" / "all-scores.json"
    poisoned.parent.mkdir()
    poisoned.write_text("{not json")
    opened = []
    real_read_bytes = pathlib.Path.read_bytes
    real_read_text = pathlib.Path.read_text

    def spy_bytes(self):
        if "score" in str(self).lower():
            opened.append(str(self))
        return real_read_bytes(self)

    def spy_text(self, *a, **k):
        if "score" in str(self).lower():
            opened.append(str(self))
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_bytes", spy_bytes)
    monkeypatch.setattr(pathlib.Path, "read_text", spy_text)
    mk.READ_LOG.clear()
    rc, rows, problems, _ = run(tmp_path, root, configs)
    assert rc == 0 and len(rows) == 24 and problems == []
    assert opened == []
    assert not any("all-scores" in str(p).lower() for p in mk.READ_LOG)


def test_bundle_root_with_score_in_its_name_builds(tmp_path):
    root = build_bundle(tmp_path / "score-blind-bundle")
    configs = write_configs(tmp_path / "configs.json")
    rc, rows, problems, _ = run(tmp_path, root, configs)
    assert rc == 0 and len(rows) == 24 and problems == []


def test_score_named_files_below_root_are_refused(tmp_path, monkeypatch):
    root = build_bundle(tmp_path / "bundle")
    (root / "scores").mkdir()
    refused = [root / "score-blind-notes.txt", root / "scores" / "all-scores.json",
               root / "scores" / "missing-scenes.json"]
    for p in refused:
        p.write_text("{}")
    monkeypatch.setattr(mk, "_BUNDLE_ROOT", Path(os.path.abspath(root)))
    for p in refused:
        with pytest.raises(PermissionError):
            mk.check_readable(p)
    # Outside the root the rule does not apply, and ordinary frames below the root stay readable.
    assert mk.check_readable(tmp_path / "configs.json")
    assert mk.check_readable(root / "dev24-001" / "solid" / "manifest.json")
