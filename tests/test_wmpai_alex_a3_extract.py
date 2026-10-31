import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "ex", Path(__file__).resolve().parents[1] / "scripts/wmpai2026/alex/a3_extract_fresh_scenes.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


def make_clip(root, clip_id, payload):
    d = root / clip_id
    d.mkdir(parents=True)
    for i, b in enumerate(payload):
        (d / f"rgba_{i:05d}.png").write_bytes(b)
    return ex.clip_input_sha256(sorted(d.glob("*.png")))


def rows_for(tmp_path):
    a = make_clip(tmp_path, "c1", [b"a", b"b"])
    b = make_clip(tmp_path, "c2", [b"c", b"d"])
    return [dict(clip_id="c1", scene_id="s1", split="development", input_sha256=a),
            dict(clip_id="c2", scene_id="s2", split="confirmatory", input_sha256=b)]


def test_plan_selects_only_requested_split(tmp_path):
    todo, problems = ex.plan(rows_for(tmp_path), tmp_path, "development")
    assert [r["clip_id"] for r in todo] == ["c1"] and problems == []


def test_plan_refuses_hash_mismatch_and_missing_frames(tmp_path):
    rows = rows_for(tmp_path)
    rows[0]["input_sha256"] = "0" * 64
    rows.append(dict(clip_id="ghost", scene_id="s9", split="development", input_sha256="x"))
    todo, problems = ex.plan(rows, tmp_path, "development")
    assert todo == [] and any("mismatch" in p for p in problems) and any("no frames" in p for p in problems)


def test_layers_must_be_explicit_and_known():
    with pytest.raises(ValueError):
        ex.check_layers(None, ["block_00"])
    with pytest.raises(ValueError):
        ex.check_layers(["block_99"], ["block_00"])
    assert ex.check_layers(["block_00"], ["block_00", "block_01"]) == ["block_00"]


def test_failed_clip_is_nan_not_zero_and_index_aligns(tmp_path):
    todo, _ = ex.plan(rows_for(tmp_path), tmp_path, "development")
    todo = todo + [dict(clip_id="c9", scene_id="s3", input_sha256="h")]
    feats = {"c1": np.ones((2, 4), np.float32), "c9": None}
    arr = ex.write_features(tmp_path / "out", todo, feats, ["block_20", "block_30"], "bfloat16", "ck")
    assert arr.shape == (2, 2, 4) and np.isnan(arr[1]).all() and (arr[0] == 1).all()
    idx = (tmp_path / "out/feature_index.csv").read_text().splitlines()
    assert idx[1].split(",")[1] == "c1" and idx[2].split(",")[4] == "1"
    assert json.load(open(tmp_path / "out/feature_meta.json"))["n_missing"] == 1
