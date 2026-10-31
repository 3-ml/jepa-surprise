import csv
import hashlib
import importlib.util
import json
import os
from itertools import product
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "scripts/wmpai2026/alex"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AUDIT = _load("a4_audit_development24_test", "a4_audit_development24.py")

WORLDS = ("solid", "gap")
BRANCHES = ("lawful", "inconsistent")
APPEARANCES = ("clean", "persistent_hue", "abrupt_event")
SIM_HZ, FPS, FRAMES = 240, 12, 48
N_STEPS = (FRAMES - 1) * SIM_HZ // FPS + 1  # 941 states, one per simulation substep
D_DEFAULT = 284  # first divergence step, as in the real first-solid/gap audits
BODY_IDS = {"sphere": 100, "support": 101, "catch_floor": 102, "left_rail": 103, "right_rail": 104}
SUPPORT_ID, RAIL_ID = BODY_IDS["support"], BODY_IDS["left_rail"]
CONFIGS = [
    {"scene_id": "dev24-t01", "frames": FRAMES, "fps": FPS, "simulation_hz": SIM_HZ},
    {"scene_id": "dev24-t02", "frames": FRAMES, "fps": FPS, "simulation_hz": SIM_HZ},
]
REAL_AUDIT_DIR = Path(os.environ.get("A4_REAL_AUDIT_DIR", "."))


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _common(s):
    return [0.0, -0.3, 7.5 - 0.01 * s]


def _stop(s, d):
    return _common(s) if s < d else [0.0, -0.3, 0.5]


def _fall(s, d):
    return _common(s) if s < d else [0.0, -0.3, -2.0 - 0.001 * s]


def _contact(other_id, force):
    """PyBullet-style contact tuple: index 2 other body id, index 9 normal force."""
    return [0, BODY_IDS["sphere"], other_id, -1, -1, 0.0, 0.0, 0.0, 0.0, force]


def _states_bytes(world, branch, d, nudge=0.0, ops=frozenset()):
    stops = (branch == "lawful") == (world == "solid")
    trace = []
    for s in range(N_STEPS):
        pos = list(_stop(s, d) if stops else _fall(s, d))
        if stops and s >= d:
            pos[2] += nudge
        contacts = []
        if stops and s >= d and "no_support" not in ops:
            contacts.append(_contact(SUPPORT_ID, 5.0))
        if not stops and s >= d and "fall_support" in ops:
            contacts.append(_contact(SUPPORT_ID, 5.0))
        if "rail" in ops and s >= d + 10:
            contacts.append(_contact(RAIL_ID, 1.0))
        trace.append({"step": s, "time_seconds": s / SIM_HZ, "position": pos,
                      "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                      "linear_velocity": [0.0, 0.0, 0.0], "angular_velocity": [0.0, 0.0, 0.0],
                      "contacts": contacts})
    return json.dumps({"branch": branch, "body_ids": BODY_IDS, "trace": trace}).encode()


def _color(branch, appearance, i):
    if i < AUDIT.CONTEXT_FRAMES:
        green = 200 if appearance == "persistent_hue" else 40
        return ((10 * i) % 256, green, 90)
    bi, ai = BRANCHES.index(branch), APPEARANCES.index(appearance)
    return ((3 * i + 50 * bi) % 256, 100 + 40 * ai, 20 + 10 * bi)


def build_bundle(root, divergence_steps=None, nudge=None, ops=None):
    """Synthetic bundle in the layout documented in the audit script (8x8 PNG frames).

    ops maps (scene, world, branch) to contact mutations: "rail", "no_support", "fall_support".
    """
    root = Path(root)
    divergence_steps = divergence_steps or {}
    ops = ops or {}
    for config in CONFIGS:
        scene = config["scene_id"]
        d = divergence_steps.get(scene, D_DEFAULT)
        for world in WORLDS:
            wdir = root / scene / world
            wdir.mkdir(parents=True)
            state_sha = {}
            for branch in BRANCHES:
                key = (scene, world, branch)
                nd = nudge[3] if nudge and nudge[:3] == key else 0.0
                data = _states_bytes(world, branch, d, nd, frozenset(ops.get(key, ())))
                (wdir / AUDIT.STATES_NAME.format(branch=branch)).write_bytes(data)
                state_sha[branch] = _sha_bytes(data)
            clips = []
            for branch, appearance in product(BRANCHES, APPEARANCES):
                hashes = []
                for i in range(FRAMES):
                    rel = f"{branch}/{appearance}/frames/rgba_{i:05d}.png"
                    path = wdir / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8), _color(branch, appearance, i)).save(path, format="PNG")
                    hashes.append(_sha_bytes(path.read_bytes()))
                clip = {
                    "scene_id": scene, "split": "development",
                    "physical_condition": branch, "appearance_condition": appearance,
                    "frames": FRAMES, "context_frames": AUDIT.CONTEXT_FRAMES, "event_frame_0based": 12,
                    "state_trace_path": AUDIT.STATES_NAME.format(branch=branch),
                    "state_trace_sha256": state_sha[branch],
                    "source_frame_paths": [f"{branch}/{appearance}/frames/rgba_{i:05d}.png"
                                           for i in range(FRAMES)],
                    "source_frame_sha256": hashes,
                    "target_area_pixels_per_frame": [4000] * FRAMES,
                    "target_area_min_target_fraction": 4000 / 65536,
                }
                clips.append(clip)
                clip_path = wdir / AUDIT.CLIP_NAME.format(branch=branch, appearance=appearance)
                clip_path.parent.mkdir(parents=True, exist_ok=True)
                clip_path.write_text(json.dumps(clip), encoding="utf-8")
            (wdir / AUDIT.MANIFEST_NAME).write_text(
                json.dumps({"status": "pass", "clips": clips, "clip_count": 6}), encoding="utf-8")
            canon = [{"branch": b, "appearance": a, "canonical_reference": "lawful/clean",
                      "raw_context_vs_reference": {"max_uint8": 0, "normalized_mean_abs": 0.0,
                                                   "changed_pixel_fraction": 0.0},
                      "future_bytes_unchanged": True}
                     for b, a in product(BRANCHES, APPEARANCES)]
            (wdir / AUDIT.CANON_NAME).write_text(json.dumps(canon), encoding="utf-8")
            (wdir / AUDIT.STATE_AUDIT_NAME).write_text(json.dumps(
                {"status": "pass", "world": world, "first_position_divergence_step": d}), encoding="utf-8")
    return root


def _configs_file(tmp_path):
    path = tmp_path / "development24-configs.json"
    path.write_text(json.dumps({"schema": 1, "status": "synthetic", "scenes": CONFIGS}), encoding="utf-8")
    return path


def _run(tmp_path, root, check_frames=False, name="audit"):
    out = tmp_path / name
    code = AUDIT.run(root, _configs_file(tmp_path), out, check_frames)
    summary = json.loads((out / "audit-summary.json").read_text(encoding="utf-8"))
    return code, summary, out


def _failed_checks(summary):
    return {check for scene in summary["per_configuration"].values()
            for check, result in scene.items() if result["status"] == "fail"}


def _failing_scenes(summary, check):
    return {scene for scene, checks in summary["per_configuration"].items()
            if checks[check]["status"] == "fail"}


def _edit_manifest(root, scene, world, fn):
    path = root / scene / world / AUDIT.MANIFEST_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    fn(data["clips"])
    path.write_text(json.dumps(data), encoding="utf-8")


def _clip(clips, branch, appearance):
    return next(c for c in clips if c["physical_condition"] == branch and c["appearance_condition"] == appearance)


def _file_sha(path):
    return _sha_bytes(Path(path).read_bytes())


def test_clean_bundle_passes_every_hard_check(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    code, summary, out = _run(tmp_path, root)
    assert code == 0
    assert summary["hard_check_failed"] is False
    for scene in ("dev24-t01", "dev24-t02"):
        checks = summary["per_configuration"][scene]
        for check in ("C1", "C2", "C3", "C4", "C5", "C6", "C8", "C9", "C10"):
            assert checks[check]["status"] == "pass", (scene, check, checks[check])
        assert checks["C10"]["notes"] == []
        assert checks["C6"]["per_world"]["solid"]["first_divergence_frame"] == pytest.approx(14.2)
        assert checks["C10"]["per_world"]["solid"]["first_support_contact_step"] == 284
    assert summary["visibility_totals"] == {"clips": 24, "zero_frame": 0, "below_164": 0, "below_256": 0}
    with (out / "audit-by-clip.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24 and all(row["present"] == "True" for row in rows)


def test_clean_bundle_passes_with_frame_hash_check(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    code, summary, _ = _run(tmp_path, root, check_frames=True)
    assert code == 0
    assert summary["per_configuration"]["dev24-t01"]["C2"]["status"] == "pass"


def test_missing_clip_triggers_only_c1(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    _edit_manifest(root, "dev24-t01", "solid",
                   lambda clips: clips.remove(_clip(clips, "lawful", "clean")))
    code, summary, _ = _run(tmp_path, root)
    assert code == 1
    assert _failed_checks(summary) == {"C1"}
    failures = summary["per_configuration"]["dev24-t01"]["C1"]["failures"]
    assert any("missing clip lawful/clean" in f and "solid" in f for f in failures)


def test_wrong_state_hash_triggers_only_c2(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    path = root / "dev24-t01" / "solid" / "lawful-states.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["trace"][5]["quaternion_xyzw"] = [0.0, 0.0, 1.0, 0.0]
    path.write_text(json.dumps(data), encoding="utf-8")
    code, summary, _ = _run(tmp_path, root)
    assert code == 1
    assert _failed_checks(summary) == {"C2"}
    assert _failing_scenes(summary, "C2") == {"dev24-t01"}


def test_context_frame_changed_in_one_branch_triggers_only_c3(tmp_path):
    root = build_bundle(tmp_path / "bundle")

    def tamper(clips):
        _clip(clips, "lawful", "clean")["source_frame_sha256"][5] = "0" * 64

    _edit_manifest(root, "dev24-t01", "solid", tamper)
    code, summary, _ = _run(tmp_path, root)
    assert code == 1
    assert _failed_checks(summary) == {"C3"}


def test_persistent_hue_context_equal_to_clean_triggers_only_c3(tmp_path):
    root = build_bundle(tmp_path / "bundle")

    def tamper(clips):
        clean = _clip(clips, "lawful", "clean")["source_frame_sha256"]
        _clip(clips, "lawful", "persistent_hue")["source_frame_sha256"][:12] = clean[:12]

    _edit_manifest(root, "dev24-t02", "gap", tamper)
    code, summary, _ = _run(tmp_path, root)
    assert code == 1
    assert _failed_checks(summary) == {"C3"}
    assert _failing_scenes(summary, "C3") == {"dev24-t02"}


def test_state_trace_differing_across_appearances_triggers_only_c4(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    other = root / "dev24-t01" / "solid" / "inconsistent-states.json"

    def tamper(clips):
        clip = _clip(clips, "lawful", "abrupt_event")
        clip["state_trace_path"] = "inconsistent-states.json"
        clip["state_trace_sha256"] = _file_sha(other)

    _edit_manifest(root, "dev24-t01", "solid", tamper)
    code, summary, _ = _run(tmp_path, root)
    assert code == 1
    assert _failed_checks(summary) == {"C4"}


def test_stop_trajectories_differing_across_worlds_triggers_only_c5(tmp_path):
    root = build_bundle(tmp_path / "bundle", nudge=("dev24-t02", "solid", "lawful", 1e-6))
    code, summary, _ = _run(tmp_path, root)
    assert code == 1
    assert _failed_checks(summary) == {"C5"}
    assert _failing_scenes(summary, "C5") == {"dev24-t02"}
    diffs = summary["per_configuration"]["dev24-t02"]["C5"]["max_abs_diff"]
    assert 9e-7 < diffs["solid/lawful vs gap/inconsistent"] < 1.1e-6
    assert diffs["solid/inconsistent vs gap/lawful"] == pytest.approx(0.0, abs=1e-12)


def test_divergence_at_frame_10_triggers_only_c6(tmp_path):
    root = build_bundle(tmp_path / "bundle", divergence_steps={"dev24-t01": 200})
    code, summary, _ = _run(tmp_path, root)
    assert code == 1
    assert _failed_checks(summary) == {"C6"}
    assert _failing_scenes(summary, "C6") == {"dev24-t01"}
    assert summary["per_configuration"]["dev24-t01"]["C6"]["per_world"]["gap"]["first_divergence_frame"] == 10.0


def test_divergence_at_frame_30_triggers_only_c6(tmp_path):
    root = build_bundle(tmp_path / "bundle", divergence_steps={"dev24-t01": 600})
    code, summary, _ = _run(tmp_path, root)
    assert code == 1
    assert _failed_checks(summary) == {"C6"}
    assert summary["per_configuration"]["dev24-t01"]["C6"]["per_world"]["solid"]["first_divergence_frame"] == 30.0


def test_area_dropping_to_zero_is_flagged_not_failed(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    _edit_manifest(root, "dev24-t01", "solid",
                   lambda clips: _clip(clips, "lawful", "clean")["target_area_pixels_per_frame"].__setitem__(20, 0))
    code, summary, out = _run(tmp_path, root)
    assert code == 0
    assert _failed_checks(summary) == set()
    assert summary["per_configuration"]["dev24-t01"]["C7"]["counts"] == {
        "clips": 12, "zero_frame": 1, "below_164": 1, "below_256": 1}
    with (out / "audit-by-clip.csv").open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["scene_id"] == "dev24-t01" and r["world"] == "solid"
                and r["branch"] == "lawful" and r["appearance"] == "clean"]
    assert rows[0]["zero_frame"] == "True" and rows[0]["min_area_frames_12_47"] == "0"


def test_area_dropping_to_200_px_is_flagged_below_256_only(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    _edit_manifest(root, "dev24-t02", "gap",
                   lambda clips: _clip(clips, "inconsistent", "abrupt_event")["target_area_pixels_per_frame"].__setitem__(30, 200))
    code, summary, _ = _run(tmp_path, root)
    assert code == 0
    assert _failed_checks(summary) == set()
    assert summary["per_configuration"]["dev24-t02"]["C7"]["counts"] == {
        "clips": 12, "zero_frame": 0, "below_164": 0, "below_256": 1}


def test_canonicalization_max_uint8_two_triggers_only_c8(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    path = root / "dev24-t01" / "gap" / AUDIT.CANON_NAME
    entries = json.loads(path.read_text(encoding="utf-8"))
    entries[0]["raw_context_vs_reference"]["max_uint8"] = 2
    path.write_text(json.dumps(entries), encoding="utf-8")
    code, summary, _ = _run(tmp_path, root)
    assert code == 1
    assert _failed_checks(summary) == {"C8"}


def test_rail_contact_in_stopping_branch_triggers_only_c9(tmp_path):
    # Gap world: the stopping branch is inconsistent.
    root = build_bundle(tmp_path / "bundle", ops={("dev24-t02", "gap", "inconsistent"): {"rail"}})
    code, summary, _ = _run(tmp_path, root)
    assert code == 1
    assert _failed_checks(summary) == {"C9"}
    assert _failing_scenes(summary, "C9") == {"dev24-t02"}


def test_support_contact_in_falling_branch_triggers_only_c10(tmp_path):
    # Solid world: the falling branch is inconsistent.
    root = build_bundle(tmp_path / "bundle", ops={("dev24-t01", "solid", "inconsistent"): {"fall_support"}})
    code, summary, _ = _run(tmp_path, root)
    assert code == 1
    assert _failed_checks(summary) == {"C10"}
    assert _failing_scenes(summary, "C10") == {"dev24-t01"}


def test_stopping_branch_without_support_contact_triggers_only_c10(tmp_path):
    root = build_bundle(tmp_path / "bundle", ops={("dev24-t01", "solid", "lawful"): {"no_support"}})
    code, summary, _ = _run(tmp_path, root)
    assert code == 1
    assert _failed_checks(summary) == {"C10"}
    assert any("stopping branch has no support contact" in f
               for f in summary["per_configuration"]["dev24-t01"]["C10"]["failures"])


def test_bundle_under_score_named_root_audits_normally(tmp_path):
    root = build_bundle(tmp_path / "score-blind-bundle")
    code, summary, _ = _run(tmp_path, root)
    assert code == 0
    assert _failed_checks(summary) == set()


def test_poisoned_side_folder_is_never_opened(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    poisoned = root / "scores"
    poisoned.mkdir()
    (poisoned / "all-scores.json").write_text("NOT JSON {{{ planted", encoding="utf-8")
    code, summary, _ = _run(tmp_path, root)
    assert code == 0
    assert _failed_checks(summary) == set()
    assert not any("score" in p.lower() for p in AUDIT.READ_LOG)


def test_score_named_file_below_root_is_refused(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    (root / "score-blind-notes.txt").write_text("not for audit", encoding="utf-8")
    AUDIT.set_bundle_root(root)
    try:
        with pytest.raises(AUDIT.ScoreBlindViolation):
            AUDIT.read_bytes(root / "score-blind-notes.txt")
        assert AUDIT.read_bytes(root / "dev24-t01" / "solid" / "lawful-states.json")
    finally:
        AUDIT.set_bundle_root(None)


def test_guard_refuses_paths_naming_scores_without_root():
    for bad in ("bundle/scores/all-scores.json", "bundle/SCORES.json", "x/Score_table.csv"):
        with pytest.raises(AUDIT.ScoreBlindViolation):
            AUDIT.read_bytes(bad)
    assert AUDIT._guard("bundle/solid/lawful-states.json") == Path("bundle/solid/lawful-states.json")


def test_guard_allows_score_named_files_outside_root(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    outside = tmp_path / "score-config-outside.json"
    outside.write_text("{}", encoding="utf-8")
    AUDIT.set_bundle_root(root)
    try:
        assert AUDIT.read_json(outside) == {}
    finally:
        AUDIT.set_bundle_root(None)


def test_reader_understands_real_state_audits():
    names = ("first-solid-state-audit.json", "first-gap-state-audit.json")
    if not all((REAL_AUDIT_DIR / name).is_file() for name in names):
        pytest.skip("real state-audit files not present; set A4_REAL_AUDIT_DIR")
    for name in names:
        audit = AUDIT.read_json(REAL_AUDIT_DIR / name)
        assert audit["first_position_divergence_step"] == 284
        assert AUDIT.divergence_from_state_audit(audit) == pytest.approx(14.2)
        assert 12 <= AUDIT.divergence_from_state_audit(audit) <= 24
        assert AUDIT.rail_status_from_state_audit(audit) == "rail contact not checkable from the bundle"
