"""Score-blind audit of Vasi's development24 render bundle, run before any model score is read.

Usage:
  python a4_audit_development24.py --root <bundle root> --configs <development24-configs.json>
         --out <dir> [--check-frames]

Writes audit-summary.json and audit-by-clip.csv to --out. Exit code 1 if a hard check
(C1 to C6, C8, C9, C10) fails, otherwise 0. C7 is reported only.

Layout assumption, derived from scripts/wmpai2026/vasi/geometry_batch_worker.py,
canonical_development_context.py and run_development24.py on origin/wmpai/vasi-experiments
(the layout names are module constants below; change them there if the bundle moves):

  <root>/<scene_id>/<world>/manifest.json                        world manifest; key "clips" (6 per world)
  <root>/<scene_id>/<world>/<branch>-states.json                 keys "trace" (per-step rows: step,
                                                                 position, quaternion_xyzw, ...,
                                                                 contacts) and "body_ids" (name -> id)
  <root>/<scene_id>/<world>/<branch>/<appearance>/clip.json      raw per-clip record (not updated by
                                                                 canonicalization; existence checked only)
  <root>/<scene_id>/<world>/context-canonicalization-audit.json  list; entries carry
                                                                 raw_context_vs_reference metrics
  <root>/<scene_id>/<world>/state-audit.json                     per-world flat summary dict

Clip records used by the checks come from the world manifest, because canonicalization
rewrites source_frame_paths and source_frame_sha256 there (model-input frames). Clip keys:
physical_condition, appearance_condition, state_trace_path, state_trace_sha256,
source_frame_paths (48), source_frame_sha256 (48), target_area_pixels_per_frame (48).
Steps per frame = simulation_hz // fps from the configuration (240 // 12 = 20), so a
divergence step s maps to frame s / 20.

Contacts: each trace row's "contacts" is a list of PyBullet getContactPoints tuples;
index 2 is the other body's unique id, index 9 the normal force.
Stopping branch: solid world lawful, gap world inconsistent. Falling branch: the other one.

Score-blind guarantee: every file access goes through _guard(). Inside the bundle root
(when set by run()), a read is refused if any path component below the root starts with
"score", equals "scores", or contains "all-scores". Outside the root (for example the
configs file) reads are not bundle content and are allowed. Without a root, the whole path
is checked. Only explicitly named files are opened; no directory is listed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

WORLDS = ("solid", "gap")
BRANCHES = ("lawful", "inconsistent")
APPEARANCES = ("clean", "persistent_hue", "abrupt_event")
MANIFEST_NAME = "manifest.json"
STATES_NAME = "{branch}-states.json"
CLIP_NAME = "{branch}/{appearance}/clip.json"
CANON_NAME = "context-canonicalization-audit.json"
STATE_AUDIT_NAME = "state-audit.json"
RAIL_NAMES = ("left_rail", "right_rail")
SUPPORT_NAME = "support"
CONTACT_BODY_INDEX = 2
CONTACT_FORCE_INDEX = 9

TOTAL_FRAMES = 48
CONTEXT_FRAMES = 12
FRAME_PIXELS = 256 * 256
VISIBILITY_FLOORS = (164, 256)
CROSS_WORLD_TOL = 1e-9
DIVERGENCE_TOL = 1e-12
DIVERGENCE_FRAME_RANGE = (12, 24)
CANON_LIMITS = {"max_uint8": 1, "normalized_mean_abs": 1e-6, "changed_pixel_fraction": 1e-4}
DEFAULT_SIM_HZ = 240
DEFAULT_FPS = 12
CHECK_IDS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10")
HARD_CHECKS = ("C1", "C2", "C3", "C4", "C5", "C6", "C8", "C9", "C10")
PASS, FAIL, NOT_CHECKABLE, REPORTED = "pass", "fail", "not_checkable", "reported"

READ_LOG: list[str] = []
_BUNDLE_ROOT: Path | None = None


class ScoreBlindViolation(RuntimeError):
    """Raised when code tries to open bundle content that may hold model scores."""


def set_bundle_root(root) -> None:
    global _BUNDLE_ROOT
    _BUNDLE_ROOT = Path(root).resolve() if root is not None else None


def _refused_component(part: str) -> bool:
    low = part.lower()
    return low.startswith("score") or low == "scores" or "all-scores" in low


def _guard(path) -> Path:
    p = Path(path)
    if _BUNDLE_ROOT is None:
        parts = p.parts
        logged = "/".join(parts)
    else:
        try:
            parts = p.resolve().relative_to(_BUNDLE_ROOT).parts
            logged = "/".join(parts)
        except ValueError:
            parts = ()  # outside the bundle root: not bundle content
            logged = str(p)
    if any(_refused_component(part) for part in parts):
        raise ScoreBlindViolation(f"score-blind guard: refusing to open {p}")
    READ_LOG.append(logged)
    return p


def exists(path) -> bool:
    return _guard(path).exists()


def read_bytes(path) -> bytes:
    return _guard(path).read_bytes()


def read_json(path):
    return json.loads(read_bytes(path).decode("utf-8"))


def sha256_file(path) -> str:
    return hashlib.sha256(read_bytes(path)).hexdigest()


def divergence_from_state_audit(audit: dict, steps_per_frame: int = DEFAULT_SIM_HZ // DEFAULT_FPS):
    """Frame of the first position divergence recorded in a real state-audit JSON, or None."""
    step = audit.get("first_position_divergence_step")
    if not isinstance(step, int) or isinstance(step, bool):
        return None
    return step / steps_per_frame


def rail_status_from_state_audit(audit: dict) -> str:
    """Rail status from state-audit fields only; the state-audit files carry none in practice."""
    rail_keys = [k for k in audit if "rail" in k.lower()]
    if not rail_keys:
        return "rail contact not checkable from the bundle"
    return PASS if all(not audit[k] for k in rail_keys) else FAIL


def _steps_per_frame(config: dict) -> int:
    return int(config.get("simulation_hz", DEFAULT_SIM_HZ)) // int(config.get("fps", DEFAULT_FPS))


def _result(status=PASS, failures=None, notes=None, **extra) -> dict:
    out = {"status": status, "failures": list(failures or []), "notes": list(notes or [])}
    out.update(extra)
    return out


def _finish(result: dict) -> dict:
    if result["failures"]:
        result["status"] = FAIL
    return result


def load_world(root, scene: str, world: str) -> dict:
    wdir = Path(root) / scene / world
    loaded = {"dir": wdir, "clips": {}, "counts": {}, "error": None}
    try:
        manifest = read_json(wdir / MANIFEST_NAME)
        clips = manifest["clips"]
        if not isinstance(clips, list):
            raise ValueError("manifest clips is not a list")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        loaded["error"] = f"manifest unreadable: {exc!r}"
        return loaded
    for clip in clips:
        key = (clip.get("physical_condition"), clip.get("appearance_condition"))
        loaded["counts"][key] = loaded["counts"].get(key, 0) + 1
        loaded["clips"].setdefault(key, clip)
    return loaded


def _clip(worlds, world, branch, appearance):
    return worlds[world]["clips"].get((branch, appearance))


def check_c1(scene, worlds):
    failures = []
    for world in WORLDS:
        W = worlds[world]
        if W["error"]:
            failures.append(f"{scene}/{world}: {W['error']}")
        for branch, appearance in product(BRANCHES, APPEARANCES):
            n = W["counts"].get((branch, appearance), 0)
            if n == 0:
                failures.append(f"{scene}/{world}: missing clip {branch}/{appearance} in manifest")
            elif n > 1:
                failures.append(f"{scene}/{world}: duplicate clip {branch}/{appearance} in manifest")
            clip_file = W["dir"] / CLIP_NAME.format(branch=branch, appearance=appearance)
            if not exists(clip_file):
                failures.append(f"{scene}/{world}: clip.json missing for {branch}/{appearance}")
    return _finish(_result(failures=failures))


def check_c2(scene, worlds, check_frames):
    failures = []
    for world in WORLDS:
        W = worlds[world]
        for (branch, appearance), clip in W["clips"].items():
            label = f"{scene}/{world}/{branch}/{appearance}"
            rel = clip.get("state_trace_path")
            recorded = clip.get("state_trace_sha256")
            if rel is None or recorded is None:
                failures.append(f"{label}: state_trace_path or state_trace_sha256 missing")
            else:
                try:
                    if sha256_file(W["dir"] / rel) != recorded:
                        failures.append(f"{label}: state trace hash mismatch ({rel})")
                except OSError as exc:
                    failures.append(f"{label}: state trace unreadable ({exc.__class__.__name__})")
            if check_frames:
                paths = clip.get("source_frame_paths") or []
                hashes = clip.get("source_frame_sha256") or []
                if len(paths) != TOTAL_FRAMES or len(hashes) != TOTAL_FRAMES:
                    failures.append(f"{label}: expected {TOTAL_FRAMES} frame paths and hashes")
                    continue
                for i, (p, h) in enumerate(zip(paths, hashes)):
                    try:
                        if sha256_file(W["dir"] / p) != h:
                            failures.append(f"{label}: frame {i} hash mismatch")
                    except OSError:
                        failures.append(f"{label}: frame {i} missing")
    return _finish(_result(failures=failures))


def _context(clip):
    hashes = clip.get("source_frame_sha256") if clip else None
    if not hashes or len(hashes) < CONTEXT_FRAMES:
        return None
    return list(hashes[:CONTEXT_FRAMES])


def check_c3(scene, worlds):
    failures = []
    for world in WORLDS:
        for appearance in APPEARANCES:
            lawful = _clip(worlds, world, "lawful", appearance)
            incons = _clip(worlds, world, "inconsistent", appearance)
            if lawful and incons:
                if _context(lawful) is None or _context(incons) is None:
                    failures.append(f"{scene}/{world}/{appearance}: fewer than 12 context hashes")
                elif _context(lawful) != _context(incons):
                    failures.append(f"{scene}/{world}/{appearance}: lawful and inconsistent context differ")
        for branch in BRANCHES:
            clean = _clip(worlds, world, branch, "clean")
            abrupt = _clip(worlds, world, branch, "abrupt_event")
            persistent = _clip(worlds, world, branch, "persistent_hue")
            if clean and abrupt and _context(clean) is not None and _context(abrupt) is not None:
                if _context(abrupt) != _context(clean):
                    failures.append(f"{scene}/{world}/{branch}: abrupt_event context differs from clean")
            if clean and persistent and _context(clean) is not None and _context(persistent) is not None:
                if _context(persistent) == _context(clean):
                    failures.append(f"{scene}/{world}/{branch}: persistent_hue context equals clean")
    return _finish(_result(failures=failures))


def check_c4(scene, worlds):
    failures = []
    for world in WORLDS:
        for branch in BRANCHES:
            present = [c for a in APPEARANCES if (c := _clip(worlds, world, branch, a)) is not None]
            if not present:
                continue
            values = {c.get("state_trace_sha256") for c in present}
            if None in values:
                failures.append(f"{scene}/{world}/{branch}: state_trace_sha256 missing")
            elif len(values) > 1:
                failures.append(f"{scene}/{world}/{branch}: state trace differs across appearances")
    return _finish(_result(failures=failures))


def _states_rel(W: dict, branch: str) -> str:
    for (b, a), clip in W["clips"].items():
        if b == branch and a == "clean" and clip.get("state_trace_path"):
            return clip["state_trace_path"]
    return STATES_NAME.format(branch=branch)


def load_states(W: dict, branch: str) -> dict:
    """Return {'error': None|'missing'|'unreadable'|'no_field', 'trace': rows, 'body_ids': dict|None}."""
    try:
        data = read_json(W["dir"] / _states_rel(W, branch))
    except FileNotFoundError:
        return {"error": "missing"}
    except (OSError, ValueError):
        return {"error": "unreadable"}
    trace = data.get("trace") if isinstance(data, dict) else None
    if not isinstance(trace, list) or not trace or any(not isinstance(row, dict) for row in trace):
        return {"error": "no_field"}
    body_ids = data.get("body_ids")
    return {"error": None, "trace": trace, "body_ids": body_ids if isinstance(body_ids, dict) else None}


def load_positions(W: dict, branch: str):
    """Return (positions, steps, error). error is None, 'missing', 'unreadable' or 'no_field'."""
    st = load_states(W, branch)
    if st["error"]:
        return None, None, st["error"]
    trace = st["trace"]
    if any("position" not in row for row in trace):
        return None, None, "no_field"
    positions = [list(map(float, row["position"])) for row in trace]
    steps = [row.get("step", i) for i, row in enumerate(trace)]
    return positions, steps, None


def check_c5(scene, worlds):
    pairs = [(("solid", "lawful"), ("gap", "inconsistent")),
             (("solid", "inconsistent"), ("gap", "lawful"))]
    failures, notes, diffs = [], [], {}
    not_checkable = False
    for (w1, b1), (w2, b2) in pairs:
        name = f"{w1}/{b1} vs {w2}/{b2}"
        p1, _, e1 = load_positions(worlds[w1], b1)
        p2, _, e2 = load_positions(worlds[w2], b2)
        if e1 == "no_field" or e2 == "no_field":
            not_checkable = True
            notes.append(f"{name}: position field not available")
            continue
        if e1 or e2:
            failures.append(f"{scene}: states file {'missing' if 'missing' in (e1, e2) else 'unreadable'} for {name}")
            continue
        if len(p1) != len(p2) or any(len(a) != len(b) for a, b in zip(p1, p2)):
            failures.append(f"{scene}: trajectory shape mismatch for {name}")
            diffs[name] = None
            continue
        diff = max((abs(x - y) for a, b in zip(p1, p2) for x, y in zip(a, b)), default=0.0)
        diffs[name] = diff
        if diff > CROSS_WORLD_TOL:
            failures.append(f"{scene}: stopping/falling trajectory differs for {name} (max {diff:.3g})")
    status = FAIL if failures else (NOT_CHECKABLE if not_checkable else PASS)
    return _result(status=status, failures=failures, notes=notes, max_abs_diff=diffs)


def check_c6(scene, worlds, steps_per_frame):
    failures, per_world = [], {}
    lo, hi = DIVERGENCE_FRAME_RANGE
    for world in WORLDS:
        p_law, steps, e1 = load_positions(worlds[world], "lawful")
        p_inc, _, e2 = load_positions(worlds[world], "inconsistent")
        if e1 or e2:
            if "no_field" in (e1, e2):
                per_world[world] = {"first_divergence_step": None, "first_divergence_frame": None,
                                    "status": NOT_CHECKABLE}
            else:
                failures.append(f"{scene}/{world}: states file missing or unreadable")
                per_world[world] = {"first_divergence_step": None, "first_divergence_frame": None,
                                    "status": FAIL}
            continue
        index = None
        for i, (a, b) in enumerate(zip(p_law, p_inc)):
            if any(abs(x - y) > DIVERGENCE_TOL for x, y in zip(a, b)):
                index = i
                break
        if index is None:
            failures.append(f"{scene}/{world}: lawful and inconsistent positions never diverge")
            per_world[world] = {"first_divergence_step": None, "first_divergence_frame": None,
                                "status": FAIL}
            continue
        step = steps[index]
        frame = step / steps_per_frame
        per_world[world] = {"first_divergence_step": step, "first_divergence_frame": frame,
                            "status": PASS}
        if not lo <= frame <= hi:
            failures.append(f"{scene}/{world}: first divergence at frame {frame:g} outside [{lo}, {hi}]")
            per_world[world]["status"] = FAIL
    status = FAIL if failures else (NOT_CHECKABLE if any(v["status"] == NOT_CHECKABLE for v in per_world.values()) else PASS)
    return _result(status=status, failures=failures, per_world=per_world)


def check_c7(scene, worlds):
    flags = {"clips": 0, "zero_frame": 0, "below_164": 0, "below_256": 0}
    rows = {}
    notes = []
    for world in WORLDS:
        for (branch, appearance), clip in worlds[world]["clips"].items():
            area = clip.get("target_area_pixels_per_frame")
            if not isinstance(area, list) or len(area) != TOTAL_FRAMES:
                notes.append(f"{scene}/{world}/{branch}/{appearance}: area list not {TOTAL_FRAMES} entries")
                continue
            window = area[CONTEXT_FRAMES:TOTAL_FRAMES]
            minimum = min(window)
            row = {"min_area": minimum, "zero_frame": any(v == 0 for v in window),
                   "below_164": minimum < VISIBILITY_FLOORS[0], "below_256": minimum < VISIBILITY_FLOORS[1]}
            rows[(world, branch, appearance)] = row
            flags["clips"] += 1
            for key in ("zero_frame", "below_164", "below_256"):
                flags[key] += int(row[key])
    return _result(status=REPORTED, notes=notes, counts=flags, rows=rows)


def check_c8(scene, worlds):
    failures = []
    for world in WORLDS:
        label = f"{scene}/{world}"
        path = worlds[world]["dir"] / CANON_NAME
        try:
            entries = read_json(path)
        except FileNotFoundError:
            failures.append(f"{label}: {CANON_NAME} missing")
            continue
        except (OSError, ValueError) as exc:
            failures.append(f"{label}: {CANON_NAME} unreadable ({exc.__class__.__name__})")
            continue
        if not isinstance(entries, list):
            failures.append(f"{label}: {CANON_NAME} is not a list")
            continue
        for i, entry in enumerate(entries):
            metrics = entry.get("raw_context_vs_reference") if isinstance(entry, dict) else None
            if not isinstance(metrics, dict):
                failures.append(f"{label}: canonicalization entry {i} has no metrics")
                continue
            for key, limit in CANON_LIMITS.items():
                value = metrics.get(key)
                if value is None:
                    failures.append(f"{label}: entry {i} lacks {key}")
                elif value > limit:
                    failures.append(f"{label}: entry {i} {key}={value} exceeds {limit}")
    return _finish(_result(failures=failures))


def _contact_rows(trace, body_index_ids):
    """Yield (step, contact) for every contact in a trace; body_index_ids unused, kept for clarity."""
    for i, row in enumerate(trace):
        for contact in row.get("contacts") or []:
            yield row.get("step", i), contact


def check_c9(scene, worlds):
    """Rail contact: fail if any trace row of either branch touches a rail body."""
    failures, notes = [], []
    checkable = 0
    for world in WORLDS:
        for branch in BRANCHES:
            st = load_states(worlds[world], branch)
            label = f"{scene}/{world}/{branch}"
            if st["error"] in ("missing", "unreadable"):
                failures.append(f"{label}: states file {st['error']}")
                continue
            trace, body = st.get("trace"), st.get("body_ids")
            if st["error"] or body is None or any(n not in body for n in RAIL_NAMES) \
                    or any("contacts" not in row for row in trace):
                notes.append(f"{label}: body_ids or contacts absent; rail contact not checkable")
                continue
            checkable += 1
            rail_ids = {body[n] for n in RAIL_NAMES}
            for step, contact in _contact_rows(trace, None):
                if len(contact) > CONTACT_BODY_INDEX and contact[CONTACT_BODY_INDEX] in rail_ids:
                    failures.append(f"{label}: rail contact at step {step}")
                    break
    if failures:
        return _result(status=FAIL, failures=failures, notes=notes)
    return _result(status=PASS if checkable else NOT_CHECKABLE, notes=notes)


def _has_support_force(row, support_id) -> bool:
    return any(len(c) > CONTACT_FORCE_INDEX and c[CONTACT_BODY_INDEX] == support_id
               and c[CONTACT_FORCE_INDEX] > 0 for c in row.get("contacts") or [])


def stopping_branch(world: str) -> str:
    return "lawful" if world == "solid" else "inconsistent"


def falling_branch(world: str) -> str:
    return "inconsistent" if world == "solid" else "lawful"


def check_c10(scene, worlds, c6_per_world):
    """Support contact: falling branch never touches the central support, stopping branch does."""
    failures, notes, per_world = [], [], {}
    checkable = 0
    for world in WORLDS:
        steps_by_branch = {}
        for branch in BRANCHES:
            st = load_states(worlds[world], branch)
            label = f"{scene}/{world}/{branch}"
            if st["error"] in ("missing", "unreadable"):
                failures.append(f"{label}: states file {st['error']}")
                continue
            trace, body = st.get("trace"), st.get("body_ids")
            if st["error"] or body is None or SUPPORT_NAME not in body \
                    or any("contacts" not in row for row in trace):
                notes.append(f"{label}: body_ids or contacts absent; support contact not checkable")
                continue
            support_id = body[SUPPORT_NAME]
            steps_by_branch[branch] = [row.get("step", i) for i, row in enumerate(trace)
                                       if _has_support_force(row, support_id)]
        stop, fall = stopping_branch(world), falling_branch(world)
        if stop not in steps_by_branch or fall not in steps_by_branch:
            continue
        checkable += 1
        fall_steps, stop_steps = steps_by_branch[fall], steps_by_branch[stop]
        if fall_steps:
            failures.append(f"{scene}/{world}/{fall}: falling branch has {len(fall_steps)} support contact steps "
                            f"(first {fall_steps[0]})")
        if not stop_steps:
            failures.append(f"{scene}/{world}/{stop}: stopping branch has no support contact")
        first_stop = stop_steps[0] if stop_steps else None
        div = c6_per_world.get(world, {}).get("first_divergence_step")
        per_world[world] = {"stopping_branch": stop, "first_support_contact_step": first_stop,
                            "first_position_divergence_step": div}
        if first_stop is not None and div is not None and first_stop != div:
            notes.append(f"{scene}/{world}: first support contact step {first_stop} differs from "
                         f"first position divergence step {div}")
    if failures:
        return _result(status=FAIL, failures=failures, notes=notes, per_world=per_world)
    return _result(status=PASS if checkable else NOT_CHECKABLE, notes=notes, per_world=per_world)


def audit_configuration(root, config: dict, check_frames: bool):
    scene = config["scene_id"]
    worlds = {world: load_world(root, scene, world) for world in WORLDS}
    steps_per_frame = _steps_per_frame(config)
    checks = {
        "C1": check_c1(scene, worlds),
        "C2": check_c2(scene, worlds, check_frames),
        "C3": check_c3(scene, worlds),
        "C4": check_c4(scene, worlds),
        "C5": check_c5(scene, worlds),
        "C6": check_c6(scene, worlds, steps_per_frame),
        "C7": check_c7(scene, worlds),
        "C8": check_c8(scene, worlds),
    }
    checks["C9"] = check_c9(scene, worlds)
    checks["C10"] = check_c10(scene, worlds, checks["C6"]["per_world"])
    divergence = {w: checks["C6"]["per_world"].get(w, {}).get("first_divergence_frame") for w in WORLDS}
    rows = []
    visibility = checks["C7"].pop("rows")
    for world, branch, appearance in product(WORLDS, BRANCHES, APPEARANCES):
        clip = worlds[world]["clips"].get((branch, appearance))
        vis = visibility.get((world, branch, appearance))
        rows.append({
            "scene_id": scene, "world": world, "branch": branch, "appearance": appearance,
            "present": clip is not None,
            "min_area_frames_12_47": vis["min_area"] if vis else "",
            "below_164": vis["below_164"] if vis else "",
            "below_256": vis["below_256"] if vis else "",
            "zero_frame": vis["zero_frame"] if vis else "",
            "first_divergence_frame": divergence[world] if divergence[world] is not None else "",
        })
    return checks, rows


def audit_bundle(root, configs_path, check_frames: bool):
    configs = read_json(configs_path)
    scenes = configs["scenes"] if isinstance(configs, dict) else configs
    per_configuration, rows = {}, []
    for config in scenes:
        checks, config_rows = audit_configuration(root, config, check_frames)
        per_configuration[config["scene_id"]] = checks
        rows.extend(config_rows)
    totals = {check: {PASS: 0, FAIL: 0, NOT_CHECKABLE: 0, REPORTED: 0} for check in CHECK_IDS}
    visibility_totals = {"clips": 0, "zero_frame": 0, "below_164": 0, "below_256": 0}
    hard_failed = False
    for checks in per_configuration.values():
        for check, result in checks.items():
            totals[check][result["status"]] += 1
            if check in HARD_CHECKS and result["status"] == FAIL:
                hard_failed = True
        for key, value in checks["C7"].get("counts", {}).items():
            visibility_totals[key] += value
    summary = {
        "configurations": len(scenes),
        "check_frames": check_frames,
        "hard_checks": list(HARD_CHECKS),
        "visibility_floors_px": list(VISIBILITY_FLOORS),
        "totals": totals,
        "visibility_totals": visibility_totals,
        "hard_check_failed": hard_failed,
        "exit_code": 1 if hard_failed else 0,
        "per_configuration": per_configuration,
        "policy": "C7 is reported, not a failure. C9 and C10 are hard. Not-checkable results are not failures.",
    }
    return summary, rows


def write_outputs(out_dir, summary, rows):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8")
    fields = ["scene_id", "world", "branch", "appearance", "present", "min_area_frames_12_47",
              "below_164", "below_256", "zero_frame", "first_divergence_frame"]
    with (out / "audit-by-clip.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(root, configs_path, out_dir, check_frames: bool = False) -> int:
    READ_LOG.clear()
    set_bundle_root(root)
    try:
        summary, rows = audit_bundle(root, configs_path, check_frames)
    finally:
        set_bundle_root(None)
    write_outputs(out_dir, summary, rows)
    return summary["exit_code"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-frames", action="store_true")
    args = parser.parse_args(argv)
    code = run(args.root, args.configs, args.out, args.check_frames)
    print(f"audit written to {args.out}; exit code {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
