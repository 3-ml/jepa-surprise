"""Local contracts for three known frozen studies; no model imports at preflight."""
import hashlib
import importlib
import importlib.metadata
import importlib.machinery
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {"physion": "physion-static", "geometry": "geometry-transfer", "final256": "final256"}
DESKTOP = "reviewer-physion-desktop-20261026-v2"
CONTROLS = "reviewer-physion-canonical-20261026-v2"
CONTACT = {"contact-development-1", "contact-development-scores-1"}
WORKSPACE = {"physion": {DESKTOP, CONTROLS}, "geometry": CONTACT | {"final256-v1", "development24-v2"},
             "final256": CONTACT | {"final256-v1"}}
ROLES = {"JEPA_VJEPA2_SOURCE": "directory", "JEPA_VJEPA2G_CHECKPOINT": "file"}
REVISION = "204698b45b3712590f06245fbfba32d3be539812"


def load(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe relative path: {name!r}")
    path = Path(name)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in name.split("/")):
        raise ValueError(f"Unsafe relative path: {name}")
    return root / path


def regular(path):
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError(f"Expected regular file without symlink ancestors: {path}")
    return path


def directory(path):
    if not path.is_dir() or any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError(f"Expected directory without symlink ancestors: {path}")
    return path


def hashed(path, expected):
    if sha(regular(path)) != expected:
        raise ValueError(f"Scientific/content hash mismatch: {path}")


def verify_published(root=None):
    root = ROOT if root is None else root
    for row in load(root / "release-manifest.json")["entries"]:
        path = regular(relative(root, row["path"]))
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Published hash mismatch: {path}")


def verify_inventory(bundle):
    directory(bundle)
    records = load(regular(bundle / "local-inputs.json"))
    if not isinstance(records, dict) or set(records) != {"files", "study_files", "workspace", "roles"}:
        raise ValueError("local-inputs.json requires files, study_files, workspace and roles only")
    for key in ("files", "workspace", "roles"):
        if not isinstance(records[key], dict):
            raise ValueError(f"Expected object: {key}")
    names = records["study_files"]
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise ValueError("study_files must contain unique relative filenames")
    actual = set()
    for path in bundle.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Only regular files/directories allowed: {path}")
        if path.is_file() and path != bundle / "local-inputs.json":
            actual.add(path.relative_to(bundle).as_posix())
    if actual != set(records["files"]):
        raise ValueError("Inventory must seal every file except the root local-inputs.json")
    for name, expected in records["files"].items():
        if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid SHA256: {name}")
        hashed(relative(bundle, name), expected)
    for name in names:
        relative(bundle, name)
        if name not in records["files"]:
            raise ValueError(f"Unsealed study file: {name}")
    return records


def overlap(a, b):
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


def output_path(path, bundle=None):
    path = path.absolute()
    if path.exists() or path.is_symlink() or overlap(path, ROOT) or (bundle is not None and overlap(path, bundle)):
        raise ValueError("Output must be fresh and disjoint from release and input bundle")
    if any(p.is_symlink() for p in path.parents):
        raise ValueError("Output ancestors must not be symlinks")
    return path


def fresh_output(path, bundle=None):
    path = output_path(path, bundle)
    path.mkdir(parents=True, exist_ok=False)
    return path


def timeout_value(value):
    if not math.isfinite(value) or not 0 < value <= 8 * 3600:
        raise ValueError("Timeout must be finite, positive and at most 28800 seconds")
    return value


def frozen(study):
    base = ROOT / "release" / STUDIES[study]
    return load(base / ("original/freeze.json" if study == "final256" else "freeze.json"))


def assets(bundle, records):
    return {key: relative(bundle, value) for key, value in records["workspace"].items()}


def check_layout(study, bundle, records):
    if set(records["workspace"]) != WORKSPACE[study]:
        raise ValueError(f"Required workspace keys for {study}: {sorted(WORKSPACE[study])}")
    if set(records["roles"]) != (set() if study == "physion" else set(ROLES)):
        raise ValueError(f"Required roles: {[] if study == 'physion' else sorted(ROLES)}")
    roots = []
    for key, value in records["workspace"].items():
        path = directory(relative(bundle, value))
        roots.append(path)
    for key, value in records["roles"].items():
        path = relative(bundle, value)
        (regular if ROLES[key] == "file" else directory)(path)
        roots.append(path)
    # Prevent nested aliases, bundle-root aliases and study-file/asset collisions.
    for i, path in enumerate(roots):
        if path == bundle or any(overlap(path, other) for other in roots[:i]):
            raise ValueError(f"Overlapping asset roles: {path}")
    for name in records["study_files"]:
        p = relative(bundle, name)
        if any(overlap(p, root) for root in roots):
            raise ValueError(f"Study file aliases an asset: {name}")


def scientific_external(study, bundle, records):
    """Selected scientific members of published freeze, never supplied authority."""
    freeze = frozen("geometry" if study == "final256" else study)
    roots = assets(bundle, records)
    for name, expected in freeze["external_files"].items():
        parts = name.replace("\\", "/").split("/")
        if parts[0] == "${JEPA_VASI_WORKSPACE}":
            suffix, tail = parts[1], "/".join(parts[2:])
            selected = (study == "physion" and (suffix == CONTROLS or (suffix == DESKTOP and
                (tail.startswith("model/") or tail in {"canonical_codec.py", "canonical_score.py", "model-source-manifest.json",
                 "source/run_vjepa2_section_surprise.py", "prior-input-seal.json", "qualification/complete.json"}))))
            selected |= study == "geometry" and suffix in {"final256-v1", "development24-v2"} and tail != "freeze.json"
            if selected:
                hashed(relative(roots[suffix], tail), expected)
        elif study != "physion" and parts[0] in {"${JEPA_VJEPA2_SOURCE}", "${JEPA_VJEPA2G_CHECKPOINT}"}:
            role = parts[0][2:-1]
            path = relative(bundle, records["roles"][role])
            hashed(relative(path, "/".join(parts[1:])) if len(parts) > 1 else path, expected)
    if study != "physion":
        source = relative(bundle, records["roles"]["JEPA_VJEPA2_SOURCE"])
        revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        if revision != REVISION:
            raise ValueError("Wrong V-JEPA2 source revision")
        known = {name.replace("\\", "/").split("/", 1)[1] for name in freeze["external_files"]
                 if name.startswith("${JEPA_VJEPA2_SOURCE}") and name.endswith(".py")}
        actual = {p.relative_to(source).as_posix() for p in (source / "src").rglob("*")
                  if p.is_file() and (p.suffix in {".py", ".pyc", ".pyd", ".so"})}
        if actual != known:
            raise ValueError("V-JEPA2 Python source closure differs from recorded freeze")
        # src is a namespace package. A sibling module wins over that directory,
        # including sourceless bytecode and platform-tagged extension modules.
        suffixes = importlib.machinery.all_suffixes()
        for path in source.iterdir():
            if path.name in {"src" + suffix for suffix in suffixes} or (
                path.name.startswith("src.") and path.suffix in {".so", ".pyd"}
            ):
                raise ValueError(f"Unrecorded src import override: {path}")
        importlib.invalidate_caches()
        spec = importlib.machinery.PathFinder.find_spec("src", [str(source)])
        if spec is None or spec.origin is not None or list(spec.submodule_search_locations or []) != [str(source / "src")]:
            raise ValueError("src import must resolve only to the recorded namespace directory")


def check_contact(roots):
    import csv
    supplied = roots["contact-development-scores-1"] / "scores.csv"
    published = ROOT / "results/25-11-42-wmpai-vasi-contact-development1/scores.csv"
    hashed(supplied, sha(published))
    row = next(r for r in csv.DictReader(published.open()) if r["physical_condition"] == "lawful" and r["appearance_condition"] == "clean")
    paths = json.loads(row["source_frame_paths_json"])
    hashes = json.loads(row["source_frame_sha256_json"])
    for name, expected in zip(paths, hashes, strict=True):
        hashed(relative(roots["contact-development-1"], name), expected)
    folder = roots["contact-development-1"] / "lawful/clean/frames"
    if {p.name for p in folder.iterdir()} != {Path(n).name for n in paths}:
        raise ValueError("Reference frame directory must contain exactly the recorded frames")


def check_frame_rows(root, rows):
    for row in rows:
        for name, expected in zip(row["source_frame_paths"], row["source_frame_sha256"], strict=True):
            hashed(relative(root, name), expected)


def check_renders(root):
    """Byte-level cold check; the original normalized input gate runs at execution."""
    original = ROOT / "release/final256/original"
    for name in ("freeze.json", "configs.json", "threshold.json", "inspected-registry.json", "render-seal.json"):
        hashed(root / name, sha(original / name))
    seal = load(root / "render-seal.json")
    configs = load(root / "configs.json")["scenes"]
    ids = {c["scene_id"] for c in configs}
    candidates = seal["candidates"]
    if len(ids) != 256 or len(candidates) != 256 or {c["scene_id"] for c in candidates} != ids or seal["eligible_count"] != 256:
        raise ValueError("Expected the published 256-scene sealed population")
    by_id = {c["scene_id"]: c for c in configs}
    for candidate in candidates:
        sid = candidate["scene_id"]
        if candidate["status"] != "eligible":
            raise ValueError("Ineligible candidate in completed render set")
        for world in ("solid", "gap"):
            data = relative(root, sid) / world
            hashed(data / "manifest.json", candidate["worlds"][world]["manifest_sha256"])
            manifest = load(data / "manifest.json")
            config = load(regular(data / "config.json"))
            if manifest["status"] != "pass" or manifest["issues"] or config != manifest["config"]:
                raise ValueError("Render/config audit failed")
            if config.get("world") != world or any(config.get(k) != v for k, v in by_id[sid].items() if k != "world"):
                raise ValueError("Render configuration differs from frozen population")
            audit = load(regular(data / "state-audit.json"))
            if audit["status"] != "pass" or not all(audit[k] for k in ("lawful_repeat_exact", "missing_stopping_contact_rejected", "different_context_state_rejected")):
                raise ValueError("State audit failed")
            canonical = load(regular(data / "context-canonicalization-audit.json"))
            if len(canonical) != 6 or not all(c["future_bytes_unchanged"] for c in canonical):
                raise ValueError("Context canonicalization audit failed")
            clips = manifest["clips"]
            expected = {(b, a) for b in ("lawful", "inconsistent") for a in ("clean", "persistent_hue", "abrupt_event")}
            if len(clips) != 6 or {(c["physical_condition"], c["appearance_condition"]) for c in clips} != expected:
                raise ValueError("Incomplete render conditions")
            for clip in clips:
                if clip["split"] != "confirmatory" or clip["scene_id"] != sid:
                    raise ValueError("Render split/scene mismatch")
                for prefix in ("source", "raw_source"):
                    paths, hashes = clip[prefix + "_frame_paths"], clip[prefix + "_frame_sha256"]
                    if len(paths) != 48 or len(hashes) != 48:
                        raise ValueError("Expected 48 frames per clip")
                    for name, expected_hash in zip(paths, hashes, strict=True):
                        hashed(relative(data, name), expected_hash)
                state_path = relative(data, clip["state_trace_path"])
                hashed(state_path, clip["state_trace_sha256"])
                if load(state_path)["config"] != config:
                    raise ValueError("State configuration mismatch")
                # The scorer loads a sorted PNG directory, not an arbitrary path list.
                frame_paths = [relative(data, name) for name in clip["source_frame_paths"]]
                folder = frame_paths[0].parent
                if any(p.parent != folder for p in frame_paths) or sorted(folder.glob("*.png")) != frame_paths:
                    raise ValueError("Scorer frame directory differs from sealed frame list")
                if clip["target_area_min_target_fraction"] < .0025:
                    raise ValueError("Render visibility gate failed")


def check_study(study, bundle):
    if study not in STUDIES:
        raise ValueError(f"Unknown study: {study}")
    records = verify_inventory(bundle)
    check_layout(study, bundle, records)
    roots = assets(bundle, records)
    base = ROOT / "release" / STUDIES[study]
    required = set()
    if study == "physion":
        for name in ("freeze.json", "input-seal.json"):
            hashed(bundle / name, sha(base / name))
            required.add(name)
        seal = load(base / "input-seal.json")
        for name, expected in seal["files"].items():
            hashed(relative(bundle, name), expected)
            required.add(name)
        for letter, total in (("A", 20708), ("B", 960)):
            rows = [json.loads(line) for line in (bundle / f"{letter}-input-manifest.jsonl").read_text().splitlines()]
            if len(rows) != total:
                raise ValueError("Incomplete scientific manifest")
            for row in rows:
                name = row["file"]
                hashed(relative(bundle, name), row["file_sha256"])
                required.add(name)
    elif study == "geometry":
        for name in ("freeze.json", "geometry-coefficients.json"):
            hashed(bundle / name, sha(base / name))
            required.add(name)
    if set(records["study_files"]) != required:
        raise ValueError("study_files must contain exactly the selected study metadata and tensor inputs")
    scientific_external(study, bundle, records)
    if study == "physion":
        controls = [json.loads(line) for line in (roots[CONTROLS] / "control-manifest.jsonl").read_text().splitlines()]
        if len(controls) != 337:
            raise ValueError("Expected 337 qualified controls")
        for row in controls:
            hashed(relative(roots[CONTROLS], row["file"]), row["file_sha256"])
    else:
        check_contact(roots)
        check_renders(roots["final256-v1"])
        if study == "geometry":
            check_frame_rows(roots["final256-v1"], load(roots["final256-v1"] / "scores/all-scores.json"))
            dev = roots["development24-v2"]
            first = load(dev / "scores/all-scores.json")[0]
            check_frame_rows(dev, [first])
            index = load(dev / "geometry-development-v2/feature-index.json")
            record = next(r for r in index if r["clip_id"] == first["clip_id"])
            hashed(relative(dev / "geometry-development-v2/clips", first["clip_id"] + ".npy"), record["feature_sha256"])
    return records


def check_versions(study):
    f = frozen(study)
    for package, expected in f["package_versions" if study == "final256" else "versions"].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError(f"Runtime version mismatch: {package}: {actual} != {expected}")

    if study == "physion":
        freeze = frozen(study)
        names = {Path(name).name: expected for name, expected in freeze["external_files"].items() if name.startswith("external-record/")}
        sources = {
            "transformers": ("transformers/models/vjepa2/modeling_vjepa2.py",
                             "transformers/models/vjepa2/configuration_vjepa2.py", "transformers/modeling_utils.py"),
            "torch": ("torch/nn/functional.py", "torch/nn/modules/module.py", "torch/nn/modules/normalization.py"),
        }
        for package, paths in sources.items():
            dist = importlib.metadata.distribution(package)
            for tail in paths:
                hashed(Path(dist.locate_file(tail)), names[Path(tail).name])


def scientific_freeze(study, bundle):
    verify_published()
    records = check_study(study, bundle)
    check_versions(study)
    return records


def run_child(command, timeout, env=None):
    timeout_value(timeout)
    if os.name != "posix" or not all(hasattr(os, name) for name in ("waitid", "WNOWAIT", "killpg")):
        raise RuntimeError("Local execution requires POSIX waitid/WNOWAIT and process groups")
    # The known entrypoints are single-process. Kill the owned session before reaping.
    child = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        # waitid WNOWAIT retains exclusive ownership of the leader PID during cleanup.
        end = time.monotonic() + timeout
        while os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            if time.monotonic() >= end:
                raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(.01)
        os.killpg(child.pid, signal.SIGKILL)
        code = child.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        if child.returncode is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise


def install_runtime(common, output, timeout, check_freeze):
    end = time.monotonic() + timeout_value(timeout)
    common.ROOT = output
    def deadline():
        if time.monotonic() >= end:
            raise TimeoutError("Local study deadline exceeded")
    def resources():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device required by frozen acquisition")
        free, total = torch.cuda.mem_get_info(0)
        return {"free": free, "total": total, "device": torch.cuda.get_device_name(0)}
    def sufficient(measured):
        if not 0 < measured["free"] <= measured["total"]:
            raise RuntimeError("No CUDA memory available")
        return True  # Availability only; not a model-fit or owner admission assertion.
    def guard():
        deadline()
        sufficient(resources())
    def wait_permit(study):
        if study not in {"physion", "geometry"}:
            raise ValueError(study)
        check_freeze()
        guard()
    common.deadline, common.inference_guard = deadline, guard
    common.wait_permit, common.check_freeze = wait_permit, check_freeze
    runtime = types.ModuleType("desktop_runtime")
    runtime.resources, runtime.sufficient, runtime.GIB = resources, sufficient, 2**30
    sys.modules["desktop_runtime"] = runtime


def dispatch_final256(output, root):
    """Only published check_sources and known roots are adapted; gate/score unchanged."""
    original = ROOT / "release/final256/original"
    source = original / "frozen-source"
    sys.path.insert(0, str(source / "scripts/wmpai2026/vasi"))
    common = importlib.import_module("final256_common")
    def check_sources(repo, lock):
        verify_published()
        if lock["source_sha256"] != frozen("final256")["source_sha256"]:
            raise ValueError("Final256 run lock differs from published freeze")
    common.check_sources = check_sources
    for name in ("freeze.json", "configs.json", "threshold.json", "inspected-registry.json", "render-seal.json"):
        shutil.copyfile(root / name, output / name)
    common.save(output / "run-lock.json", {"source_sha256": frozen("final256")["source_sha256"], "mode": "local-sealed-renders"})
    for scene in load(original / "configs.json")["scenes"]:
        sid = scene["scene_id"]
        (output / sid).symlink_to(root / sid, target_is_directory=True)
    sys.argv = ["final256_input_gate", "--root", str(output)]
    gate = importlib.import_module("final256_input_gate")
    gate.main()
    # Gate is recomputed, never trusted from a previous run.
    if sha(output / "input-gate-manifest.json") != sha(original / "input-gate-manifest.json"):
        raise ValueError("Recomputed normalized gate differs from published manifest")
    sys.argv = ["final256_score", "--root", str(output)]
    score = importlib.import_module("final256_score")
    score.main()


def bind_model_source(source):
    """Called only after scientific asset verification, before scientific imports."""
    if any(name == "src" or name.startswith("src.") for name in sys.modules):
        raise RuntimeError("src modules already loaded before verified namespace binding")
    namespace = types.ModuleType("src")
    spec = importlib.machinery.ModuleSpec("src", loader=None, is_package=True)
    spec.submodule_search_locations = [str(source / "src")]
    namespace.__spec__ = spec
    namespace.__path__ = spec.submodule_search_locations
    namespace.__package__ = "src"
    sys.modules["src"] = namespace


def dispatch(study, bundle, output, timeout):
    timeout_value(timeout)
    # Child, including direct invocation, performs all validation before any write/import.
    output_path(output, bundle)
    sys.dont_write_bytecode = True
    records = scientific_freeze(study, bundle)
    if study != "physion":
        bind_model_source(relative(bundle, records["roles"]["JEPA_VJEPA2_SOURCE"]))
    output = fresh_output(output, bundle)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    workspace = output / "workspace"
    workspace.mkdir()
    for suffix, target in records["workspace"].items():
        (workspace / suffix).symlink_to(relative(bundle, target), target_is_directory=True)
    os.environ["JEPA_VASI_WORKSPACE"] = str(workspace)
    for key, value in records["roles"].items():
        os.environ[key] = str(relative(bundle, value))
    for name in records["study_files"]:
        dest = relative(output, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with regular(relative(bundle, name)).open("rb") as src, dest.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    if study == "final256":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for final256 scoring")
        torch.cuda.mem_get_info(0)
        dispatch_final256(output, assets(bundle, records)["final256-v1"])
        return
    source = ROOT / "release" / STUDIES[study] / "frozen-source.zip"
    sys.path.insert(0, str(source))
    common = importlib.import_module("study_common")
    install_runtime(common, output, timeout, lambda: scientific_freeze(study, bundle))
    if study == "physion":
        canonical = importlib.import_module("canonical_score")
        canonical.ROOT = output
    entry = importlib.import_module("run_physion" if study == "physion" else "extract_geometry")
    entry.main()
