from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("_recreated_figures"))
    parser.add_argument("--skip-pytest", action="store_true")
    return parser.parse_args()


def assert_close(name: str, actual: float, expected: float, tolerance: float = 5e-4) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def check_manifest(root: Path, output_dir: Path) -> None:
    manifest_path = root / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    listed = [entry["path"] for entry in entries]
    if manifest.get("files") != len(entries):
        raise AssertionError("archive_manifest.json file count does not match entry count")
    if len(listed) != len(set(listed)):
        raise AssertionError("archive_manifest.json contains duplicate paths")
    problems = []
    total = 0
    for entry in entries:
        rel_path = Path(entry["path"])
        if rel_path.is_absolute() or ".." in rel_path.parts:
            problems.append(f"{entry['path']}: invalid archive-relative path")
            continue
        path = root / rel_path
        if not path.is_file():
            problems.append(f"{entry['path']}: missing file")
            continue
        size, digest = file_sha256(path)
        total += size
        if size != entry["bytes"]:
            problems.append(f"{entry['path']}: expected {entry['bytes']} bytes, found {size}")
        elif digest != entry["sha256"]:
            problems.append(f"{entry['path']}: sha256 mismatch")
    if manifest.get("bytes") != total:
        problems.append(f"manifest byte total {manifest.get('bytes')} differs from verified total {total}")
    ignored_output = None
    try:
        ignored_output = output_dir.resolve().relative_to(root).parts
    except ValueError:
        pass
    actual = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] == ".git" or relative.as_posix() == "archive_manifest.json":
            continue
        if ignored_output and relative.parts[: len(ignored_output)] == ignored_output:
            continue
        actual.add(relative.as_posix())
    missing = sorted(set(listed) - actual)
    extra_paths = sorted(actual - set(listed))
    problems.extend(f"{path}: absent from archive tree" for path in missing)
    problems.extend(f"{path}: unlisted archive file" for path in extra_paths)
    if problems:
        shown = "\n".join(problems[:10])
        extra = "" if len(problems) <= 10 else f"\n... {len(problems) - 10} more"
        raise AssertionError(f"archive manifest integrity failed:\n{shown}{extra}")


def check_intphys(root: Path) -> None:
    stats = root / "results/intphys2/scene_crossfit_probe_audit/sealed/statistics"
    accuracy = pd.read_csv(stats / "scene_clustered_accuracy_ci.csv")
    delta = pd.read_csv(stats / "scene_clustered_model_delta_ci.csv")
    expected = {
        ("V-JEPA2-G", "physics_all"): (0.5789473684210527, 0.5416666666666666, 0.618421052631579),
        ("VideoMAE2-G", "physics_all"): (0.4923245614035088, 0.4725877192982456, 0.5087719298245614),
        ("PixelSummary", "physics_all"): (0.5, 0.48903508771929827, 0.5098684210526315),
    }
    for (model, task), expected_values in expected.items():
        row = accuracy[accuracy.model.eq(model) & accuracy.task.eq(task)].iloc[0]
        for column, expected_value in zip(("mean", "ci_low", "ci_high"), expected_values):
            assert_close(f"{model} {task} {column}", float(row[column]), expected_value)
    expected_delta = {
        ("VideoMAE2-G", "physics_all"): (0.08662280701754387, 0.043859649122807015, 0.13267543859649122),
        ("PixelSummary", "physics_all"): (0.07894736842105263, 0.04057017543859649, 0.11842105263157894),
    }
    for (right_model, task), expected_values in expected_delta.items():
        row = delta[delta.right_model.eq(right_model) & delta.task.eq(task)].iloc[0]
        for column, expected_value in zip(("mean", "ci_low", "ci_high"), expected_values):
            assert_close(f"V-JEPA2-G minus {right_model} {task} {column}", float(row[column]), expected_value)
    audit = pd.read_csv(stats / "split_audit.csv")
    if set(audit.test_scenes.unique()) != {11, 12} or set(audit.test_pairs.unique()) != {22, 24}:
        raise AssertionError("sealed cross-fit fold sizes are unexpected")
    coverage = pd.read_csv(stats / "coverage_audit.csv")
    if set(coverage.unique_test_scenes) != {57} or set(coverage.unique_test_pairs) != {114}:
        raise AssertionError("sealed cross-fit population coverage is incomplete")
    if set(coverage.test_folds_per_scene_min) != {1} or set(coverage.test_folds_per_scene_max) != {1}:
        raise AssertionError("sealed scenes are not tested exactly once")
    if set(coverage.pairs_per_scene_min) != {2} or set(coverage.pairs_per_scene_max) != {2}:
        raise AssertionError("sealed scene pair support is incomplete")


def check_geometry(root: Path) -> None:
    controlled = root / "results/intphys2/controlled_geometry/sealed"
    pao = pd.read_csv(controlled / "PAO_K_SENSITIVITY_SEALED.csv")
    expected = {
        ("V-JEPA2-G", "block_00", 4): (0.4013410804958177, 0.320333, 0.482871),
        ("V-JEPA2-G", "final", 4): (0.059580, 0.049560, 0.070509),
    }
    for (model, layer, k), values in expected.items():
        row = pao[pao.model.eq(model) & pao.layer_name.eq(layer) & pao.k.eq(k)].iloc[0]
        for column, value in zip(("pao_raw", "pao_ci_low", "pao_ci_high"), values):
            assert_close(f"{model} {layer} k={k} {column}", float(row[column]), value)
    summary = json.loads(
        (root / "results/intphys2/multidose_geometry/VJEPA2G_GRID_SUMMARY.json").read_text()
    )
    expected_straightness = {
        "brightness": 0.906525291793712,
        "hue": 0.6918841348263429,
        "selective_recolor": 0.5527009999235307,
    }
    for factor, value in expected_straightness.items():
        assert_close(
            f"multi-dose {factor} straightness",
            float(summary["factors"][factor]["straightness"]),
            value,
        )
    expected_rank = {
        "brightness": 5.433315426330934,
        "hue": 9.830036762636643,
        "selective_recolor": 13.715682146125754,
    }
    for factor, value in expected_rank.items():
        assert_close(
            f"multi-dose {factor} effective rank",
            float(summary["factors"][factor]["effective_rank"]),
            value,
        )
    original_spec = root / "configs/original_sealed_analysis_spec/README.md"
    _, digest = file_sha256(original_spec)
    expected_digest = "f7e0ca1651c9586c180ba89c8be347d02d99de9045a8b1b9b75ed6d9fac7b465"
    if digest != expected_digest:
        raise AssertionError("original sealed specification hash mismatch")


def check_frozen_dino_controls(root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.validate_jepa_wm_pusht_controlled_displacement",
            "--result-dir",
            str(root / "results/jepa_wm/pusht_controlled_displacement"),
            "--dino-result-dir",
            str(root / "results/dinowm/pusht_controlled_displacement"),
            "--check-only",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def run_pytest(root: Path) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "tests"],
        cwd=root,
        check=True,
        env=env,
    )


def run_figures(root: Path, output_dir: Path) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.make_paper_figures",
            "--archive-root",
            str(root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        check=True,
        env=env,
    )
    expected = [
        "figure4_intphys_scene_crossfit.pdf",
        "figure2_controlled_geometry_compact.pdf",
        "figure3_lewm_construct_controls.pdf",
        "appendix_exploratory_trajectories.pdf",
    ]
    missing = [name for name in expected if not (output_dir / name).is_file()]
    if missing:
        raise AssertionError(f"missing regenerated figures: {missing}")


def main() -> None:
    args = parse_args()
    root = args.archive_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    check_manifest(root, output_dir)
    check_intphys(root)
    check_geometry(root)
    check_frozen_dino_controls(root)
    if not args.skip_pytest:
        run_pytest(root)
    run_figures(root, output_dir)
    print(json.dumps({"status": "passed", "figures": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
