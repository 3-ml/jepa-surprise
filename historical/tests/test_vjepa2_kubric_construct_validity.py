import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _run_analysis(output_dir: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    record = root / "results/kubric/construct_validity_pilot"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/analyze_vjepa2_kubric_construct_validity.py"),
            "--scores",
            str(record / "model/scores.csv"),
            "--render-metrics",
            str(record / "render_validation/render_metrics.csv"),
            "--output-dir",
            str(output_dir),
            "--bootstrap-replicates",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_kubric_analysis_is_byte_deterministic_and_reuses_control_intervals(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run_analysis(first)
    _run_analysis(second)

    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in first.iterdir()
    }
    second_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in second.iterdir()
    }
    assert first_hashes == second_hashes

    analysis = json.loads((first / "analysis.json").read_text())
    with (first / "condition_summary.csv").open(newline="") as handle:
        conditions = {row["condition"]: row for row in csv.DictReader(handle)}
    gate = analysis["clean_predictive_sensitivity_gate"]
    for condition in ("target_hold", "target_shuffle"):
        assert gate[f"{condition}_ci95"] == [
            float(conditions[condition]["ci95_low"]),
            float(conditions[condition]["ci95_high"]),
        ]

    with (first / "per_scene_deltas.csv").open(newline="") as handle:
        header = next(csv.reader(handle))
    assert header[:4] == [
        "scene_id",
        "delta_clean",
        "delta_hue-030",
        "delta_hue-060",
    ]
