import copy
import importlib.util
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "a2", Path(__file__).resolve().parents[1] / "scripts/wmpai2026/alex/a2_scene_stats.py")
a2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a2)


def fixture():
    """Synthetic tooling fixture: two development and two confirmatory scenes, clean + hue."""
    rows = []
    for sc, split, base in [("s1", "development", 0.0), ("s2", "development", 0.1),
                            ("s3", "confirmatory", 0.0), ("s4", "confirmatory", 0.1)]:
        for ap in ("clean", "hue"):
            for ph, delta in (("lawful", 0.0), ("inconsistent", 0.5)):
                rows.append(dict(scene_id=sc, clip_id=f"{sc}-{ph}-{ap}", split=split,
                                 physical_condition=ph, appearance_condition=ap,
                                 score=str(base + delta + (0.05 if ap == "hue" else 0)),
                                 context_sha=f"ctx-{sc}-{ap}", state_sha=f"st-{sc}-{ph}"))
    return rows


def test_clean_fixture_validates():
    assert a2.validate(fixture()) == []


def test_scene_in_two_splits_rejected():
    rows = fixture()
    rows[0]["split"] = "confirmatory"
    assert any("several splits" in e for e in a2.validate(rows))


def test_context_and_state_hash_mismatch_rejected():
    rows = fixture()
    rows[0]["context_sha"] = "other"
    rows[3]["state_sha"] = "other"
    errs = a2.validate(rows)
    assert any("context hash" in e for e in errs) and any("state hash" in e for e in errs)


def test_duplicate_cell_rejected():
    rows = fixture()
    rows.append(copy.deepcopy(rows[0]))
    assert any("duplicate cell" in e for e in a2.validate(rows))


def test_missing_cell_is_missing_not_zero():
    rows = fixture()
    rows[1]["score"] = ""  # s1 inconsistent clean
    m = a2.margins(rows, "development", "clean")
    assert "s1" not in m and "s2" in m
    assert a2.missing_scenes(rows, "development", "clean") == ["s1"]


def test_margins_and_swapped_labels():
    rows = fixture()
    assert all(abs(v - 0.5) < 1e-12 for v in a2.margins(rows, "confirmatory", "clean").values())
    for r in rows:
        r["physical_condition"] = "lawful" if r["physical_condition"] == "inconsistent" else "inconsistent"
    assert all(abs(v + 0.5) < 1e-12 for v in a2.margins(rows, "confirmatory", "clean").values())


def test_all_equal_scores_give_ties_and_chance_auroc():
    rows = fixture()
    for r in rows:
        r["score"] = "1.0"
    s = a2.pair_stats(a2.margins(rows, "confirmatory", "clean"), n_boot=200)
    assert s["frac_pos"] == 0.0 and s["ties"] == 2
    assert a2.auroc([1.0, 1.0], [1.0, 1.0]) == 0.5


def test_pair_stats_scene_bootstrap_and_threshold():
    rows = fixture()
    s = a2.pair_stats(a2.margins(rows, "confirmatory", "clean"), n_boot=500, seed=1)
    assert s["frac_pos"] == 1.0 and s["n_scenes"] == 2 and s["mean_ci"][0] <= 0.5 <= s["mean_ci"][1]
    t = a2.calibrate_threshold(rows)
    e = a2.threshold_errors(rows, t)
    assert math.isclose(e["auroc"], 1.0) and e["miss_rate"] == 0.0 and e["false_alarm_rate"] == 0.0


def with_status(rows):
    for r in rows:
        r["output_status"] = "complete"
    return rows


def test_table_with_status_column_validates_when_consistent():
    assert a2.validate(with_status(fixture())) == []


def test_failed_cell_with_a_score_is_rejected_but_blank_is_accepted():
    rows = with_status(fixture())
    rows[1]["output_status"], rows[1]["score"] = "failed_score", "0"
    assert any("must have a blank score" in e for e in a2.validate(rows))
    rows[1]["score"] = ""
    assert a2.validate(rows) == []


def test_excluded_cell_also_needs_a_blank_score():
    rows = with_status(fixture())
    rows[0]["output_status"], rows[0]["score"] = "excluded_score_blind", "0.2"
    assert any("excluded_score_blind cell must have a blank score" in e for e in a2.validate(rows))


def test_complete_cell_needs_a_finite_score():
    for bad in ("", "nan", "inf", "abc"):
        rows = with_status(fixture())
        rows[0]["score"] = bad
        assert any("lacks a finite score" in e for e in a2.validate(rows)), bad


def test_unknown_or_blank_status_is_rejected_when_the_column_exists():
    for bad in ("done", ""):
        rows = with_status(fixture())
        rows[0]["output_status"] = bad
        assert any("bad output_status" in e for e in a2.validate(rows)), bad


def test_failed_cell_with_zero_score_is_missing_in_margins_even_without_validation():
    rows = with_status(fixture())
    rows[1]["output_status"], rows[1]["score"] = "failed_score", "0"  # s1 inconsistent clean, zero-filled
    m = a2.margins(rows, "development", "clean")
    assert "s1" not in m and "s2" in m
    assert a2.missing_scenes(rows, "development", "clean") == ["s1"]


def test_common_mode_offset_breaks_threshold_not_pairing():
    rows = fixture()
    for r in rows:
        if r["split"] == "confirmatory" and r["appearance_condition"] == "hue":
            r["score"] = str(float(r["score"]) + 2.0)  # large common-mode shift
    t = a2.calibrate_threshold(rows, appearances={"clean"})
    assert a2.pair_stats(a2.margins(rows, "confirmatory", "hue"), n_boot=200)["frac_pos"] == 1.0
    assert a2.threshold_errors(rows, t)["false_alarm_rate"] > 0
