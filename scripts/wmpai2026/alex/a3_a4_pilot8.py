"""A3/A4 on the archived eight-scene Kubric pilot. EXPLORATORY: regression of code and a
first look; 8 scenes cannot support a held-out claim. Reads stored tables only; no frames
or features are needed (pilot PNGs are not on this machine).

Usage: python scripts/wmpai2026/alex/a3_a4_pilot8.py --out <results dir>
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lib", HERE / "a3_a4_lib.py")
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)

P = Path("results/23-05-00-vjepa2g-kubric-construct-validity-pilot8")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def independent_validity(rm, bv):
    """Checks recomputed from stored tables, written before looking at any score column."""
    rows = pd.DataFrame(bv["rows"])
    phys, app = rows[rows.family == "physical"], rows[rows.family == "appearance"]
    out = {
        "blend_rows": len(rows), "scenes": int(rows.scene_id.nunique()),
        "all_passed": bool(rows.passed.all()),
        "non_target_transform_max_abs": float(rows.non_target_transform_max_abs.max()),
        "pre_event_target_transform_max_abs": float(rows.target_pre_event_transform_max_abs.max()),
        "appearance_target_transform_max_abs": float(app.target_transform_max_abs.max()),
        "physical_target_transform_recorded": bool(phys.target_transform_max_abs.notna().all()),
        "appearance_other_material_equal_all": bool(app.other_material_fields_equal.all()),
        "event_center_in_frame_all": bool(rows.target_event_center_in_frame.all()),
        "render_rows": len(rm), "context_exact_all": bool(rm.context_exact.all()),
        "state_validation_passed_all": bool(rm.state_validation_passed.all()),
        "min_target_segmentation_area_px": int(rm.target_clean_segmentation_area_min_target.min()),
        "scenes_with_target_invisible_in_some_target_frame": sorted(
            rm[rm.target_clean_segmentation_area_min_target == 0].scene_id.unique().tolist()),
        "event_frame_segmentation_area_min_px": int(rm.target_clean_segmentation_area_event.min()),
        "changed_pixel_fraction_range_nonclean": [float(rm[rm.condition != "clean"].target_changed_pixel_fraction.min()),
                                                 float(rm[rm.condition != "clean"].target_changed_pixel_fraction.max())],
    }
    out["pass"] = bool(out["all_passed"] and out["non_target_transform_max_abs"] == 0
                       and out["pre_event_target_transform_max_abs"] == 0
                       and out["appearance_target_transform_max_abs"] == 0
                       and out["context_exact_all"] and out["event_frame_segmentation_area_min_px"] > 0)
    # Reported, not gated: physical state change is not recorded in this field, and some scenes lose
    # the target in later target frames (so interventions cannot change those frames' pixels).
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--perm", type=int, default=5000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sc = pd.read_csv(P / "model/scores.csv")
    rm = pd.read_csv(P / "render_validation/render_metrics.csv")
    bv = json.load(open(P / "render_validation/blend-validation.json"))
    validity = independent_validity(rm, bv)

    clean = sc[sc.condition == "clean"].set_index("scene_id").scalar_l1
    d = sc[sc.condition.isin(["clean"]) == False].copy()  # noqa: E712
    d["delta"] = d.scalar_l1.values - clean.loc[d.scene_id].values
    d = d.merge(rm[["scene_id", "condition", "target_rgb_l1_01", "target_changed_pixel_fraction"]],
                on=["scene_id", "condition"])
    d["is_physical"] = (d.family == "physical").astype(float)
    d.to_csv(out / "merged_pilot_table.csv", index=False)
    y, scene, cond = d.delta.values, d.scene_id.values, d.condition.values

    # A3: does encoder-side displacement (the only geometry quantity stored for this pilot)
    # predict the native score change on held-out scenes?
    obs, null_mean, p = lib.permutation_null_loso(d.target_shift_l1_layernorm.values, y, scene, cond, a.perm)
    a3 = {"predictor": "target_shift_l1_layernorm", "loso_r2": obs, "perm_null_mean_r2": null_mean,
          "perm_p": p, "spearman_pooled": lib.spearman(d.target_shift_l1_layernorm.values, y), "n_rows": len(d)}

    # A4: shortcut tests. Does the physical flag add anything beyond rendered change / encoder shift?
    a4 = {}
    for name, base in [("rgb_l1", d.target_rgb_l1_01.values),
                       ("changed_pixel_fraction", d.target_changed_pixel_fraction.values),
                       ("encoder_shift", d.target_shift_l1_layernorm.values)]:
        b, f, g = lib.nested_gain(base, d.is_physical.values, y, scene)
        a4[name] = {"base_loso_r2": b, "with_physical_flag_loso_r2": f, "gain": g}
    json.dump({"status": "EXPLORATORY, 8 scenes, stored-table analysis; not held-out evidence",
               "source_hashes": {str(P / n): sha(P / n) for n in ("model/scores.csv", "render_validation/render_metrics.csv",
                                                                  "render_validation/blend-validation.json")},
               "independent_validity_check": validity, "A3_geometry_to_error": a3, "A4_shortcut_nested": a4,
               "perm": a.perm},
              open(out / "summary.json", "w"), indent=1)
    print(json.dumps({"validity_pass": validity["pass"], "A3": a3, "A4": a4}, indent=1))


if __name__ == "__main__":
    main()
