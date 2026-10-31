"""Run the acceptance checks and write figs/kubric_figs_provenance.json.

Run last, after fig1_crossover.py and fig3_threshold.py:
python scripts/provenance.py
"""
import json
import platform
import sys

import matplotlib
import numpy as np

from kubric_common import (
    APPS, BOOT_SEED, FIGS, JITTER_SEED, N_BOOT, N_SUBSET, SUBSET_SEED, T_STAR, WORK,
    boot_ci, boot_indices, format_table, group_means, load, margin_vector, run_checks,
    sha256)

SCRIPTS = ["kubric_common.py", "fig1_crossover.py", "fig3_threshold.py", "provenance.py"]
OUTPUTS = ["fig1_crossover.png", "fig1_crossover.pdf",
           "fig3_threshold_distributions.png", "fig3_threshold_distributions.pdf"]
COMMANDS = [
    "python scripts/fig1_crossover.py",
    "python scripts/fig3_threshold.py",
    "python scripts/provenance.py",
]


def rel(path):
    return path.relative_to(WORK).as_posix()


def main():
    S, scenes, info = load()
    rows = run_checks(S)
    print(format_table(rows))
    all_pass = all(r["ok"] for r in rows)
    print("ALL CHECKS PASS" if all_pass else "SOME CHECKS FAILED")

    means = group_means(S)
    print("lawful and inconsistent means (12 values)")
    for app in APPS:
        for world in ("solid", "gap"):
            m = means[app][world]
            print(f"  {app:<11}{world:<7} lawful {m['lawful_mean']:.12f}  "
                  f"inconsistent {m['inconsistent_mean']:.12f}")

    idx = boot_indices(len(scenes))
    margin_ci = {}
    for app in APPS:
        for world in ("solid", "gap"):
            est, lo, hi = boot_ci(margin_vector(S, world, app), idx)
            margin_ci[f"{app}/{world}"] = {"estimate": est, "ci95_lo": lo, "ci95_hi": hi}

    scripts = {name: sha256(WORK / "scripts" / name) for name in SCRIPTS}
    outputs = {name: sha256(FIGS / name) for name in OUTPUTS}

    provenance = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "seeds": {
            "bootstrap_numpy_default_rng": BOOT_SEED,
            "bootstrap_draws": N_BOOT,
            "subset_numpy_default_rng": SUBSET_SEED,
            "subset_size_scenes": N_SUBSET,
            "violin_jitter_numpy_default_rng": JITTER_SEED,
        },
        "threshold_t_star": T_STAR,
        "commands": COMMANDS,
        "input": {
            "path": rel(WORK / "f256" / "original" / "scores" / "all-scores.json"),
            "sha256": info["input_sha256"],
            "records_total": info["records_total"],
            "records_dropped_not_complete": info["records_dropped_not_complete"],
            "records_used": info["records_used"],
            "scenes": info["scenes"],
        },
        "notes": [
            "Appearance labels in the file are persistent_hue and abrupt_event; "
            "they are mapped to persistent and abrupt.",
            "The same 10000 scene-bootstrap index draws (BOOT_SEED) are used for "
            "the Figure 1 error bars and the margin intervals.",
            "Miss rate = share of inconsistent clips with score below t*.",
            "Figure 1 margin annotations use the point estimates, not the bootstrap means.",
        ],
        "acceptance_checks": rows,
        "acceptance_checks_all_pass": all_pass,
        "lawful_inconsistent_means": means,
        "identity_note": (
            "For each appearance, lawful mean minus inconsistent mean at a world equals "
            "minus the margin at that world (margin = inconsistent minus lawful). "
            "Checked in acceptance_checks rows named 'identity lawful-inc'."),
        "margin_bootstrap_95ci": margin_ci,
        "scripts_sha256": {k: v for k, v in scripts.items()},
        "outputs_sha256": outputs,
    }
    # Hashes of scripts and outputs are recorded in full. The provenance file cannot hash itself.
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "kubric_figs_provenance.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)
        fh.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
