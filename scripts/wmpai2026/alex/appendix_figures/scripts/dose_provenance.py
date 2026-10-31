"""Write figs/dose_figs_provenance.json: Python version, commands, input and output SHA-256.

Run last, after fig5_trajectories.py and fig7_cosines.py:
    python -I scripts/dose_provenance.py
"""
import os
import site
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(site.getusersitepackages())

import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

import dose_common as dc  # noqa: E402


DOSE_SCRIPTS = ["dose_common.py", "fig5_trajectories.py", "fig7_cosines.py", "dose_provenance.py"]
DOSE_OUTPUTS = [
    "fig5_dose_trajectories.png", "fig5_dose_trajectories.pdf",
    "fig5_all_layers.png", "fig5_all_layers.pdf",
    "fig7_displacement_cosines.png", "fig7_displacement_cosines.pdf",
    "dose_stats.json", "fig7_cosines.json",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(path):
    return {"path": str(path), "bytes": os.path.getsize(path), "sha256": sha256(path)}


def main():
    inputs = {name: entry(p) for name, p in [
        ("layer_features.npz", dc.NPZ),
        ("samples.json", dc.SAMPLES),
        ("analysis_split.json", dc.SPLIT),
    ]}
    scripts = {name: entry(dc.SCRIPTS / name) for name in DOSE_SCRIPTS}
    outputs = {name: entry(dc.FIGS / name) for name in DOSE_OUTPUTS}
    prov = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "commands": [
            "python -I scripts/fig5_trajectories.py",
            "python -I scripts/fig7_cosines.py",
            "python -I scripts/dose_provenance.py",
        ],
        "input_dir": str(dc.CACHE),
        "input_access": "read only; np.load with allow_pickle=False",
        "inputs": inputs,
        "scripts": scripts,
        "outputs": outputs,
        "notes": [
            "Factor name in samples.json is selective_recolor (the brief calls it recolor).",
            "Figure 5 projections are uncentered onto the train-fitted PC axes so that clean is the origin.",
            "Final-layer annotation uses the zhat variant (matches the paper values more closely).",
        ],
    }
    target = dc.FIGS / "dose_figs_provenance.json"
    target.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print("wrote", target)


if __name__ == "__main__":
    main()
