#!/usr/bin/env python3
"""Development-only V1 regression using on-demand disk block offload, batch one.

The old pilot is a scoring fixture, NOT a lawful/impossible physics test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(str(__import__("pathlib").Path(__file__).resolve().parents[3]))  # EDIT 1: repo root (copy lives outside repo, parents[3] wrong)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # EDIT 2: find disk_offload_vjepa2 copy in scratch
import run_intphys2_dense as core
import run_vjepa2_kubric_construct_validity as pilot
import disk_offload_vjepa2 as disk

PILOT = ROOT / "results/23-05-00-vjepa2g-kubric-construct-validity-pilot8"
CHECKPOINT = Path(__import__("os").environ['JEPA_VJEPA2G_CHECKPOINT'])  # EDIT 3: laptop checkpoint path
MODEL_ROOT = Path(__import__("os").environ['JEPA_VJEPA2_SOURCE'])  # EDIT 4: laptop V-JEPA2 source path
FRAMES = Path(__import__("os").path.join(__import__("os").environ['JEPA_PILOT8_FRAMES'], 'seed-230200'))  # EDIT 5: laptop frames path (same layout)
OUTPUT = Path("run_output")  # EDIT 6 (published copy): scratch path redacted; the run wrote to a scratch folder
CHECKPOINT_HASH = "67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e"
MODEL_REVISION = "204698b45b3712590f06245fbfba32d3be539812"
# Development tolerance fixed before looking at newly produced scores; NOT a device parity tolerance.
SCALAR_ATOL = 0.002


def digest(path: Path) -> str:
    # EDIT 9: hashlib.file_digest needs Python 3.11; this laptop runs 3.10.11, so hash in chunks.
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inputs() -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    clean = pilot.load_clip(FRAMES / "clean" / "frames")
    hue = pilot.load_clip(FRAMES / "hue-030" / "frames")
    if not np.array_equal(clean[:12], hue[:12]):
        raise ValueError("archived fixture context mismatch")
    shuffled, permutation = pilot.target_shuffle(clean, "kubric-construct-validity-pilot-v1", "seed-230200")
    clips = {
        "clean": clean, "hue-030": hue,
        "target_hold": pilot.target_hold(clean), "target_shuffle": shuffled,
    }
    meta = {name: {"source_sha256": [digest(p) for p in sorted((FRAMES / name / "frames").glob("rgba_*.png"))]}
            for name in ("clean", "hue-030")}
    meta["target_hold"] = {"source_condition": "clean", "held_source_index_0based": 11}
    meta["target_shuffle"] = {"source_condition": "clean", "target_tubelet_permutation": permutation}
    return clips, meta


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    if core.revision(MODEL_ROOT) != MODEL_REVISION:
        raise ValueError("unexpected official model source revision")
    # EDIT 7: the 16 GB checkpoint is NOT re-hashed here; size is checked instead.
    # The SHA-256 (CHECKPOINT_HASH) is the value supplied by the user and is recorded in provenance.
    if CHECKPOINT.stat().st_size != 16464208210:
        raise ValueError("unexpected model checkpoint size")
    clips, meta = inputs()
    archive_hashes = {}
    with (PILOT / "full_artifacts/full-artifacts-sha256sums.txt").open() as handle:
        for line in handle:
            checksum, relative = line.split(maxsplit=1)
            archive_hashes[relative.strip()] = checksum
    for condition in ("clean", "hue-030"):
        for path, actual in zip(sorted((FRAMES / condition / "frames").glob("rgba_*.png")),
                                meta[condition]["source_sha256"]):
            relative = f"pilot/rendered/seed-230200/{condition}/frames/{path.name}"
            if archive_hashes.get(relative) != actual:
                raise ValueError(f"archived frame SHA-256 mismatch: {relative}")
    expected = {}
    with (PILOT / "model/scores.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scene_id"] == "seed-230200" and row["condition"] in clips:
                expected[row["condition"]] = float(row["scalar_l1"])
    if len(expected) != len(clips):
        raise ValueError("archived reference scores incomplete")
    sys.path.insert(0, str(MODEL_ROOT))
    from src.masks.utils import apply_masks
    spec = core.MODEL_SPECS["vjepa2-g"]
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with bf16 required; do not change model precision")
    free, total = torch.cuda.mem_get_info()
    if free < 3 * 2**30:  # EDIT 8: 4 GiB laptop card; the archived 5 GiB guard cannot be met on this device
        raise RuntimeError(f"insufficient GPU free memory for guarded attempt: {free} bytes")
    OUTPUT.mkdir(parents=True)
    provenance = {"status": "started", "role": "development_only_archived_pilot_regression",
                  "script_sha256": digest(Path(__file__)), "checkpoint_sha256": CHECKPOINT_HASH,
                  "vjepa2_revision": MODEL_REVISION, "model_spec": spec,
                  "gpu": torch.cuda.get_device_name(0), "gpu_free_bytes_before": free,
                  "torch": torch.__version__, "scalar_atol_precommitted": SCALAR_ATOL,
                  "condition_batch_size": 1, "loader": "verified_zip_index_disk_to_gpu_block_offload",
                  "activation_checkpointing": False,
                  "fixture_metadata": meta}
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    try:
        started = time.perf_counter()
        models = disk.DiskOffload(CHECKPOINT, MODEL_ROOT, spec)
        encoder, target_encoder, predictor = models.models
        provenance["model_load_seconds"] = time.perf_counter() - started
        provenance["load_detail"] = models.detail
        rows = []
        for name, clip in clips.items():
            # Hash model inputs after the exact canonical normalization, not only the PNGs.
            normalized = core.normalize_video(torch.from_numpy(clip.copy()), spec["resolution"])
            model_input_sha = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
            torch.cuda.reset_peak_memory_stats()
            x = normalized.permute(1, 0, 2, 3).unsqueeze(0).contiguous().to("cuda")
            enc_mask, pred_mask, full_mask = core.token_masks(12, spec, 1)
            tic = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                target_all = target_encoder(x, full_mask)
                raw = apply_masks(target_all, [pred_mask], concat=False)[0]
                targets = F.layer_norm(raw, (raw.size(-1),))
                del target_all, raw
                encoded = encoder(x, enc_mask)
                predictions = predictor(encoded, enc_mask, pred_mask)
                error = (predictions - targets).abs().float()
                score = error.mean()
                temporal = error.mean(-1).reshape(1, 18, 16, 16).mean((-1, -2))
            if not bool(torch.isfinite(error).all()):
                raise RuntimeError(f"nonfinite error: {name}")
            score_value = float(score)
            delta = score_value - expected[name]
            pred_digest = hashlib.sha256(predictions.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()
            rows.append({"scene_id": "seed-230200", "condition": name,
                         "scalar_l1": score_value, "archived_scalar_l1": expected[name],
                         "delta": delta, "temporal_tubelet_l1": temporal[0].cpu().tolist(),
                         "model_input_sha256": model_input_sha,
                         "predictor_bytes_sha256": pred_digest,
                         "inference_seconds": time.perf_counter()-tic,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            print(json.dumps({k: rows[-1][k] for k in ("condition", "scalar_l1", "delta", "peak_allocated_bytes")}), flush=True)
            del x, encoded, predictions, targets, error, score, temporal
            torch.cuda.empty_cache()
        (OUTPUT / "scores.json").write_text(json.dumps(rows, indent=2) + "\n")
        models.close()
        if len(set(r["predictor_bytes_sha256"] for r in rows)) != 1:
            raise RuntimeError("same-context predictor differs across target interventions")
        if any(abs(row["delta"]) > SCALAR_ATOL for row in rows):
            raise RuntimeError("archived pilot score regression failed tolerance")
        provenance["status"] = "pass"
    except BaseException as exc:
        provenance["status"] = "failed"
        provenance["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
