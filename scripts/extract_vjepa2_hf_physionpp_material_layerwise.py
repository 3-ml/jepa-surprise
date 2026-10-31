#!/usr/bin/env python3
"""Extract HF V-JEPA2 layerwise features for local Physion++ material probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import VJEPA2Model, VJEPA2VideoProcessor

try:
    from extract_vjepa2_features import read_video
except ModuleNotFoundError:  # Imported as scripts.extract_vjepa2_hf_physionpp_material_layerwise
    from scripts.extract_vjepa2_features import read_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, default=Path("data/physionpp/mass_only"))
    parser.add_argument(
        "--clean-manifest",
        type=Path,
        default=Path("data/physionpp/mass_only/manifest_mass.csv"),
    )
    parser.add_argument(
        "--shift-root",
        type=Path,
        default=Path("data/physionpp/mass_collision_material_full"),
    )
    parser.add_argument(
        "--shift-manifest",
        type=Path,
        default=Path(
            "data/physionpp/mass_collision_material_full/manifest_material_mass_collision.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16" if torch.cuda.is_available() else "float32",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def project_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_model_snapshot(model_id: str, local_files_only: bool) -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, local_files_only=local_files_only))
    except Exception:
        return None


def write_atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def clean_rows_by_source(clean_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["rgb_mp4"]: row for row in clean_rows}


def build_samples(args: argparse.Namespace) -> list[dict[str, object]]:
    clean_by_source = clean_rows_by_source(read_csv(args.clean_manifest))
    shifted = read_csv(args.shift_manifest)
    grouped: dict[str, dict[str, object]] = {}
    for row in shifted:
        source = row["source_rgb_mp4"]
        if source not in clean_by_source:
            raise ValueError(f"Shifted source is absent from clean manifest: {source}")
        entry = grouped.setdefault(
            source,
            {
                "clean": clean_by_source[source],
                "variants": {},
            },
        )
        entry["variants"][row["material_variant"]] = row

    sources = sorted(grouped)
    if args.max_sources is not None:
        if args.max_sources < 1:
            raise ValueError("--max-sources must be positive")
        sources = sources[: args.max_sources]

    samples: list[dict[str, object]] = []
    for source in sources:
        entry = grouped[source]
        clean = entry["clean"]
        split = str(clean["split"])
        base_id = f"{split}|{source}"
        label_value = int(clean["label"])
        label = f"label_{label_value}"
        samples.append(
            {
                "sample_id": f"{base_id}|clean",
                "name": source,
                "file_name": source,
                "pair_id": base_id,
                "pair_label": label,
                "pair_label_value": label_value,
                "appearance_condition": "clean",
                "split": split,
                "scenario": clean.get("scenario", ""),
                "condition": clean.get("scenario", ""),
                "difficulty": "",
                "camera": "",
                "window_count": 1,
                "video_path": args.clean_root / source,
                "source_rgb_mp4": source,
                "material_variant": "clean",
            }
        )
        variants = entry["variants"]
        for variant in ("metal", "foam"):
            if variant not in variants:
                raise ValueError(f"Missing {variant} variant for {source}")
            row = variants[variant]
            shifted_label = int(row["label"])
            if shifted_label != label_value:
                raise ValueError(f"Label mismatch for {source} {variant}")
            samples.append(
                {
                    "sample_id": f"{base_id}|{variant}",
                    "name": source,
                    "file_name": row["rgb_mp4"],
                    "pair_id": base_id,
                    "pair_label": label,
                    "pair_label_value": label_value,
                    "appearance_condition": variant,
                    "split": split,
                    "scenario": row.get("scenario", ""),
                    "condition": row.get("scenario", ""),
                    "difficulty": "",
                    "camera": "",
                    "window_count": 1,
                    "video_path": args.shift_root / row["rgb_mp4"],
                    "source_rgb_mp4": source,
                    "material_variant": variant,
                }
            )
    missing = [str(sample["video_path"]) for sample in samples if not Path(sample["video_path"]).exists()]
    if missing:
        raise FileNotFoundError("Missing videos:\n" + "\n".join(missing[:20]))
    return samples


def extract_sample(
    sample: dict[str, object],
    *,
    processor: VJEPA2VideoProcessor,
    model: VJEPA2Model,
    frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    video = read_video(Path(sample["video_path"]), frames)
    batch = processor([video], return_tensors="pt")
    inputs = batch["pixel_values_videos"].to(device)
    if device.type == "cuda":
        inputs = inputs.to(dtype=dtype)
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32
    ):
        outputs = model(inputs, skip_predictor=True, output_hidden_states=True)
    if outputs.hidden_states is None:
        raise RuntimeError("Model did not return hidden states")
    pooled = torch.stack(
        [state.detach().float().mean(dim=1).squeeze(0).cpu() for state in outputs.hidden_states],
        dim=0,
    )
    if not bool(torch.isfinite(pooled).all()):
        raise RuntimeError(f"Non-finite features for {sample['sample_id']}")
    del inputs, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pooled.numpy().astype(np.float32)


def write_samples_csv(path: Path, samples: list[dict[str, object]]) -> None:
    fields = [
        "sample_id",
        "name",
        "file_name",
        "pair_id",
        "pair_label",
        "pair_label_value",
        "appearance_condition",
        "split",
        "scenario",
        "condition",
        "difficulty",
        "camera",
        "window_count",
        "source_rgb_mp4",
        "material_variant",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for sample in samples:
            writer.writerow({field: sample[field] for field in fields})
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    representations_path = args.output_dir / "representations.npz"
    if representations_path.exists() and not args.overwrite:
        raise FileExistsError(f"{representations_path} exists; pass --overwrite to replace")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = getattr(torch, args.dtype)
    started = time.perf_counter()
    samples = build_samples(args)
    model_snapshot = cached_model_snapshot(args.model_id, not args.allow_download)
    model_source = str(model_snapshot) if model_snapshot is not None else args.model_id
    processor = VJEPA2VideoProcessor.from_pretrained(
        model_source,
        local_files_only=not args.allow_download,
    )
    model = VJEPA2Model.from_pretrained(
        model_source,
        local_files_only=not args.allow_download,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    features = []
    for index, sample in enumerate(samples, start=1):
        features.append(
            extract_sample(
                sample,
                processor=processor,
                model=model,
                frames=args.frames,
                device=device,
                dtype=dtype,
            )
        )
        print(
            json.dumps(
                {
                    "done": index,
                    "total": len(samples),
                    "sample_id": sample["sample_id"],
                }
            ),
            flush=True,
        )

    x = np.stack(features).astype(np.float32)
    layer_names = np.array(
        ["embedding"] + [f"block_{index:02d}" for index in range(x.shape[1] - 1)]
    )
    metadata_fields = [
        "sample_id",
        "name",
        "file_name",
        "pair_id",
        "pair_label",
        "pair_label_value",
        "appearance_condition",
        "split",
        "scenario",
        "condition",
        "difficulty",
        "camera",
        "window_count",
        "source_rgb_mp4",
        "material_variant",
    ]
    arrays = {
        "X": x,
        "layer_names": layer_names,
        "metadata_columns": np.asarray(metadata_fields, dtype=str),
    }
    for field in metadata_fields:
        values = [sample[field] for sample in samples]
        if field in {"pair_label_value", "window_count"}:
            arrays[field] = np.asarray(values, dtype=np.int64)
        else:
            arrays[field] = np.asarray(values, dtype=str)
    temporary_npz = representations_path.with_suffix(".npz.tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary_npz.replace(representations_path)
    write_samples_csv(args.output_dir / "samples.csv", samples)

    provenance = {
        "schema_version": 1,
        "script": "scripts/extract_vjepa2_hf_physionpp_material_layerwise.py",
        "script_sha256": sha256_file(Path(__file__)),
        "project_revision": project_revision(),
        "diagnostic_role": (
            "local HF V-JEPA2-L Physion++ material probe diagnostic; not the locked "
            "V-JEPA2-G IntPhys2 paper-facing extraction"
        ),
        "model_id": args.model_id,
        "model_snapshot": str(model_snapshot) if model_snapshot is not None else None,
        "model_safetensors_sha256": (
            sha256_file(model_snapshot / "model.safetensors")
            if model_snapshot is not None and (model_snapshot / "model.safetensors").exists()
            else None
        ),
        "model_config": {
            "frames_per_clip": int(model.config.frames_per_clip),
            "hidden_size": int(model.config.hidden_size),
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "patch_size": int(model.config.patch_size),
            "tubelet_size": int(model.config.tubelet_size),
        },
        "clean_manifest": str(args.clean_manifest),
        "clean_manifest_sha256": sha256_file(args.clean_manifest),
        "shift_manifest": str(args.shift_manifest),
        "shift_manifest_sha256": sha256_file(args.shift_manifest),
        "clean_root": str(args.clean_root),
        "shift_root": str(args.shift_root),
        "frames": args.frames,
        "max_sources": args.max_sources,
        "samples": int(x.shape[0]),
        "source_videos": int(len({sample["pair_id"] for sample in samples})),
        "appearance_conditions": sorted({str(sample["appearance_condition"]) for sample in samples}),
        "representations_shape": [int(value) for value in x.shape],
        "feature_pooling": "mean encoder tokens per hidden state; one uniformly sampled clip per video",
        "local_files_only": not args.allow_download,
        "device": str(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "wall_seconds": time.perf_counter() - started,
    }
    write_atomic_text(args.output_dir / "provenance.json", json.dumps(provenance, indent=2) + "\n")
    write_atomic_text(
        args.output_dir / "status.json",
        json.dumps(
            {
                "status": "complete",
                "samples": int(x.shape[0]),
                "layers": int(x.shape[1]),
                "feature_dim": int(x.shape[2]),
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
