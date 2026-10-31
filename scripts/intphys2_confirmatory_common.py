from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def sha256_text_file(path: Path) -> str:
    return hashlib.sha256(canonical_text_bytes(path.read_bytes())).hexdigest()


def validate_retained_transform_audit(
    validation_path: Path, expected_sha256: str, metrics_path: Path
) -> dict:
    actual_hash = sha256_text_file(validation_path)
    if actual_hash != expected_sha256:
        raise ValueError("Retained full-audit validation SHA-256 mismatch")
    validation = json.loads(validation_path.read_text())
    if validation.get("status") != "passed" or validation.get(
        "transform_audit"
    ) not in {"active", "all"}:
        raise ValueError("Retained validation did not contain a full transform audit")
    if sha256_text_file(metrics_path) != validation.get("transform_metrics_sha256"):
        raise ValueError("Transform metrics differ from the retained full audit")
    return validation


def read_samples(path: Path, partition: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["partition"] == partition]
    if not rows:
        raise ValueError(f"No samples for partition {partition!r} in {path}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs")
    return rows


def decode_selected_frames(path: Path, indices: list[int]) -> np.ndarray:
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("Source-frame indices must be sorted and unique")
    try:
        from decord import VideoReader, cpu

        reader = VideoReader(str(path), ctx=cpu(0))
        if indices[-1] >= len(reader):
            raise ValueError(f"Frame {indices[-1]} exceeds {path} length {len(reader)}")
        return reader.get_batch(indices).asnumpy()
    except ModuleNotFoundError:
        import av

        needed = set(indices)
        decoded: dict[int, np.ndarray] = {}
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in needed:
                    decoded[index] = frame.to_ndarray(format="rgb24")
                if index >= indices[-1]:
                    break
        missing = [index for index in indices if index not in decoded]
        if missing:
            raise ValueError(f"Could not decode frames {missing[:5]} from {path}")
        return np.stack([decoded[index] for index in indices])


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.uint8)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    height, width = hsv.shape[:2]
    image = Image.frombytes("HSV", (width, height), hsv.tobytes())
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def apply_appearance(
    frames: np.ndarray,
    row: dict[str, str],
    shared_config: dict,
) -> np.ndarray:
    family = row["transform_family"]
    feasible = row["transform_feasible"].lower() == "true"
    if family == "clean" or not feasible:
        return frames.copy()
    value = float(row["transform_value"])
    if family == "brightness":
        return np.clip(np.rint(frames.astype(np.float32) * value), 0, 255).astype(
            np.uint8
        )
    output = []
    shift = int(round(value * 256.0 / 360.0))
    for frame in frames:
        hsv = rgb_to_hsv(frame).copy()
        shifted = (hsv[..., 0].astype(np.uint16) + shift) % 256
        if family == "hue":
            hsv[..., 0] = shifted
        elif family == "selective_recolor":
            if not row["selective_center_degrees"]:
                raise ValueError("Feasible selective recoloring lacks a hue center")
            center = float(row["selective_center_degrees"])
            hue_degrees = hsv[..., 0].astype(np.float32) * (360.0 / 256.0)
            distance = np.abs((hue_degrees - center + 180.0) % 360.0 - 180.0)
            selective = shared_config["selective_recolor"]
            mask = (distance <= float(selective["half_width_degrees"])) & (
                hsv[..., 1] >= int(selective["minimum_hsv_saturation_uint8"])
            )
            hsv[..., 0][mask] = shifted[mask]
        else:
            raise ValueError(f"Unknown appearance family {family!r}")
        output.append(hsv_to_rgb(hsv))
    return np.stack(output)


def coarse_token_summary(
    tokens, native_shape: tuple[int, int, int], output_shape: tuple[int, int, int]
):

    import torch

    if tokens.ndim != 3:
        raise ValueError(
            f"Expected [batch,tokens,channels], found {tuple(tokens.shape)}"
        )
    if int(np.prod(native_shape)) != tokens.shape[1]:
        raise ValueError((native_shape, tuple(tokens.shape)))
    for native, output in zip(native_shape, output_shape):
        if native % output:
            raise ValueError(
                f"Native grid {native_shape} is not divisible by {output_shape}"
            )
    batch, _, channels = tokens.shape
    nt, nh, nw = native_shape
    ot, oh, ow = output_shape
    grid = tokens.reshape(batch, nt, nh, nw, channels)
    grid = grid.reshape(
        batch,
        ot,
        nt // ot,
        oh,
        nh // oh,
        ow,
        nw // ow,
        channels,
    )
    return grid.mean(dim=(2, 4, 6)).reshape(batch, ot * oh * ow, channels)


def write_atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return sha256_file(path)


def validate_feature_shard(
    path: Path,
    *,
    expected_sample_id: str,
    layers: int,
    coarse_tokens: int,
    feature_dim: int,
    expected_metadata: dict[str, object] | None = None,
    expected_relative_depths: list[float] | None = None,
    expected_native_layers: list[int] | None = None,
) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "pooled",
            "coarse_tokens",
            "relative_depths",
            "native_layers",
            "metadata_json",
        }
        if set(payload.files) != required:
            raise ValueError(f"Unexpected arrays in {path}: {sorted(payload.files)}")
        pooled = payload["pooled"]
        coarse = payload["coarse_tokens"]
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata["sample_id"] != expected_sample_id:
            raise ValueError(f"Sample ID mismatch in {path}")
        if expected_metadata is not None:
            for key, expected in expected_metadata.items():
                if key not in metadata or metadata[key] != expected:
                    raise ValueError(
                        f"Metadata mismatch for {key!r} in {path}: "
                        f"{metadata.get(key)!r} != {expected!r}"
                    )
        if expected_relative_depths is not None and not np.array_equal(
            payload["relative_depths"],
            np.asarray(expected_relative_depths, dtype=np.float32),
        ):
            raise ValueError(f"Relative-depth mismatch in {path}")
        if expected_native_layers is not None and not np.array_equal(
            payload["native_layers"], np.asarray(expected_native_layers, dtype=np.int64)
        ):
            raise ValueError(f"Native-layer mismatch in {path}")
        if pooled.shape != (layers, feature_dim):
            raise ValueError(f"Pooled shape mismatch in {path}: {pooled.shape}")
        if coarse.shape != (layers, coarse_tokens, feature_dim):
            raise ValueError(f"Coarse-token shape mismatch in {path}: {coarse.shape}")
        if not np.isfinite(pooled).all() or not np.isfinite(coarse).all():
            raise ValueError(f"Non-finite features in {path}")
        if pooled.dtype != np.float16 or coarse.dtype != np.float16:
            raise ValueError(f"Feature dtype mismatch in {path}")
        return metadata
