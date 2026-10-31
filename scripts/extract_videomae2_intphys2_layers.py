#!/usr/bin/env python3
"""Extract locked VideoMAE2-G layer summaries for IntPhys2 confirmatory samples."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

try:
    import intphys2_confirmatory_common as common
    import intphys2_confirmatory_extraction as extraction
except ModuleNotFoundError:
    from scripts import intphys2_confirmatory_common as common
    from scripts import intphys2_confirmatory_extraction as extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    extraction.add_common_args(parser)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser.parse_args()


class VideoMAE2Extractor:
    def __init__(self, model, processor, coarse_shape: tuple[int, int, int]):
        self.wrapper = model
        self.model = model.model
        self.processor = processor
        self.coarse_shape = coarse_shape
        self.native_chunk_shape = (8, 16, 16)
        self.concatenated_shape = (32, 16, 16)
        if len(self.model.blocks) != 40:
            raise ValueError(f"Expected 40 blocks, found {len(self.model.blocks)}")

    def _concatenate_chunks(self, tokens: torch.Tensor, batch: int) -> torch.Tensor:
        expected_tokens = int(np.prod(self.native_chunk_shape))
        if tokens.shape[0] != batch * 4 or tokens.shape[1] != expected_tokens:
            raise ValueError(f"Unexpected VideoMAE2 token shape {tuple(tokens.shape)}")
        channels = tokens.shape[-1]
        grid = tokens.reshape(batch, 4, *self.native_chunk_shape, channels)
        return grid.reshape(batch, *self.concatenated_shape, channels).reshape(
            batch, int(np.prod(self.concatenated_shape)), channels
        )

    def _summarize(
        self, tokens: torch.Tensor, batch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        concatenated = self._concatenate_chunks(tokens.detach().float(), batch)
        pooled = concatenated.mean(dim=1).cpu()
        coarse = common.coarse_token_summary(
            concatenated, self.concatenated_shape, self.coarse_shape
        ).cpu()
        return pooled, coarse

    def preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
        chunks = []
        for sample in frames:
            if sample.shape[0] != 64:
                raise ValueError(f"Expected 64 shared frames, found {sample.shape}")
            for start in range(0, 64, 16):
                chunks.append(list(sample[start : start + 16]))
        processed = self.processor(chunks, return_tensors="pt")
        pixels = processed["pixel_values"]
        if pixels.ndim != 5:
            raise ValueError(f"Unexpected processor output {tuple(pixels.shape)}")
        if pixels.shape[1] == 16 and pixels.shape[2] == 3:
            pixels = pixels.permute(0, 2, 1, 3, 4)
        if pixels.shape[1:] != (3, 16, 224, 224):
            raise ValueError(f"Unexpected native input shape {tuple(pixels.shape)}")
        return pixels.contiguous()

    def infer(self, frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        batch = len(frames)
        pixels = self.preprocess(frames).to("cuda", non_blocking=True)
        captures: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = self.model.patch_embed(pixels)
                if self.model.pos_embed is not None:
                    tokens = tokens + self.model.pos_embed.expand(
                        tokens.shape[0], -1, -1
                    ).type_as(tokens).to(tokens.device)
                tokens = self.model.pos_drop(tokens)
                captures[0] = self._summarize(tokens, batch)
                for index, block in enumerate(self.model.blocks, start=1):
                    tokens = block(tokens)
                    if index in {10, 20, 30}:
                        captures[index] = self._summarize(tokens, batch)
                final = self.model.norm(tokens)
                captures[40] = self._summarize(final, batch)
        finally:
            del pixels
        expected = [0, 10, 20, 30, 40]
        if sorted(captures) != expected:
            raise RuntimeError(f"Missing layer captures: {sorted(captures)}")
        pooled = np.stack([captures[layer][0].numpy() for layer in expected], axis=1)
        coarse = np.stack([captures[layer][1].numpy() for layer in expected], axis=1)
        return pooled, coarse


def validate_checkpoint_key_partition(
    expected_keys: set[str], checkpoint_keys: set[str], model_config: dict
) -> set[str]:
    expected_count = int(model_config["checkpoint_encoder_tensor_count"])
    if len(expected_keys) != expected_count:
        raise ValueError(
            f"Expected {expected_count} encoder tensors, constructed {len(expected_keys)}"
        )
    missing = expected_keys - checkpoint_keys
    excluded = checkpoint_keys - expected_keys
    if missing:
        raise ValueError(f"VideoMAE2 checkpoint lacks encoder tensors: {sorted(missing)[:5]}")
    allowed_prefixes = tuple(model_config["checkpoint_excluded_prefixes"])
    allowed_keys = set(model_config["checkpoint_excluded_keys"])
    invalid_excluded = {
        key
        for key in excluded
        if key not in allowed_keys and not key.startswith(allowed_prefixes)
    }
    if invalid_excluded or len(excluded) != int(
        model_config["checkpoint_excluded_tensor_count"]
    ):
        raise ValueError(
            "VideoMAE2 checkpoint contains an unexpected non-encoder tensor set: "
            f"count={len(excluded)}, invalid={sorted(invalid_excluded)[:5]}"
        )
    return excluded


def load_model(args: argparse.Namespace, model_config: dict):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint = checkpoint_dir / model_config["checkpoint_filename"]
    if checkpoint.stat().st_size != int(model_config["checkpoint_bytes"]):
        raise ValueError("VideoMAE2 checkpoint byte count mismatch")
    checkpoint_hash = common.sha256_file(checkpoint)
    if checkpoint_hash != model_config["checkpoint_sha256"]:
        raise ValueError("VideoMAE2 checkpoint SHA-256 mismatch")
    for filename, expected_hash in model_config["snapshot_file_sha256s"].items():
        path = checkpoint_dir / filename
        if common.sha256_file(path) != expected_hash:
            raise ValueError(f"Pinned VideoMAE2 snapshot hash mismatch: {filename}")
    modeling_path = checkpoint_dir / "modeling_videomaev2.py"

    from safetensors import safe_open
    from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor
    import transformers

    if transformers.__version__ != model_config["runtime_transformers_version"]:
        raise ValueError(
            f"Transformers version {transformers.__version__} differs from the locked "
            f"{model_config['runtime_transformers_version']}"
        )
    config = AutoConfig.from_pretrained(
        checkpoint_dir, trust_remote_code=True, local_files_only=True
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = AutoModel.from_config(config, trust_remote_code=True)
    finally:
        torch.set_default_dtype(previous_dtype)

    expected_keys = set(model.state_dict())
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        checkpoint_keys = set(handle.keys())
        excluded = validate_checkpoint_key_partition(
            expected_keys, checkpoint_keys, model_config
        )
        encoder_state = {
            key: handle.get_tensor(key) for key in sorted(expected_keys)
        }
    load_result = model.load_state_dict(encoder_state, strict=True, assign=True)
    del encoder_state
    gc.collect()
    if model.model.fc_norm is not None or isinstance(model.model.norm, torch.nn.Identity):
        raise ValueError("Pinned VideoMAE2 final token normalization semantics changed")
    model = model.to("cuda", dtype=torch.bfloat16).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = VideoMAEImageProcessor.from_pretrained(
        checkpoint_dir, local_files_only=True
    )
    return model, processor, checkpoint_hash, {
        "strict_encoder_load": str(load_result),
        "encoder_tensor_count": len(expected_keys),
        "excluded_checkpoint_tensor_count": len(excluded),
        "excluded_checkpoint_tensors": sorted(excluded),
        "huggingface_revision": model_config["huggingface_revision"],
        "modeling_sha256": common.sha256_file(modeling_path),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "transformers": transformers.__version__,
    }


def main() -> None:
    args = parse_args()
    shared, model_config, _ = extraction.load_locked_contract(args, "videomae2")
    torch.manual_seed(230826)
    torch.cuda.manual_seed_all(230826)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, processor, checkpoint_hash, load_provenance = load_model(args, model_config)
    extractor = VideoMAE2Extractor(
        model,
        processor,
        tuple(shared["output_views"]["coarse_tokens"]["shape"]),
    )

    def infer_features(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return extractor.infer(frames)

    provenance = extraction.run_extraction(
        args,
        model_key="videomae2",
        checkpoint_path=args.checkpoint_dir.resolve()
        / model_config["checkpoint_filename"],
        checkpoint_hash=checkpoint_hash,
        infer_features=infer_features,
        load_provenance=load_provenance,
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
