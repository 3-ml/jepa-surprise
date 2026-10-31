from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import intphys2_confirmatory_common as common
    import intphys2_confirmatory_extraction as extraction
except ModuleNotFoundError:
    from scripts import intphys2_confirmatory_common as common
    from scripts import intphys2_confirmatory_extraction as extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    extraction.add_common_args(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa2-root", type=Path, required=True)
    return parser.parse_args()


def clean_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module.backbone."): value for key, value in state.items()}


def normalize_batch(frames: list[np.ndarray], resolution: int) -> torch.Tensor:
    arrays = np.stack(frames)
    tensor = torch.from_numpy(arrays).float()
    batch, steps, height, width, channels = tensor.shape
    tensor = tensor.reshape(batch * steps, height, width, channels).permute(0, 3, 1, 2)
    tensor = F.interpolate(
        tensor, size=(resolution, resolution), mode="bilinear", align_corners=False
    )
    tensor = tensor.reshape(batch, steps, channels, resolution, resolution)
    tensor = tensor.permute(0, 2, 1, 3, 4).contiguous() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    return (tensor - mean) / std


class VJEPASummaryCapture:
    def __init__(self, model: torch.nn.Module, coarse_shape: tuple[int, int, int]):
        self.model = model
        self.coarse_shape = coarse_shape
        self.native_shape = (32, 16, 16)
        self.captures: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        modules = {
            0: model.patch_embed,
            10: model.blocks[9],
            20: model.blocks[19],
            30: model.blocks[29],
        }
        self.handles = [
            module.register_forward_hook(self._hook(layer))
            for layer, module in modules.items()
        ]

    def _summarize(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = tokens.detach().float()
        pooled = values.mean(dim=1).cpu()
        coarse = common.coarse_token_summary(
            values, self.native_shape, self.coarse_shape
        ).cpu()
        return pooled, coarse

    def _hook(self, layer: int):
        def capture(_module, _inputs, output) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self.captures[layer] = self._summarize(tensor)

        return capture

    def infer(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        self.captures.clear()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            final = self.model(x)
        self.captures[40] = self._summarize(final)
        expected = [0, 10, 20, 30, 40]
        if sorted(self.captures) != expected:
            raise RuntimeError(f"Missing layer captures: {sorted(self.captures)}")
        pooled = np.stack(
            [self.captures[layer][0].numpy() for layer in expected], axis=1
        )
        coarse = np.stack(
            [self.captures[layer][1].numpy() for layer in expected], axis=1
        )
        return pooled, coarse

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def load_encoder(args: argparse.Namespace, model_config: dict):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = args.checkpoint.resolve()
    actual_hash = common.sha256_file(checkpoint)
    if actual_hash != model_config["checkpoint_sha256"]:
        raise ValueError("V-JEPA2 checkpoint SHA-256 mismatch")
    root = args.vjepa2_root.resolve()
    revision = model_config["upstream_revision"]
    sys.path.insert(0, str(root))
    from src.models import vision_transformer as vit

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    architecture = getattr(vit, model_config["architecture"])
    encoder = architecture(
        img_size=int(model_config["native_resolution"]),
        num_frames=int(model_config["native_frames_per_clip"]),
        uniform_power=True,
        use_activation_checkpointing=False,
        use_rope=True,
    )
    load_result = encoder.load_state_dict(
        clean_state(payload[model_config["checkpoint_container_key"]]), strict=True
    )
    checkpoint_keys = sorted(payload)
    del payload
    gc.collect()
    encoder = encoder.to("cuda").eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return (
        encoder,
        actual_hash,
        {
            "strict_load": str(load_result),
            "checkpoint_keys": checkpoint_keys,
            "upstream_revision": revision,
            "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        },
    )


def main() -> None:
    args = parse_args()
    _, model_config, _ = extraction.load_locked_contract(args, "vjepa2")
    torch.manual_seed(230826)
    torch.cuda.manual_seed_all(230826)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    encoder, checkpoint_hash, load_provenance = load_encoder(args, model_config)
    coarse_shape = tuple(
        json.loads(args.shared_config.read_text())["output_views"]["coarse_tokens"][
            "shape"
        ]
    )
    capture = VJEPASummaryCapture(encoder, coarse_shape)

    def infer_features(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        normalized = normalize_batch(frames, int(model_config["native_resolution"]))
        x = normalized.to("cuda", non_blocking=True)
        try:
            return capture.infer(x)
        finally:
            del x, normalized

    try:
        provenance = extraction.run_extraction(
            args,
            model_key="vjepa2",
            checkpoint_path=args.checkpoint.resolve(),
            checkpoint_hash=checkpoint_hash,
            infer_features=infer_features,
            load_provenance=load_provenance,
        )
    finally:
        capture.close()
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
