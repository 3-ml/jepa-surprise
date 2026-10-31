#!/usr/bin/env python3
"""Audit native DINO-WM PushT prediction error under action and hue controls."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as tvf


ACTION_FACTORS = (1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0)
HUE_DEGREES = (30.0, 60.0, 120.0, 180.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-wm-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", default="dinowm-pusht-native-error-audit-v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def add_dinowm_safe_globals(root: Path) -> None:
    sys.path.insert(0, str(root.resolve()))
    import accelerate.optimizer as aopt
    import accelerate.state as ast
    import accelerate.utils.dataclasses as aud
    import torch.nn as nn
    import torch.optim as optim
    from models.proprio import ProprioceptiveEmbedding
    from models.vit import Attention, FeedForward, Transformer, ViTPredictor
    from models.vqvae import Decoder, Quantize, ResBlock, VQVAE

    safe = [
        ViTPredictor,
        Transformer,
        FeedForward,
        Attention,
        ProprioceptiveEmbedding,
        VQVAE,
        Decoder,
        ResBlock,
        Quantize,
        nn.Softmax,
        nn.LayerNorm,
        nn.ConvTranspose2d,
        nn.GELU,
        nn.Dropout,
        nn.ReLU,
        nn.Conv2d,
        nn.Sequential,
        dict,
        nn.Linear,
        nn.Conv1d,
        nn.ModuleList,
        optim.Adam,
        optim.AdamW,
        collections.defaultdict,
        ast.GradientState,
        ast.AcceleratorState,
        aud.DynamoBackend,
        aud.TorchDynamoPlugin,
        aud.DistributedType,
        aopt.AcceleratedOptimizer,
    ]
    torch.serialization.add_safe_globals(safe)


def load_model(args: argparse.Namespace, device: torch.device):
    import hydra
    from omegaconf import OmegaConf

    from models.visual_world_model import VWorldModel  # noqa: F401

    model_dir = args.checkpoint_root / "outputs" / "pusht"
    cfg = OmegaConf.load(model_dir / "hydra.yaml")
    ckpt_path = model_dir / "checkpoints" / "model_latest.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    encoder = hydra.utils.instantiate(cfg.encoder)
    model = hydra.utils.instantiate(
        cfg.model,
        encoder=encoder,
        proprio_encoder=ckpt["proprio_encoder"],
        action_encoder=ckpt["action_encoder"],
        predictor=ckpt["predictor"],
        decoder=ckpt["decoder"],
        proprio_dim=cfg.proprio_emb_dim,
        action_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=cfg.num_proprio_repeat,
    )
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, cfg, ckpt_path, int(ckpt["epoch"])


def load_dataset(args: argparse.Namespace):
    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import PushTDataset

    return PushTDataset(
        transform=default_transform(224),
        data_path=str(args.dataset_root / "pusht_noise" / args.split),
        normalize_action=True,
        with_velocity=True,
    )


def select_windows(dataset, samples: int, seed: str, num_frames: int, frameskip: int):
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big"))
    episodes = [i for i in range(len(dataset)) if dataset.get_seq_length(i) >= num_frames * frameskip]
    if samples > len(episodes):
        raise ValueError(f"Requested {samples} episodes, but only {len(episodes)} are long enough")
    selected_eps = rng.choice(episodes, size=samples, replace=False)
    rows = []
    for ep in selected_eps:
        max_start = dataset.get_seq_length(int(ep)) - num_frames * frameskip
        start = int(rng.integers(0, max_start + 1))
        rows.append({"episode": int(ep), "start_step": start})
    return sorted(rows, key=lambda row: (row["episode"], row["start_step"]))


def raw_frames(dataset, episode: int, frames: list[int]) -> torch.Tensor:
    from decord import VideoReader

    video = dataset.data_path / "obses" / f"episode_{episode:03d}.mp4"
    reader = VideoReader(str(video), num_threads=1)
    frames_thwc = reader.get_batch(frames)
    return frames_thwc.permute(0, 3, 1, 2).contiguous().to(torch.uint8)


def transform_frames(dataset, frames_uint8: torch.Tensor) -> torch.Tensor:
    return dataset.transform(frames_uint8.float() / 255.0)


def visual_difference(changed: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    absolute = (changed.float() - reference.float()).abs() / 255.0
    changed_pixels = (changed != reference).any(dim=0)
    return {
        "rgb_l1_01": float(absolute.mean()),
        "rgb_rmse_01": float(torch.sqrt((absolute**2).mean())),
        "changed_pixel_fraction": float(changed_pixels.float().mean()),
    }


def make_items(dataset, windows: list[dict], num_frames: int, frameskip: int) -> list[dict]:
    items = []
    for base_id, row in enumerate(windows):
        episode = row["episode"]
        start = row["start_step"]
        frame_indices = list(range(start, start + num_frames * frameskip, frameskip))
        raw = raw_frames(dataset, episode, frame_indices)
        visual = transform_frames(dataset, raw)
        proprio = dataset.proprios[episode, frame_indices].float()
        actions = dataset.actions[episode, start : start + num_frames * frameskip].float()
        actions = actions.reshape(num_frames, frameskip * dataset.action_dim)
        target_ref = raw[-1]

        for factor in ACTION_FACTORS:
            altered_actions = actions.clone()
            altered_actions[num_frames - 2] *= factor
            items.append(
                {
                    "base_id": base_id,
                    "episode": episode,
                    "start_step": start,
                    "condition": f"action_factor_{factor:g}",
                    "family": "action",
                    "value": float(factor),
                    "action_factor": float(factor),
                    "hue_degrees": float("nan"),
                    "visual": visual,
                    "proprio": proprio,
                    "actions": altered_actions,
                    "target_rgb_l1_from_clean_01": 0.0,
                    "target_changed_pixel_fraction": 0.0,
                    "target_sha256": hashlib.sha256(target_ref.numpy().tobytes()).hexdigest(),
                }
            )

        for degrees in HUE_DEGREES:
            hue_raw = raw.clone()
            hue_raw[-1] = tvf.adjust_hue(hue_raw[-1], degrees / 360.0)
            diff = visual_difference(hue_raw[-1], target_ref)
            items.append(
                {
                    "base_id": base_id,
                    "episode": episode,
                    "start_step": start,
                    "condition": f"hue_target_{int(degrees):03d}",
                    "family": "hue",
                    "value": float(degrees),
                    "action_factor": 1.0,
                    "hue_degrees": float(degrees),
                    "visual": transform_frames(dataset, hue_raw),
                    "proprio": proprio,
                    "actions": actions,
                    "target_rgb_l1_from_clean_01": diff["rgb_l1_01"],
                    "target_changed_pixel_fraction": diff["changed_pixel_fraction"],
                    "target_sha256": hashlib.sha256(hue_raw[-1].numpy().tobytes()).hexdigest(),
                }
            )
    return items


def score_items(model, items: list[dict], device: torch.device, batch_size: int) -> list[dict]:
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(items), batch_size):
            batch = items[offset : offset + batch_size]
            obs = {
                "visual": torch.stack([item["visual"] for item in batch]).to(device),
                "proprio": torch.stack([item["proprio"] for item in batch]).to(device),
            }
            act = torch.stack([item["actions"] for item in batch]).to(device)
            z = model.encode(obs, act)
            z_src = z[:, : model.num_hist]
            z_tgt = z[:, model.num_pred :]
            z_pred = model.predict(z_src)
            if model.concat_dim == 0:
                visual_pred = z_pred[:, :, :-2, :]
                visual_tgt = z_tgt[:, :, :-2, :]
                full_pred = z_pred[:, :, :-1, :]
                full_tgt = z_tgt[:, :, :-1, :]
            else:
                cutoff = -(model.proprio_dim + model.action_dim)
                visual_pred = z_pred[:, :, :, :cutoff]
                visual_tgt = z_tgt[:, :, :, :cutoff]
                full_pred = z_pred[:, :, :, :-model.action_dim]
                full_tgt = z_tgt[:, :, :, :-model.action_dim]
            visual_mse = (visual_pred - visual_tgt).pow(2).flatten(2).mean(dim=2)
            z_mse = (full_pred - full_tgt).pow(2).flatten(2).mean(dim=2)
            for local, item in enumerate(batch):
                rows.append(
                    {
                        "task": "pusht",
                        "model": "DINO-WM",
                        "base_id": item["base_id"],
                        "episode": item["episode"],
                        "start_step": item["start_step"],
                        "condition": item["condition"],
                        "family": item["family"],
                        "value": item["value"],
                        "action_factor": item["action_factor"],
                        "hue_degrees": item["hue_degrees"],
                        "prediction_mse": float(visual_mse[local, -1].cpu()),
                        "z_loss_mse": float(z_mse[local, -1].cpu()),
                        "temporal_visual_mse": json.dumps([float(x) for x in visual_mse[local].cpu()]),
                        "target_rgb_l1_from_clean_01": item["target_rgb_l1_from_clean_01"],
                        "target_changed_pixel_fraction": item["target_changed_pixel_fraction"],
                        "target_sha256": item["target_sha256"],
                    }
                )
    return rows


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n: int = 10_000):
    draws = rng.integers(0, len(values), size=(n, len(values)))
    return np.quantile(values[draws].mean(axis=1), [0.025, 0.975])


def summarize(rows: list[dict], output_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(20260901)
    baseline = df[df.condition == "action_factor_1"].set_index("base_id")["prediction_mse"]

    condition_rows = []
    contrast_rows = []
    for condition, group in df.groupby("condition", sort=False):
        values = group["prediction_mse"].to_numpy(float)
        low, high = bootstrap_mean(values, rng)
        condition_rows.append(
            {
                "condition": condition,
                "family": group["family"].iloc[0],
                "value": group["value"].iloc[0],
                "mean_prediction_mse": values.mean(),
                "ci_low": low,
                "ci_high": high,
                "mean_target_rgb_l1_from_clean_01": group["target_rgb_l1_from_clean_01"].mean(),
                "mean_target_changed_pixel_fraction": group["target_changed_pixel_fraction"].mean(),
                "n": len(group),
            }
        )
        paired = group.set_index("base_id")["prediction_mse"] - baseline
        low, high = bootstrap_mean(paired.to_numpy(float), rng)
        contrast_rows.append(
            {
                "condition": condition,
                "family": group["family"].iloc[0],
                "value": group["value"].iloc[0],
                "mean_delta_vs_lawful": paired.mean(),
                "ci_low": low,
                "ci_high": high,
                "n": paired.notna().sum(),
            }
        )

    pd.DataFrame(condition_rows).to_csv(output_dir / "condition_summary.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(output_dir / "paired_contrasts.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    add_dinowm_safe_globals(args.dino_wm_root)
    device = torch.device(args.device)
    model, cfg, ckpt_path, epoch = load_model(args, device)
    dataset = load_dataset(args)
    windows = select_windows(dataset, args.samples, args.seed, cfg.num_hist + cfg.num_pred, cfg.frameskip)
    items = make_items(dataset, windows, cfg.num_hist + cfg.num_pred, cfg.frameskip)
    rows = score_items(model, items, device, args.batch_size)

    fieldnames = list(rows[0])
    with (args.output_dir / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "selection_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["base_id", "episode", "start_step"])
        writer.writeheader()
        writer.writerows({"base_id": i, **row} for i, row in enumerate(windows))
    summarize(rows, args.output_dir)
    provenance = {
        "script": str(Path(__file__).resolve()),
        "command": " ".join(sys.argv),
        "wall_seconds": time.time() - started,
        "dino_wm_repo": str(args.dino_wm_root.resolve()),
        "dino_wm_revision": git_revision(args.dino_wm_root),
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_epoch": epoch,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_split": args.split,
        "samples": args.samples,
        "seed": args.seed,
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "score": "final-step DINO-WM visual latent MSE against the encoded target",
        "intervention_note": "Action factors alter only the final context action block while target frames are held fixed; hue interventions alter only the target frame while actions/proprio are held fixed.",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (args.output_dir / "README.md").write_text(
        "# DINO-WM PushT native-error audit\n\n"
        "Exploratory extension of the LeWM construct-validity audit to the released DINO-WM PushT checkpoint. "
        "The audit preserves the target video and changes the final context action block, then compares those "
        "native visual-latent MSE changes with target-only hue interventions.\n"
    )


if __name__ == "__main__":
    main()



