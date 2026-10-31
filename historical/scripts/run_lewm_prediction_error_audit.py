from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvf

TASKS = {
    "tworoom": {
        "model_repo": "quentinll/lewm-tworooms",
        "model_revision": "77adaae0bc31deab21c93740d1f8bb947cd0bdec",
        "weights_sha256": "566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd",
        "dataset_revision": "6903a2de048b13819d812da0b4dd661290bc01e4",
        "dataset_blob_sha256": "494b1a02f0765cd9a0d9daf1786c419ced1009977fc45d01e3158932f8d080ca",
        "state_key": "proprio",
        "episode_key": "ep_idx",
    },
    "pusht": {
        "model_repo": "quentinll/lewm-pusht",
        "model_revision": "22b330c28c27ead4bfd1888615af1340e3fe9052",
        "weights_sha256": "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607",
        "dataset_revision": "655cd446b9929369d7d406001da85c15d1457850",
        "dataset_blob_sha256": "7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318",
        "state_key": "state",
        "episode_key": "episode_idx",
    },
}

ACTION_FACTORS = (0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0)
HUE_DEGREES = (5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 180.0)
BRIGHTNESS_FACTORS = (0.5, 0.7, 0.85, 0.9, 0.95, 0.98, 1.05, 1.15)
INVARIANCE_ATOL = 1e-5


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    scope: str
    value: float
    action_mode: str = "clean"


def conditions(
    action_factors: tuple[float, ...] = ACTION_FACTORS,
    hue_degrees: tuple[float, ...] = HUE_DEGREES,
    brightness_factors: tuple[float, ...] = BRIGHTNESS_FACTORS,
) -> tuple[Condition, ...]:
    rows = [Condition("clean", "control", "none", 1.0)]
    for value in action_factors:
        if value == 0:
            name = "action_zero_final"
        elif value == -1:
            name = "action_opposite_final"
        else:
            token = str(value).replace("-", "neg").replace(".", "p")
            name = f"action_scale_{token}_final"
        rows.append(Condition(name, "action", "final_block", value, "scale_final"))
    for value in hue_degrees:
        token = f"{int(value):03d}"
        rows.extend(
            [
                Condition(f"hue_target_{token}", "hue", "target", value),
                Condition(f"hue_persistent_{token}", "hue", "persistent", value),
            ]
        )
    for value in brightness_factors:
        token = str(value).replace(".", "p")
        rows.extend(
            [
                Condition(f"brightness_target_{token}", "brightness", "target", value),
                Condition(
                    f"brightness_persistent_{token}",
                    "brightness",
                    "persistent",
                    value,
                ),
            ]
        )
    return tuple(rows)


def apply_condition_pixels(frames: torch.Tensor, condition: Condition) -> torch.Tensor:

    if frames.ndim != 4 or frames.shape[1] != 3 or frames.dtype != torch.uint8:
        raise ValueError(
            f"Expected uint8 [T,3,H,W], found {frames.dtype} {frames.shape}"
        )
    output = frames.clone()
    if condition.family not in {"hue", "brightness"}:
        return output
    indices = (
        [frames.shape[0] - 1] if condition.scope == "target" else range(frames.shape[0])
    )
    for index in indices:
        if condition.family == "hue":
            output[index] = tvf.adjust_hue(output[index], condition.value / 360.0)
        else:
            output[index] = tvf.adjust_brightness(output[index], condition.value)
    return output


def apply_condition_actions(
    actions: torch.Tensor, condition: Condition
) -> torch.Tensor:

    if actions.ndim != 2 or actions.shape[0] < 2:
        raise ValueError(f"Expected [T,A] actions, found {actions.shape}")
    output = actions.clone()
    target_leading_index = actions.shape[0] - 2
    if condition.action_mode == "scale_final":
        output[target_leading_index] *= condition.value
    elif condition.action_mode != "clean":
        raise ValueError(f"Unknown action mode: {condition.action_mode}")
    return output


def visual_difference(
    changed: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:

    if changed.shape != reference.shape or changed.shape[-3] != 3:
        raise ValueError(
            "Visual-difference inputs must have matching [...,3,H,W] shapes"
        )
    absolute = (changed.float() - reference.float()).abs() / 255.0
    changed_pixels = (changed != reference).any(dim=-3)
    per_pixel_l1 = absolute.mean(dim=-3)
    return {
        "rgb_l1_01": float(absolute.mean()),
        "rgb_rmse_01": float(torch.sqrt((absolute**2).mean())),
        "changed_pixel_fraction": float(changed_pixels.float().mean()),
        "changed_pixel_rgb_l1_01": float(
            per_pixel_l1[changed_pixels].mean() if changed_pixels.any() else 0.0
        ),
    }


def normalize_action_blocks(
    actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:

    if actions.shape[-1] % mean.numel() != 0:
        raise ValueError("Flattened action dimension is incompatible with statistics")
    shape = (*actions.shape[:-1], -1, mean.numel())
    normalized = (actions.reshape(shape) - mean) / std

    return torch.nan_to_num(normalized.reshape(actions.shape).float(), 0.0)


def select_clip_indices(
    dataset, count: int, seed: str, excluded_episodes: set[int] | None = None
) -> list[int]:

    by_episode: dict[int, list[int]] = defaultdict(list)
    for index, (episode, _start) in enumerate(dataset.clip_indices):
        by_episode[int(episode)].append(index)
    for episode in excluded_episodes or set():
        by_episode.pop(int(episode), None)
    if count > len(by_episode):
        raise ValueError(f"Requested {count} episodes but only {len(by_episode)} exist")
    material = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    rng = np.random.default_rng(material)
    episodes = rng.choice(sorted(by_episode), size=count, replace=False)
    selected = [int(rng.choice(by_episode[int(episode)])) for episode in episodes]
    return sorted(selected, key=lambda index: dataset.clip_indices[index])


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def archive_revision(root: Path) -> str:
    return "anonymous-source-snapshot"


def state_metrics(task: str, state: torch.Tensor) -> dict[str, float]:
    if task == "tworoom":
        displacement = torch.linalg.vector_norm(state[-1, :2] - state[-2, :2])
        return {
            "agent_displacement_state": float(displacement),
            "agent_displacement_diameters": float(displacement / 14.0),
            "block_displacement_state": float("nan"),
            "block_angle_change_radians": float("nan"),
        }
    agent = torch.linalg.vector_norm(state[-1, :2] - state[-2, :2])
    block = torch.linalg.vector_norm(state[-1, 2:4] - state[-2, 2:4])
    angle = (
        torch.remainder(state[-1, 4] - state[-2, 4] + torch.pi, 2 * torch.pi) - torch.pi
    )
    return {
        "agent_displacement_state": float(agent),
        "agent_displacement_diameters": float(agent / 30.0),
        "block_displacement_state": float(block),
        "block_angle_change_radians": float(angle.abs()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", default="lewm-prediction-error-audit-v1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--phase", choices=["exploratory", "confirmatory"], default="exploratory"
    )
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    return parser.parse_args()


def load_model(task: str, device: torch.device):
    import hydra
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf

    spec = TASKS[task]
    snapshot = Path(
        snapshot_download(
            repo_id=spec["model_repo"],
            revision=spec["model_revision"],
            allow_patterns=["weights.pt", "config.json", "README.md"],
        )
    )
    weights = snapshot / "weights.pt"
    digest = sha256_file(weights)
    if digest != spec["weights_sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {digest}")
    config = OmegaConf.create(json.loads((snapshot / "config.json").read_text()))
    model = hydra.utils.instantiate(config)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval().requires_grad_(False)
    return model, snapshot, config, len(state)


def make_dataset(task: str, path: Path):
    import stable_pretraining as spt
    from stable_worldmodel.data import HDF5Dataset

    state_key = TASKS[task]["state_key"]
    episode_key = TASKS[task]["episode_key"]
    keys = ["pixels", "action", state_key, episode_key, "step_idx"]
    dataset = HDF5Dataset(
        path=str(path),
        frameskip=5,
        num_steps=4,
        keys_to_load=keys,
        keys_to_cache=["action", state_key, episode_key, "step_idx"],
    )
    raw_action = torch.from_numpy(np.asarray(dataset.get_col_data("action"))).float()
    raw_action = raw_action[~torch.isnan(raw_action).any(dim=1)]
    action_mean = raw_action.mean(dim=0)
    action_std = raw_action.std(dim=0)
    preprocess = spt.data.transforms.Compose(
        spt.data.transforms.ToImage(
            **spt.data.dataset_stats.ImageNet, source="pixels", target="pixels"
        ),
        spt.data.transforms.Resize(224, source="pixels", target="pixels"),
    )
    return dataset, action_mean, action_std, preprocess


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("samples and batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    started = time.time()

    model, snapshot, model_config, state_dict_keys = load_model(args.task, device)
    dataset, action_mean, action_std, preprocess = make_dataset(args.task, args.dataset)
    excluded_episodes: set[int] = set()
    for manifest_path in args.exclude_manifest:
        with manifest_path.open(newline="") as handle:
            excluded_episodes.update(
                int(row["episode"]) for row in csv.DictReader(handle)
            )
    selected = select_clip_indices(
        dataset,
        args.samples,
        f"{args.seed}:{args.task}",
        excluded_episodes=excluded_episodes,
    )
    condition_rows = conditions()

    identity = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
    assert torch.equal(
        identity,
        apply_condition_pixels(identity, Condition("identity", "hue", "target", 0.0)),
    )
    assert torch.equal(
        identity,
        apply_condition_pixels(
            identity, Condition("identity", "brightness", "persistent", 1.0)
        ),
    )

    manifest: list[dict] = []
    base_samples: list[dict] = []
    state_key = TASKS[args.task]["state_key"]
    episode_key = TASKS[args.task]["episode_key"]
    for dataset_index in selected:
        sample = dataset[dataset_index]
        episode = int(sample[episode_key][0])
        start_step = int(sample["step_idx"][0])
        base_id = f"{args.task}:{episode}:{start_step}"
        raw_pixels = sample["pixels"].to(torch.uint8)
        raw_actions = sample["action"].float()
        metrics = state_metrics(args.task, sample[state_key].float())
        source_change = visual_difference(raw_pixels[-1], raw_pixels[-2])
        clean_target_hash = hashlib.sha256(raw_pixels[-1].numpy().tobytes()).hexdigest()
        manifest.append(
            {
                "base_id": base_id,
                "dataset_index": dataset_index,
                "episode": episode,
                "start_step": start_step,
                "source_target_sha256": clean_target_hash,
                "source_boundary_target_rgb_l1_01": source_change["rgb_l1_01"],
                "source_boundary_target_rgb_rmse_01": source_change["rgb_rmse_01"],
                "source_boundary_target_changed_pixel_fraction": source_change[
                    "changed_pixel_fraction"
                ],
                **metrics,
            }
        )
        base_samples.append(
            {
                "base_id": base_id,
                "episode": episode,
                "start_step": start_step,
                "pixels": raw_pixels,
                "actions": raw_actions,
            }
        )

    def work_items():
        for base in base_samples:
            raw_pixels = base["pixels"]
            raw_actions = base["actions"]
            for condition in condition_rows:
                conditioned_pixels = apply_condition_pixels(raw_pixels, condition)
                conditioned_actions = apply_condition_actions(raw_actions, condition)
                target_difference = visual_difference(
                    conditioned_pixels[-1], raw_pixels[-1]
                )
                context_difference = visual_difference(
                    conditioned_pixels[:-1], raw_pixels[:-1]
                )
                action_index = raw_actions.shape[0] - 2
                action_change = (
                    conditioned_actions[action_index] - raw_actions[action_index]
                )
                yield {
                    "base_id": base["base_id"],
                    "episode": base["episode"],
                    "start_step": base["start_step"],
                    "condition": condition,
                    "pixels": preprocess({"pixels": conditioned_pixels})["pixels"],
                    "actions": normalize_action_blocks(
                        conditioned_actions, action_mean, action_std
                    ),
                    "target_rgb_l1_from_clean_01": target_difference["rgb_l1_01"],
                    "target_rgb_rmse_from_clean_01": target_difference["rgb_rmse_01"],
                    "target_changed_pixel_fraction": target_difference[
                        "changed_pixel_fraction"
                    ],
                    "target_changed_pixel_rgb_l1_01": target_difference[
                        "changed_pixel_rgb_l1_01"
                    ],
                    "context_rgb_l1_from_clean_01": context_difference["rgb_l1_01"],
                    "context_changed_pixel_fraction": context_difference[
                        "changed_pixel_fraction"
                    ],
                    "final_action_raw_rms": float(
                        torch.sqrt(
                            torch.nan_to_num(raw_actions[action_index], 0.0)
                            .pow(2)
                            .mean()
                        )
                    ),
                    "final_action_raw_change_rms": float(
                        torch.sqrt(torch.nan_to_num(action_change, 0.0).pow(2).mean())
                    ),
                    "final_action_nan_fraction": float(
                        torch.isnan(raw_actions[action_index]).float().mean()
                    ),
                    "target_sha256": hashlib.sha256(
                        conditioned_pixels[-1].numpy().tobytes()
                    ).hexdigest(),
                }

    scores: list[dict] = []
    predictor_cache: dict[tuple[str, str], torch.Tensor] = {}
    target_cache: dict[tuple[str, str], torch.Tensor] = {}

    def score_batch(batch: list[dict]) -> None:
        pixels = torch.stack([row["pixels"] for row in batch]).to(device)
        actions = torch.stack([row["actions"] for row in batch]).to(device)
        encoded = model.encode({"pixels": pixels, "action": actions})
        predicted = model.predict(encoded["emb"][:, :3], encoded["act_emb"][:, :3])
        temporal = (predicted - encoded["emb"][:, 1:]).pow(2).mean(dim=-1)
        temporal_cosine = 1 - torch.nn.functional.cosine_similarity(
            predicted, encoded["emb"][:, 1:], dim=-1
        )
        for index, row in enumerate(batch):
            condition = row["condition"]
            predictor_cache[(row["base_id"], condition.name)] = predicted[
                index, -1
            ].cpu()
            target_cache[(row["base_id"], condition.name)] = encoded["emb"][
                index, -1
            ].cpu()
            scores.append(
                {
                    "task": args.task,
                    "base_id": row["base_id"],
                    "episode": row["episode"],
                    "start_step": row["start_step"],
                    "condition": condition.name,
                    "family": condition.family,
                    "scope": condition.scope,
                    "value": condition.value,
                    "action_mode": condition.action_mode,
                    "prediction_mse": float(temporal[index, -1]),
                    "prediction_cosine_distance": float(temporal_cosine[index, -1]),
                    "temporal_prediction_mse": json.dumps(
                        [float(value) for value in temporal[index].cpu()]
                    ),
                    "target_rgb_l1_from_clean_01": row["target_rgb_l1_from_clean_01"],
                    "target_rgb_rmse_from_clean_01": row[
                        "target_rgb_rmse_from_clean_01"
                    ],
                    "target_changed_pixel_fraction": row[
                        "target_changed_pixel_fraction"
                    ],
                    "target_changed_pixel_rgb_l1_01": row[
                        "target_changed_pixel_rgb_l1_01"
                    ],
                    "context_rgb_l1_from_clean_01": row["context_rgb_l1_from_clean_01"],
                    "context_changed_pixel_fraction": row[
                        "context_changed_pixel_fraction"
                    ],
                    "final_action_raw_rms": row["final_action_raw_rms"],
                    "final_action_raw_change_rms": row["final_action_raw_change_rms"],
                    "final_action_nan_fraction": row["final_action_nan_fraction"],
                    "target_sha256": row["target_sha256"],
                }
            )

    with torch.inference_mode():
        batch: list[dict] = []
        for item in work_items():
            batch.append(item)
            if len(batch) == args.batch_size:
                score_batch(batch)
                batch = []
        if batch:
            score_batch(batch)

    nonfinite_scores = [row for row in scores if not np.isfinite(row["prediction_mse"])]
    if nonfinite_scores:
        raise RuntimeError(f"Found {len(nonfinite_scores)} nonfinite prediction scores")

    maximum_target_difference = 0.0
    maximum_target_only_predictor_difference = 0.0
    for row in manifest:
        base_id = row["base_id"]
        clean_target = target_cache[(base_id, "clean")]
        for condition in condition_rows:
            if condition.family == "action":
                difference = (
                    (target_cache[(base_id, condition.name)] - clean_target).abs().max()
                )
                maximum_target_difference = max(
                    maximum_target_difference, float(difference)
                )
            if condition.scope == "target":
                difference = (
                    (
                        predictor_cache[(base_id, condition.name)]
                        - predictor_cache[(base_id, "clean")]
                    )
                    .abs()
                    .max()
                )
                maximum_target_only_predictor_difference = max(
                    maximum_target_only_predictor_difference, float(difference)
                )
    if maximum_target_difference > INVARIANCE_ATOL:
        raise RuntimeError(
            f"Same-image target embeddings differ: {maximum_target_difference}"
        )
    if maximum_target_only_predictor_difference > INVARIANCE_ATOL:
        raise RuntimeError(
            "Target-only appearance changed predictor output: "
            f"{maximum_target_only_predictor_difference}"
        )

    write_csv(args.output_dir / "scores.csv", scores)
    write_csv(args.output_dir / "selection_manifest.csv", manifest)
    script_path = Path(__file__).resolve()
    provenance = {
        "status": f"complete_{args.phase}",
        "phase": args.phase,
        "task": args.task,
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "samples": args.samples,
        "conditions": [asdict(row) for row in condition_rows],
        "excluded_manifests": [str(path.resolve()) for path in args.exclude_manifest],
        "excluded_episodes": sorted(excluded_episodes),
        "model_repo": TASKS[args.task]["model_repo"],
        "model_revision": TASKS[args.task]["model_revision"],
        "model_snapshot": str(snapshot),
        "weights_sha256": TASKS[args.task]["weights_sha256"],
        "state_dict_keys": state_dict_keys,
        "model_config": json.loads(json.dumps(model_config, default=str)),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "dataset_revision": TASKS[args.task]["dataset_revision"],
        "dataset_blob_sha256": TASKS[args.task]["dataset_blob_sha256"],
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "script": str(script_path.relative_to(Path.cwd())),
        "script_sha256": sha256_file(script_path),
        "project_revision": archive_revision(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "maximum_same_image_target_embedding_difference": maximum_target_difference,
        "maximum_target_only_predictor_difference": maximum_target_only_predictor_difference,
        "invariance_atol": INVARIANCE_ATOL,
        "wall_seconds": time.time() - started,
        "limitations": [
            "Action interventions test action-outcome consistency, not the paper's random-teleport protocol.",
            "Hue and brightness are pixel-space interventions on official observations.",
        ],
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    (args.output_dir / "status.json").write_text(
        json.dumps(
            {
                "status": f"complete_{args.phase}",
                "phase": args.phase,
                "task": args.task,
                "base_episodes": len(manifest),
                "conditions": len(condition_rows),
                "scores": len(scores),
            },
            indent=2,
        )
        + "\n"
    )
    print((args.output_dir / "status.json").read_text())


if __name__ == "__main__":
    main()
