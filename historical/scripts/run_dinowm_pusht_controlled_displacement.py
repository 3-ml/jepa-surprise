                      


from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import functional as tvf

from scripts.run_dinowm_pusht_native_error_audit import (
    add_dinowm_safe_globals,
    load_model,
    sha256_file,
    visual_difference,
)

ACTION_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
HUE_SCALE_MATCHES = {30.0: 0.25, 60.0: 0.75, 120.0: 1.0, 180.0: 1.0}
INVARIANCE_ATOL = 1e-5
DINO_ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
DINO_ACTION_STD = torch.tensor([0.2019, 0.2002])
DINO_PROPRIO_MEAN = torch.tensor([236.6155, 264.5674, -2.93032027, 2.54307914])
DINO_PROPRIO_STD = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-wm-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", default="dinowm-pusht-controlled-displacement-v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-scale", type=float, default=100.0)
    parser.add_argument("--phase", choices=["smoke", "confirmatory"], default="smoke")
    return parser.parse_args()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def scale_token(value: float) -> str:
    return str(value).replace(".", "p")


def angle_difference(a: float, b: float) -> float:
    return float(abs((a - b + np.pi) % (2 * np.pi) - np.pi))


def normalize_action_blocks_for_dinowm(actions: torch.Tensor, action_scale: float) -> torch.Tensor:
    if actions.shape[-1] % DINO_ACTION_MEAN.numel() != 0:
        raise ValueError(f"Action shape {tuple(actions.shape)} incompatible with DINO action stats")
    raw = actions.reshape(*actions.shape[:-1], -1, DINO_ACTION_MEAN.numel()) / action_scale
    normalized = (raw - DINO_ACTION_MEAN) / DINO_ACTION_STD
    return torch.nan_to_num(normalized.reshape(actions.shape).float(), 0.0)


def normalize_proprio_for_dinowm(states: torch.Tensor) -> torch.Tensor:
    if states.shape[-1] < 7:
        raise ValueError(f"Expected PushT state with agent, block, angle, velocity; got {tuple(states.shape)}")
    proprio = states[..., [0, 1, 5, 6]].float()
    return torch.nan_to_num((proprio - DINO_PROPRIO_MEAN) / DINO_PROPRIO_STD, 0.0)


def transform_frames(transform, frames_uint8: torch.Tensor) -> torch.Tensor:
    return transform(frames_uint8.float() / 255.0)


def simulate_target(state: torch.Tensor, actions: torch.Tensor):
    import gymnasium as gym
    import stable_worldmodel                                      

    source = state.numpy().copy()
    initial = source.copy()
    initial[:2] -= source[-2:] * 0.01
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=0, options={"state": initial})
        realized_start = torch.from_numpy(observation["state"]).float()
        for action in actions.numpy():
            observation, *_ = env.step(action)
        final_state = torch.from_numpy(observation["state"]).float()
        frame = torch.from_numpy(env.render()).permute(2, 0, 1).to(torch.uint8)
    finally:
        env.close()
    return realized_start, final_state, frame


def load_hdf5_dataset(path: Path):
    try:
        from stable_worldmodel.data import HDF5Dataset
    except ImportError:
        from stable_worldmodel.data.formats.hdf5 import HDF5Dataset

    return HDF5Dataset(
        path=str(path),
        frameskip=5,
        num_steps=4,
        keys_to_load=["pixels", "action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state", "episode_idx", "step_idx"],
    )


def read_candidates(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n: int = 10_000):
    draws = rng.integers(0, len(values), size=(n, len(values)))
    return tuple(np.quantile(values[draws].mean(axis=1), [0.025, 0.975]))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def move_predictor_masks_to_device(model, device: torch.device) -> None:
    for module in model.predictor.modules():
        bias = getattr(module, "bias", None)
        if isinstance(bias, torch.Tensor):
            module.bias = bias.to(device)


def score_items(model, items: list[dict], device: torch.device, batch_size: int) -> tuple[list[dict], dict, dict]:
    move_predictor_masks_to_device(model, device)
    scores = []
    predictor_cache = {}
    target_cache = {}
    with torch.inference_mode():
        for offset in range(0, len(items), batch_size):
            batch = items[offset : offset + batch_size]
            obs = {
                "visual": torch.stack([row["visual"] for row in batch]).to(device),
                "proprio": torch.stack([row["proprio"] for row in batch]).to(device),
            }
            act = torch.stack([row["actions"] for row in batch]).to(device)
            z = model.encode(obs, act)
            z_src = z[:, : model.num_hist]
            z_tgt = z[:, model.num_pred :]
            z_pred = model.predict(z_src)
            if model.concat_dim == 0:
                visual_pred = z_pred[:, :, :-2, :]
                visual_tgt = z_tgt[:, :, :-2, :]
            else:
                cutoff = -(model.proprio_dim + model.action_dim)
                visual_pred = z_pred[:, :, :, :cutoff]
                visual_tgt = z_tgt[:, :, :, :cutoff]
            visual_mse = (visual_pred - visual_tgt).pow(2).flatten(2).mean(dim=2)
            for index, row in enumerate(batch):
                predictor_cache[(row["base_id"], row["condition"])] = visual_pred[index, -1].cpu()
                target_cache[(row["base_id"], row["condition"])] = visual_tgt[index, -1].cpu()
                scores.append(
                    {
                        "task": "pusht",
                        "model": "DINO-WM",
                        "base_id": row["base_id"],
                        "episode": row["episode"],
                        "start_step": row["start_step"],
                        "condition": row["condition"],
                        "target_group": row["target_group"],
                        "action_mode": row["action_mode"],
                        "action_scale": row["action_scale"],
                        "intended_displacement_scale": row["intended_displacement_scale"],
                        "agent_displacement_state": row["agent_displacement_state"],
                        "block_displacement_state": row["block_displacement_state"],
                        "block_angle_change_radians": row["block_angle_change_radians"],
                        "prediction_mse": float(visual_mse[index, -1].cpu()),
                        "temporal_prediction_mse": json.dumps([float(x) for x in visual_mse[index].cpu()]),
                        "target_rgb_l1_from_static_01": row["target_rgb_l1_from_static_01"],
                        "target_changed_pixel_fraction_from_static": row["target_changed_pixel_fraction_from_static"],
                        "target_sha256": row["target_sha256"],
                    }
                )
    return scores, predictor_cache, target_cache


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("samples and batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = torch.device(args.device)

    add_dinowm_safe_globals(args.dino_wm_root)
    model, cfg, ckpt_path, epoch = load_model(args, device)
    dataset = load_hdf5_dataset(args.dataset)
    from datasets.img_transforms import default_transform
    transform = default_transform(int(cfg.img_size))

    candidates = read_candidates(args.selection_manifest)
    candidates = candidates[: args.samples]
    if len(candidates) != args.samples:
        raise RuntimeError(f"Only {len(candidates)} candidates for {args.samples} requested")

    selected = []
    rejected = []
    scores = []
    maximum_same_target_embedding_difference = 0.0
    maximum_start_state_error = 0.0
    maximum_official_endpoint_error = 0.0

    for candidate in candidates:
        dataset_index = int(candidate["dataset_index"])
        sample = dataset[dataset_index]
        raw_actions = sample["action"].float()
        action_block = raw_actions[2].reshape(5, 2)
        if not torch.isfinite(action_block).all():
            rejected.append({**candidate, "reason": "nonfinite_final_action_block"})
            continue
        source_state = sample["state"][2].float()
        official_state = sample["state"][3].float()
        rendered = {}
        states = {}
        start_errors = []
        for scale in ACTION_SCALES:
            realized_start, final_state, frame = simulate_target(source_state, action_block * scale)
            start_errors.append(float((realized_start - source_state).abs().max()))
            states[scale] = final_state
            rendered[scale] = frame
        start_error = max(start_errors)
        endpoint_error = float((states[1.0] - official_state).abs().max())
        maximum_start_state_error = max(maximum_start_state_error, start_error)
        maximum_official_endpoint_error = max(maximum_official_endpoint_error, endpoint_error)
        if start_error > 2e-5 or endpoint_error > 2e-4:
            rejected.append({**candidate, "reason": f"dynamics_identity:start={start_error}:endpoint={endpoint_error}"})
            continue

        episode = int(sample["episode_idx"][0])
        start_step = int(sample["step_idx"][0])
        base_id = candidate.get("base_id") or f"pusht:{episode}:{start_step}"
        stored = sample["pixels"].to(torch.uint8)
        stored_states = sample["state"].float()
        static_frame = rendered[0.0]
        natural_agent = float(torch.linalg.vector_norm(official_state[:2] - source_state[:2]))
        natural_block = float(torch.linalg.vector_norm(official_state[2:4] - source_state[2:4]))
        natural_angle = angle_difference(float(official_state[4]), float(source_state[4]))
        natural_identity = visual_difference(rendered[1.0], stored[-1])
        boundary_identity = visual_difference(static_frame, stored[2])
        selected.append(
            {
                "base_id": base_id,
                "dataset_index": dataset_index,
                "episode": episode,
                "start_step": start_step,
                "start_state_max_abs_error": start_error,
                "official_endpoint_max_abs_error": endpoint_error,
                "natural_target_renderer_rgb_l1_01": natural_identity["rgb_l1_01"],
                "boundary_renderer_rgb_l1_01": boundary_identity["rgb_l1_01"],
                "natural_agent_displacement_state": natural_agent,
                "natural_block_displacement_state": natural_block,
                "natural_block_angle_change_radians": natural_angle,
                "natural_block_moving": int(natural_block >= 1.0 or natural_angle >= 0.02),
            }
        )

        local_items = []

        def add_item(condition, target_group, action_mode, action_scale, intended_scale, target_frame, target_state, final_block):
            pixels = torch.cat([stored[:3], target_frame.unsqueeze(0)], dim=0)
            states_for_proprio = stored_states.clone()
            states_for_proprio[3] = target_state
            raw_for_model = raw_actions.clone()
            raw_for_model[2] = final_block.reshape(-1)
            difference = visual_difference(target_frame, static_frame)
            agent_displacement = float(torch.linalg.vector_norm(target_state[:2] - states[0.0][:2]))
            block_displacement = float(torch.linalg.vector_norm(target_state[2:4] - states[0.0][2:4]))
            block_angle = angle_difference(float(target_state[4]), float(states[0.0][4]))
            local_items.append(
                {
                    "base_id": base_id,
                    "episode": episode,
                    "start_step": start_step,
                    "condition": condition,
                    "target_group": target_group,
                    "action_mode": action_mode,
                    "action_scale": action_scale,
                    "intended_displacement_scale": intended_scale,
                    "agent_displacement_state": agent_displacement,
                    "block_displacement_state": block_displacement,
                    "block_angle_change_radians": block_angle,
                    "visual": transform_frames(transform, pixels),
                    "proprio": normalize_proprio_for_dinowm(states_for_proprio),
                    "actions": normalize_action_blocks_for_dinowm(raw_for_model, args.action_scale),
                    "target_rgb_l1_from_static_01": difference["rgb_l1_01"],
                    "target_changed_pixel_fraction_from_static": difference["changed_pixel_fraction"],
                    "target_sha256": hashlib.sha256(target_frame.numpy().tobytes()).hexdigest(),
                }
            )

        add_item("official_clean", "official_stored", "official", 1.0, 1.0, stored[-1], official_state, action_block)
        for scale in ACTION_SCALES:
            token = scale_token(scale)
            target_frame = rendered[scale]
            target_state = states[scale]
            add_item(f"lawful_scale_{token}", f"scale_{token}", "lawful", scale, scale, target_frame, target_state, action_block * scale)
            if scale > 0:
                add_item(f"zero_scale_{token}", f"scale_{token}", "zero", 0.0, scale, target_frame, target_state, torch.zeros_like(action_block))
                add_item(f"opposite_scale_{token}", f"scale_{token}", "opposite", -scale, scale, target_frame, target_state, -action_block * scale)
        for hue_degrees, matched_scale in HUE_SCALE_MATCHES.items():
            hue_frame = tvf.adjust_hue(static_frame, hue_degrees / 360.0)
            add_item(
                f"appearance_hue_{int(hue_degrees):03d}",
                f"appearance_hue_{int(hue_degrees):03d}",
                "appearance",
                0.0,
                0.0,
                hue_frame,
                states[0.0],
                torch.zeros_like(action_block),
            )

        local_scores, _local_predictors, local_targets = score_items(model, local_items, device, args.batch_size)
        if not all(np.isfinite(row["prediction_mse"]) for row in local_scores):
            raise RuntimeError(f"Nonfinite score for {base_id}")
        scores.extend(local_scores)
        for scale in ACTION_SCALES:
            token = scale_token(scale)
            names = [f"lawful_scale_{token}"]
            if scale > 0:
                names += [f"zero_scale_{token}", f"opposite_scale_{token}"]
            reference = local_targets[(base_id, names[0])]
            for name in names[1:]:
                maximum_same_target_embedding_difference = max(
                    maximum_same_target_embedding_difference,
                    float((local_targets[(base_id, name)] - reference).abs().max()),
                )
        if len(selected) % 25 == 0 or len(selected) == args.samples:
            print(f"processed {len(selected)}/{args.samples} clips", flush=True)

    if len(selected) != args.samples:
        raise RuntimeError(f"Only {len(selected)} valid clips for {args.samples} requested; rejected {len(rejected)}")

    if maximum_same_target_embedding_difference > INVARIANCE_ATOL:
        raise RuntimeError(f"Same-target visual embedding invariance failed: {maximum_same_target_embedding_difference}")

    write_csv(args.output_dir / "scores.csv", scores)
    write_csv(args.output_dir / "selection_manifest.csv", selected)
    if rejected:
        write_csv(args.output_dir / "rejected_candidates.csv", rejected)

    frame = pd.DataFrame(scores)
    pivot = frame.pivot(index="base_id", columns="condition", values="prediction_mse")
    rng = np.random.default_rng(20260902)
    summary = []
    for condition in frame.condition.drop_duplicates():
        group = frame[frame.condition == condition]
        values = pivot[condition].to_numpy()
        low, high = bootstrap_mean(values, rng)
        first = group.iloc[0]
        summary.append(
            {
                "condition": condition,
                "action_mode": first.action_mode,
                "action_scale": first.action_scale,
                "intended_displacement_scale": first.intended_displacement_scale,
                "n_episodes": len(values),
                "score_mean": values.mean(),
                "score_ci95_low": low,
                "score_ci95_high": high,
                "score_median": np.median(values),
                "agent_displacement_state_mean": group.agent_displacement_state.mean(),
                "block_displacement_state_mean": group.block_displacement_state.mean(),
                "block_angle_change_radians_mean": group.block_angle_change_radians.mean(),
                "target_rgb_l1_from_static_01_mean": group.target_rgb_l1_from_static_01.mean(),
            }
        )
    write_csv(args.output_dir / "condition_summary.csv", summary)

    contrasts = []
    moving = pd.DataFrame(selected).set_index("base_id").natural_block_moving
    for scale in ACTION_SCALES[1:]:
        token = scale_token(scale)
        lawful = pivot[f"lawful_scale_{token}"]
        for mode in ("zero", "opposite"):
            delta = pivot[f"{mode}_scale_{token}"] - lawful
            low, high = bootstrap_mean(delta.to_numpy(), rng)
            row = {
                "action_mode": mode,
                "intended_displacement_scale": scale,
                "n_episodes": len(delta),
                "inconsistent_minus_lawful_mean": delta.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "median": delta.median(),
                "positive": int((delta > 0).sum()),
            }
            for group_name, mask in [("block_static", moving == 0), ("block_moving", moving == 1)]:
                values = delta.loc[mask.index[mask]].to_numpy()
                row[f"{group_name}_n"] = len(values)
                row[f"{group_name}_mean"] = values.mean() if len(values) else np.nan
            contrasts.append(row)
    write_csv(args.output_dir / "same_target_contrasts.csv", contrasts)

    construct_rows = []
    static = pivot["lawful_scale_0p0"]
    for hue_degrees, scale in HUE_SCALE_MATCHES.items():
        hue_condition = f"appearance_hue_{int(hue_degrees):03d}"
        appearance_delta = (pivot[hue_condition] - static).to_numpy()
        appearance_low, appearance_high = bootstrap_mean(appearance_delta, rng)
        row = {
            "hue_degrees": hue_degrees,
            "matched_displacement_scale": scale,
            "appearance_minus_static_mean": appearance_delta.mean(),
            "appearance_ci95_low": appearance_low,
            "appearance_ci95_high": appearance_high,
        }
        for mode in ("zero", "opposite"):
            condition = f"{mode}_scale_{scale_token(scale)}"
            difference = (pivot[hue_condition] - pivot[condition]).to_numpy()
            low, high = bootstrap_mean(difference, rng)
            row[f"appearance_minus_{mode}_mean"] = difference.mean()
            row[f"appearance_minus_{mode}_ci95_low"] = low
            row[f"appearance_minus_{mode}_ci95_high"] = high
        hue_rows = frame[frame.condition == hue_condition]
        physical_rows = frame[frame.condition == f"lawful_scale_{scale_token(scale)}"]
        row["appearance_rgb_l1_mean"] = hue_rows.target_rgb_l1_from_static_01.mean()
        row["physical_rgb_l1_mean"] = physical_rows.target_rgb_l1_from_static_01.mean()
        construct_rows.append(row)
    write_csv(args.output_dir / "construct_comparisons.csv", construct_rows)

    summary_frame = pd.DataFrame(summary)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for mode, marker in [("lawful", "o"), ("zero", "s"), ("opposite", "^")]:
        group = summary_frame[summary_frame.action_mode == mode].sort_values("intended_displacement_scale")
        ax.plot(group.intended_displacement_scale, group.score_mean, marker=marker, label=mode)
    ax.set_xlabel("Official action-block scale")
    ax.set_ylabel("DINO-WM visual-latent MSE")
    ax.set_title("PushT: identical target, DINO-WM scorer")
    ax.legend()
    fig.savefig(args.output_dir / "controlled_displacement.png", dpi=180)
    plt.close(fig)

    provenance = {
        "status": f"complete_{args.phase}",
        "phase": args.phase,
        "task": "pusht",
        "model": "DINO-WM",
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "samples": args.samples,
        "action_scales": ACTION_SCALES,
        "hue_scale_matches": HUE_SCALE_MATCHES,
        "action_scale_before_dinowm_normalization": args.action_scale,
        "reset_dt_compensation": 0.01,
        "dino_wm_repo": str(args.dino_wm_root.resolve()),
        "dino_wm_revision": git_revision(args.dino_wm_root),
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_epoch": epoch,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "maximum_start_state_error": maximum_start_state_error,
        "maximum_official_endpoint_error": maximum_official_endpoint_error,
        "maximum_same_target_visual_embedding_difference": maximum_same_target_embedding_difference,
        "script_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "wall_seconds": time.time() - started,
        "score": "final-step DINO-WM visual latent MSE against the encoded target",
        "claim_boundary": "Same PushT controlled targets and episode units as the LeWorldModel Table 5 PushT rows; scorer is the released DINO-WM PushT visual-latent predictor.",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (args.output_dir / "README.md").write_text(
        "# DINO-WM PushT controlled-displacement native-error audit\n\n"
        "This run scores the same PushT controlled-displacement construction used for the LeWorldModel Table 5 PushT rows with the released DINO-WM PushT checkpoint. "
        "Targets are simulator-lawful frames generated from the LeWM PushT HDF5 states/actions, then reused under lawful, zero, and opposite final action blocks; appearance controls hue-rotate the static target.\n"
    )


if __name__ == "__main__":
    main()