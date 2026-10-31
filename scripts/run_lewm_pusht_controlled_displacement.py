#!/usr/bin/env python3
"""Controlled lawful-versus-action-inconsistent PushT LeWM audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.transforms.v2 import functional as tvf

from scripts.run_lewm_prediction_error_audit import (
    INVARIANCE_ATOL,
    load_model,
    make_dataset,
    normalize_action_blocks,
    sha256_file,
    visual_difference,
    write_csv,
)

ACTION_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
HUE_SCALE_MATCHES = {30.0: 0.25, 60.0: 0.75, 120.0: 1.0, 180.0: 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", default="pusht-controlled-displacement-v1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--phase", choices=["exploratory", "confirmatory"], default="exploratory"
    )
    return parser.parse_args()


def candidates(path: Path, seed: str) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    material = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    rng = np.random.default_rng(material)
    return [rows[index] for index in rng.permutation(len(rows))]


def scale_token(value: float) -> str:
    return str(value).replace(".", "p")


def angle_difference(a: float, b: float) -> float:
    return float(abs((a - b + np.pi) % (2 * np.pi) - np.pi))


def simulate_target(state: torch.Tensor, actions: torch.Tensor):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401 - registers environment

    source = state.numpy().copy()
    # PushT _set_state advances one dt during reset; compensate agent position.
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


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n=10_000):
    draws = rng.integers(0, len(values), size=(n, len(values)))
    return tuple(np.quantile(values[draws].mean(axis=1), [0.025, 0.975]))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = torch.device(args.device)
    model, snapshot, _config, state_dict_keys = load_model("pusht", device)
    dataset, action_mean, action_std, preprocess = make_dataset("pusht", args.dataset)
    pool = candidates(args.selection_manifest, args.seed)

    selected = []
    rejected = []
    scores = []
    targets = {}
    maximum_start_state_error = 0.0
    maximum_official_endpoint_error = 0.0

    def score_items(items: list[dict]) -> None:
        with torch.inference_mode():
            for offset in range(0, len(items), args.batch_size):
                batch = items[offset : offset + args.batch_size]
                pixels = torch.stack([row["pixels"] for row in batch]).to(device)
                actions = torch.stack([row["actions"] for row in batch]).to(device)
                encoded = model.encode({"pixels": pixels, "action": actions})
                predicted = model.predict(
                    encoded["emb"][:, :3], encoded["act_emb"][:, :3]
                )
                mse = (predicted - encoded["emb"][:, 1:]).pow(2).mean(dim=-1)
                for index, row in enumerate(batch):
                    targets[(row["base_id"], row["target_group"], row["condition"])] = encoded[
                        "emb"
                    ][index, -1].cpu()
                    scores.append(
                        {
                            "task": "pusht",
                            "base_id": row["base_id"],
                            "episode": row["episode"],
                            "start_step": row["start_step"],
                            "condition": row["condition"],
                            "target_group": row["target_group"],
                            "action_mode": row["action_mode"],
                            "action_scale": row["action_scale"],
                            "intended_displacement_scale": row[
                                "intended_displacement_scale"
                            ],
                            "agent_displacement_state": row[
                                "agent_displacement_state"
                            ],
                            "block_displacement_state": row[
                                "block_displacement_state"
                            ],
                            "block_angle_change_radians": row[
                                "block_angle_change_radians"
                            ],
                            "prediction_mse": float(mse[index, -1]),
                            "temporal_prediction_mse": json.dumps(
                                [float(value) for value in mse[index].cpu()]
                            ),
                            "target_rgb_l1_from_static_01": row[
                                "target_rgb_l1_from_static_01"
                            ],
                            "target_changed_pixel_fraction_from_static": row[
                                "target_changed_pixel_fraction_from_static"
                            ],
                            "target_sha256": row["target_sha256"],
                        }
                    )

    for candidate in pool:
        if len(selected) == args.samples:
            break
        dataset_index = int(candidate["dataset_index"])
        sample = dataset[dataset_index]
        action_block = sample["action"][2].float().reshape(5, 2)
        if not torch.isfinite(action_block).all():
            rejected.append({**candidate, "reason": "nonfinite_final_action_block"})
            continue
        source_state = sample["state"][2].float()
        official_state = sample["state"][3].float()
        rendered = {}
        states = {}
        start_errors = []
        for scale in ACTION_SCALES:
            realized_start, final_state, frame = simulate_target(
                source_state, action_block * scale
            )
            start_errors.append(float((realized_start - source_state).abs().max()))
            states[scale] = final_state
            rendered[scale] = frame
        start_error = max(start_errors)
        endpoint_error = float((states[1.0] - official_state).abs().max())
        maximum_start_state_error = max(maximum_start_state_error, start_error)
        maximum_official_endpoint_error = max(
            maximum_official_endpoint_error, endpoint_error
        )
        if start_error > 2e-5 or endpoint_error > 2e-4:
            rejected.append(
                {
                    **candidate,
                    "reason": f"dynamics_identity:start={start_error}:endpoint={endpoint_error}",
                }
            )
            continue

        static_frame = rendered[0.0]
        natural_identity = visual_difference(
            rendered[1.0], sample["pixels"][-1].to(torch.uint8)
        )
        boundary_identity = visual_difference(
            static_frame, sample["pixels"][2].to(torch.uint8)
        )
        natural_agent = float(
            torch.linalg.vector_norm(official_state[:2] - source_state[:2])
        )
        natural_block = float(
            torch.linalg.vector_norm(official_state[2:4] - source_state[2:4])
        )
        natural_angle = angle_difference(float(official_state[4]), float(source_state[4]))
        selected.append(
            {
                "base_id": candidate["base_id"],
                "dataset_index": dataset_index,
                "episode": int(candidate["episode"]),
                "start_step": int(candidate["start_step"]),
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

        base_id = candidate["base_id"]
        stored = sample["pixels"].to(torch.uint8)
        base_work = []
        base_work.append(
            {
                "base_id": base_id,
                "episode": int(candidate["episode"]),
                "start_step": int(candidate["start_step"]),
                "condition": "official_clean",
                "target_group": "official_stored",
                "action_mode": "official",
                "action_scale": 1.0,
                "intended_displacement_scale": 1.0,
                "agent_displacement_state": natural_agent,
                "block_displacement_state": natural_block,
                "block_angle_change_radians": natural_angle,
                "pixels": preprocess({"pixels": stored})["pixels"],
                "actions": normalize_action_blocks(
                    sample["action"].float(), action_mean, action_std
                ),
                "target_rgb_l1_from_static_01": visual_difference(
                    stored[-1], static_frame
                )["rgb_l1_01"],
                "target_changed_pixel_fraction_from_static": visual_difference(
                    stored[-1], static_frame
                )["changed_pixel_fraction"],
                "target_sha256": hashlib.sha256(stored[-1].numpy().tobytes()).hexdigest(),
            }
        )
        for scale in ACTION_SCALES:
            token = scale_token(scale)
            target_frame = rendered[scale]
            pixels = torch.cat([stored[:3], target_frame.unsqueeze(0)], dim=0)
            difference = visual_difference(target_frame, static_frame)
            agent_displacement = float(
                torch.linalg.vector_norm(states[scale][:2] - states[0.0][:2])
            )
            block_displacement = float(
                torch.linalg.vector_norm(states[scale][2:4] - states[0.0][2:4])
            )
            block_angle = angle_difference(float(states[scale][4]), float(states[0.0][4]))
            modes = [("lawful", action_block * scale)]
            if scale > 0:
                modes.extend(
                    [("zero", torch.zeros_like(action_block)), ("opposite", -action_block * scale)]
                )
            for mode, final_block in modes:
                raw_actions = sample["action"].float().clone()
                raw_actions[2] = final_block.reshape(-1)
                condition = f"{mode}_scale_{token}"
                base_work.append(
                    {
                        "base_id": base_id,
                        "episode": int(candidate["episode"]),
                        "start_step": int(candidate["start_step"]),
                        "condition": condition,
                        "target_group": f"scale_{token}",
                        "action_mode": mode,
                        "action_scale": scale if mode == "lawful" else (0.0 if mode == "zero" else -scale),
                        "intended_displacement_scale": scale,
                        "agent_displacement_state": agent_displacement,
                        "block_displacement_state": block_displacement,
                        "block_angle_change_radians": block_angle,
                        "pixels": preprocess({"pixels": pixels})["pixels"],
                        "actions": normalize_action_blocks(
                            raw_actions, action_mean, action_std
                        ),
                        "target_rgb_l1_from_static_01": difference["rgb_l1_01"],
                        "target_changed_pixel_fraction_from_static": difference[
                            "changed_pixel_fraction"
                        ],
                        "target_sha256": hashlib.sha256(
                            target_frame.numpy().tobytes()
                        ).hexdigest(),
                    }
                )

        for hue_degrees in HUE_SCALE_MATCHES:
            hue_frame = tvf.adjust_hue(static_frame, hue_degrees / 360.0)
            hue_pixels = torch.cat([stored[:3], hue_frame.unsqueeze(0)], dim=0)
            raw_actions = sample["action"].float().clone()
            raw_actions[2] = 0
            hue_difference = visual_difference(hue_frame, static_frame)
            base_work.append(
                {
                    "base_id": base_id,
                    "episode": int(candidate["episode"]),
                    "start_step": int(candidate["start_step"]),
                    "condition": f"appearance_hue_{int(hue_degrees):03d}",
                    "target_group": f"appearance_hue_{int(hue_degrees):03d}",
                    "action_mode": "appearance",
                    "action_scale": 0.0,
                    "intended_displacement_scale": 0.0,
                    "agent_displacement_state": 0.0,
                    "block_displacement_state": 0.0,
                    "block_angle_change_radians": 0.0,
                    "pixels": preprocess({"pixels": hue_pixels})["pixels"],
                    "actions": normalize_action_blocks(
                        raw_actions, action_mean, action_std
                    ),
                    "target_rgb_l1_from_static_01": hue_difference["rgb_l1_01"],
                    "target_changed_pixel_fraction_from_static": hue_difference[
                        "changed_pixel_fraction"
                    ],
                    "target_sha256": hashlib.sha256(
                        hue_frame.numpy().tobytes()
                    ).hexdigest(),
                }
            )
        score_items(base_work)

    if len(selected) != args.samples:
        raise RuntimeError(f"Only {len(selected)} valid clips for {args.samples} requested")
    if not all(np.isfinite(row["prediction_mse"]) for row in scores):
        raise RuntimeError("Nonfinite score")

    maximum_same_target_embedding_difference = 0.0
    for base in selected:
        base_id = base["base_id"]
        for scale in ACTION_SCALES:
            token = scale_token(scale)
            names = [f"lawful_scale_{token}"]
            if scale > 0:
                names += [f"zero_scale_{token}", f"opposite_scale_{token}"]
            reference = targets[(base_id, f"scale_{token}", names[0])]
            for name in names[1:]:
                difference = float(
                    (targets[(base_id, f"scale_{token}", name)] - reference).abs().max()
                )
                maximum_same_target_embedding_difference = max(
                    maximum_same_target_embedding_difference, difference
                )
    if maximum_same_target_embedding_difference > INVARIANCE_ATOL:
        raise RuntimeError("Same-target embedding invariance failed")

    write_csv(args.output_dir / "scores.csv", scores)
    write_csv(args.output_dir / "selection_manifest.csv", selected)
    if rejected:
        write_csv(args.output_dir / "rejected_candidates.csv", rejected)

    frame = pd.DataFrame(scores)
    pivot = frame.pivot(index="base_id", columns="condition", values="prediction_mse")
    rng = np.random.default_rng(20260823)
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
            for group_name, mask in [
                ("block_static", moving == 0),
                ("block_moving", moving == 1),
            ]:
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
        physical_rows = frame[
            frame.condition == f"lawful_scale_{scale_token(scale)}"
        ]
        row["appearance_rgb_l1_mean"] = hue_rows.target_rgb_l1_from_static_01.mean()
        row["physical_rgb_l1_mean"] = physical_rows.target_rgb_l1_from_static_01.mean()
        construct_rows.append(row)
    write_csv(args.output_dir / "construct_comparisons.csv", construct_rows)

    summary_frame = pd.DataFrame(summary)
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for mode, marker in [("lawful", "o"), ("zero", "s"), ("opposite", "^")]:
        group = summary_frame[summary_frame.action_mode == mode].sort_values(
            "intended_displacement_scale"
        )
        axis.plot(
            group.intended_displacement_scale,
            group.score_mean,
            marker=marker,
            label=mode,
        )
    axis.set_xlabel("Official action-block scale")
    axis.set_ylabel("LeWM prediction MSE")
    axis.set_title("PushT: identical target, different final action block")
    axis.legend()
    figure.savefig(args.output_dir / "controlled_displacement.png", dpi=180)
    plt.close(figure)

    provenance = {
        "status": f"complete_{args.phase}",
        "phase": args.phase,
        "task": "pusht",
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "samples": args.samples,
        "action_scales": ACTION_SCALES,
        "hue_scale_matches": HUE_SCALE_MATCHES,
        "reset_dt_compensation": 0.01,
        "model_snapshot": str(snapshot),
        "weights_sha256": "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607",
        "state_dict_keys": state_dict_keys,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "source_selection_manifest": str(args.selection_manifest.resolve()),
        "source_selection_manifest_sha256": sha256_file(args.selection_manifest),
        "maximum_start_state_error": maximum_start_state_error,
        "maximum_official_endpoint_error": maximum_official_endpoint_error,
        "maximum_same_target_embedding_difference": maximum_same_target_embedding_difference,
        "script_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "wall_seconds": time.time() - started,
        "limitations": [
            "Same-target counterfactuals test action-conditioned consistency, not the unreleased paper teleport intervention.",
            "Current PushT rendering is not byte-identical to stored observations; all controlled targets use the paired current renderer.",
            "Block-moving strata are defined from realized displacement, not contact metadata.",
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
                "episodes": args.samples,
                "conditions": int(frame.condition.nunique()),
                "scores": len(scores),
                "rejected_candidates": len(rejected),
            },
            indent=2,
        )
        + "\n"
    )
    print((args.output_dir / "status.json").read_text())


if __name__ == "__main__":
    main()
