from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

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
HUE_SCALE_MATCHES = {30.0: 0.25, 60.0: 0.5, 120.0: 0.75, 180.0: 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", default="tworoom-controlled-displacement-v1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--phase", choices=["exploratory", "confirmatory"], default="exploratory"
    )
    return parser.parse_args()


def candidate_manifest(path: Path, seed: str) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    material = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    rng = np.random.default_rng(material)
    return [rows[index] for index in rng.permutation(len(rows))]


def simulate_target(position, actions, target_position):
    import gymnasium as gym
    import stable_worldmodel

    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    try:
        observation, _ = env.reset(
            seed=0,
            options={"state": position.numpy(), "target_state": target_position},
        )
        for action in actions.numpy():
            observation, *_ = env.step(action)
        frame = torch.from_numpy(env.render()).permute(2, 0, 1).to(torch.uint8)
        final_position = torch.as_tensor(observation[:2]).float()
    finally:
        env.close()
    return final_position, frame


def scale_token(value: float) -> str:
    return str(value).replace(".", "p")


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n=10_000):
    draws = rng.integers(0, len(values), size=(n, len(values)))
    return tuple(np.quantile(values[draws].mean(axis=1), [0.025, 0.975]))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = torch.device(args.device)
    model, snapshot, _config, state_dict_keys = load_model("tworoom", device)
    dataset, action_mean, action_std, preprocess = make_dataset("tworoom", args.dataset)
    candidates = candidate_manifest(args.selection_manifest, args.seed)

    scores = []
    targets = {}
    selected_rows = []
    rejected_rows = []
    maximum_official_dynamics_position_error = 0.0

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
                    targets[(row["base_id"], row["target_group"], row["condition"])] = (
                        encoded["emb"][index, -1].cpu()
                    )
                    scores.append(
                        {
                            "task": "tworoom",
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
                            "actual_displacement_pixels": row[
                                "actual_displacement_pixels"
                            ],
                            "actual_displacement_diameters": row[
                                "actual_displacement_pixels"
                            ]
                            / 14.0,
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

    target_position = np.asarray([164.0, 112.0], dtype=np.float32)
    for candidate in candidates:
        if len(selected_rows) == args.samples:
            break
        dataset_index = int(candidate["dataset_index"])
        sample = dataset[dataset_index]
        raw_action_block = sample["action"][2].float().reshape(5, 2)
        if not torch.isfinite(raw_action_block).all():
            rejected_rows.append(
                {**candidate, "reason": "nonfinite_final_action_block"}
            )
            continue
        base_id = candidate["base_id"]
        start_position = sample["proprio"][2].float()
        official_target_position = sample["proprio"][3].float()
        static_position, static_frame = simulate_target(
            start_position, torch.zeros_like(raw_action_block), target_position
        )
        if not torch.allclose(static_position, start_position, atol=1e-6, rtol=0):
            raise RuntimeError("Zero action changed TwoRoom position")

        rendered_targets = {}
        target_positions = {}
        for action_scale in ACTION_SCALES:
            final_position, frame = simulate_target(
                start_position, raw_action_block * action_scale, target_position
            )
            rendered_targets[action_scale] = frame
            target_positions[action_scale] = final_position
        official_error = float(
            (target_positions[1.0] - official_target_position).abs().max()
        )
        maximum_official_dynamics_position_error = max(
            maximum_official_dynamics_position_error, official_error
        )
        if official_error > 1e-5:
            rejected_rows.append(
                {**candidate, "reason": f"official_dynamics_mismatch:{official_error}"}
            )
            continue

        identity_target = visual_difference(
            rendered_targets[1.0], sample["pixels"][-1].to(torch.uint8)
        )
        identity_boundary = visual_difference(
            static_frame, sample["pixels"][2].to(torch.uint8)
        )
        selected_rows.append(
            {
                "base_id": base_id,
                "dataset_index": dataset_index,
                "episode": int(candidate["episode"]),
                "start_step": int(candidate["start_step"]),
                "official_dynamics_position_max_abs_error": official_error,
                "natural_target_renderer_rgb_l1_01": identity_target["rgb_l1_01"],
                "boundary_renderer_rgb_l1_01": identity_boundary["rgb_l1_01"],
                "natural_displacement_pixels": float(
                    torch.linalg.vector_norm(official_target_position - start_position)
                ),
            }
        )

        base_work = []
        official_pixels = sample["pixels"].to(torch.uint8)
        official_actions = normalize_action_blocks(
            sample["action"].float(), action_mean, action_std
        )
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
                "actual_displacement_pixels": float(
                    torch.linalg.vector_norm(official_target_position - start_position)
                ),
                "pixels": preprocess({"pixels": official_pixels})["pixels"],
                "actions": official_actions,
                "target_rgb_l1_from_static_01": visual_difference(
                    official_pixels[-1], static_frame
                )["rgb_l1_01"],
                "target_changed_pixel_fraction_from_static": visual_difference(
                    official_pixels[-1], static_frame
                )["changed_pixel_fraction"],
                "target_sha256": hashlib.sha256(
                    official_pixels[-1].numpy().tobytes()
                ).hexdigest(),
            }
        )

        for action_scale in ACTION_SCALES:
            target_frame = rendered_targets[action_scale]
            target_group = f"scale_{scale_token(action_scale)}"
            target_pixels = torch.cat(
                [official_pixels[:3], target_frame.unsqueeze(0)], dim=0
            )
            target_difference = visual_difference(target_frame, static_frame)
            target_hash = hashlib.sha256(target_frame.numpy().tobytes()).hexdigest()
            actual_displacement = float(
                torch.linalg.vector_norm(
                    target_positions[action_scale] - start_position
                )
            )
            modes = [("lawful", raw_action_block * action_scale)]
            if action_scale > 0:
                modes.extend(
                    [
                        ("zero", torch.zeros_like(raw_action_block)),
                        ("opposite", -raw_action_block * action_scale),
                    ]
                )
            for mode, final_block in modes:
                raw_actions = sample["action"].float().clone()
                raw_actions[2] = final_block.reshape(-1)
                normalized_actions = normalize_action_blocks(
                    raw_actions, action_mean, action_std
                )
                condition = f"{mode}_scale_{scale_token(action_scale)}"
                base_work.append(
                    {
                        "base_id": base_id,
                        "episode": int(candidate["episode"]),
                        "start_step": int(candidate["start_step"]),
                        "condition": condition,
                        "target_group": target_group,
                        "action_mode": mode,
                        "action_scale": action_scale
                        if mode == "lawful"
                        else (0.0 if mode == "zero" else -action_scale),
                        "intended_displacement_scale": action_scale,
                        "actual_displacement_pixels": actual_displacement,
                        "pixels": preprocess({"pixels": target_pixels})["pixels"],
                        "actions": normalized_actions,
                        "target_rgb_l1_from_static_01": target_difference["rgb_l1_01"],
                        "target_changed_pixel_fraction_from_static": target_difference[
                            "changed_pixel_fraction"
                        ],
                        "target_sha256": target_hash,
                    }
                )

        for hue_degrees in HUE_SCALE_MATCHES:
            hue_frame = tvf.adjust_hue(static_frame, hue_degrees / 360.0)
            hue_pixels = torch.cat([official_pixels[:3], hue_frame.unsqueeze(0)], dim=0)
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
                    "actual_displacement_pixels": 0.0,
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

    if len(selected_rows) != args.samples:
        raise RuntimeError(
            f"Only {len(selected_rows)} valid clips found for {args.samples} requested"
        )
    if not all(np.isfinite(row["prediction_mse"]) for row in scores):
        raise RuntimeError("Nonfinite score")

    maximum_same_target_embedding_difference = 0.0
    for base in selected_rows:
        base_id = base["base_id"]
        for scale in ACTION_SCALES:
            group = f"scale_{scale_token(scale)}"
            condition_names = [f"lawful_scale_{scale_token(scale)}"]
            if scale > 0:
                condition_names.extend(
                    [
                        f"zero_scale_{scale_token(scale)}",
                        f"opposite_scale_{scale_token(scale)}",
                    ]
                )
            reference = targets[(base_id, group, condition_names[0])]
            for condition in condition_names[1:]:
                difference = float(
                    (targets[(base_id, group, condition)] - reference).abs().max()
                )
                maximum_same_target_embedding_difference = max(
                    maximum_same_target_embedding_difference, difference
                )
    if maximum_same_target_embedding_difference > INVARIANCE_ATOL:
        raise RuntimeError(
            f"Same-target embedding invariance failed: {maximum_same_target_embedding_difference}"
        )

    write_csv(args.output_dir / "scores.csv", scores)
    write_csv(args.output_dir / "selection_manifest.csv", selected_rows)
    if rejected_rows:
        write_csv(args.output_dir / "rejected_candidates.csv", rejected_rows)

    frame = pd.DataFrame(scores)
    pivot = frame.pivot(index="base_id", columns="condition", values="prediction_mse")
    rng = np.random.default_rng(20260823)
    summaries = []
    for condition in frame.condition.drop_duplicates():
        values = pivot[condition].to_numpy()
        low, high = bootstrap_mean(values, rng)
        metadata = frame[frame.condition == condition].iloc[0]
        summaries.append(
            {
                "condition": condition,
                "action_mode": metadata.action_mode,
                "action_scale": metadata.action_scale,
                "intended_displacement_scale": metadata.intended_displacement_scale,
                "n_episodes": len(values),
                "score_mean": values.mean(),
                "score_ci95_low": low,
                "score_ci95_high": high,
                "score_median": np.median(values),
                "actual_displacement_pixels_mean": frame[
                    frame.condition == condition
                ].actual_displacement_pixels.mean(),
            }
        )
    write_csv(args.output_dir / "condition_summary.csv", summaries)

    contrast_rows = []
    for scale in ACTION_SCALES[1:]:
        token = scale_token(scale)
        lawful = pivot[f"lawful_scale_{token}"]
        for mode in ("zero", "opposite"):
            difference = (pivot[f"{mode}_scale_{token}"] - lawful).to_numpy()
            low, high = bootstrap_mean(difference, rng)
            contrast_rows.append(
                {
                    "action_mode": mode,
                    "intended_displacement_scale": scale,
                    "n_episodes": len(difference),
                    "inconsistent_minus_lawful_mean": difference.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "median": np.median(difference),
                    "positive": int((difference > 0).sum()),
                    "actual_displacement_pixels_mean": frame[
                        frame.condition == f"lawful_scale_{token}"
                    ].actual_displacement_pixels.mean(),
                }
            )
    write_csv(args.output_dir / "same_target_contrasts.csv", contrast_rows)

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
            physical_condition = f"{mode}_scale_{scale_token(scale)}"
            difference = (pivot[hue_condition] - pivot[physical_condition]).to_numpy()
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

    summary_frame = pd.DataFrame(summaries)
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for mode, marker in [("lawful", "o"), ("zero", "s"), ("opposite", "^")]:
        group = summary_frame[summary_frame.action_mode == mode].sort_values(
            "actual_displacement_pixels_mean"
        )
        axis.plot(
            group.actual_displacement_pixels_mean,
            group.score_mean,
            marker=marker,
            label=mode,
        )
    axis.set_xlabel("Simulator displacement (pixels)")
    axis.set_ylabel("LeWM prediction MSE")
    axis.set_title("TwoRoom: identical target, different final action block")
    axis.legend()
    figure.savefig(args.output_dir / "controlled_displacement.png", dpi=180)
    plt.close(figure)

    provenance = {
        "status": f"complete_{args.phase}",
        "phase": args.phase,
        "task": "tworoom",
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "samples": args.samples,
        "action_scales": ACTION_SCALES,
        "hue_scale_matches": HUE_SCALE_MATCHES,
        "model_snapshot": str(snapshot),
        "weights_sha256": "566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd",
        "state_dict_keys": state_dict_keys,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "source_selection_manifest": str(args.selection_manifest.resolve()),
        "source_selection_manifest_sha256": sha256_file(args.selection_manifest),
        "maximum_official_dynamics_position_error": maximum_official_dynamics_position_error,
        "maximum_same_target_embedding_difference": maximum_same_target_embedding_difference,
        "script_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "wall_seconds": time.time() - started,
        "limitations": [
            "Same-target counterfactuals test action-conditioned consistency, not the unreleased paper teleport intervention.",
            "Scaled official action blocks are simulator-lawful but may differ in frequency from the training action distribution.",
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
                "rejected_candidates": len(rejected_rows),
            },
            indent=2,
        )
        + "\n"
    )
    print((args.output_dir / "status.json").read_text())


if __name__ == "__main__":
    main()
