from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.run_lewm_prediction_error_audit import (
    INVARIANCE_ATOL,
    load_model,
    make_dataset,
    normalize_action_blocks,
    sha256_file,
    visual_difference,
    write_csv,
)


@dataclass(frozen=True)
class ColorIntervention:
    name: str
    variation_key: str
    rgb: tuple[int, int, int]
    role_collision: str


INTERVENTIONS = (
    ColorIntervention("block_blue", "block.color", (65, 105, 225), "agent"),
    ColorIntervention("block_green", "block.color", (144, 238, 144), "goal"),
    ColorIntervention("block_red", "block.color", (220, 45, 45), "none"),
    ColorIntervention(
        "block_orange_matched", "block.color", (185, 120, 87), "none_matched_blue"
    ),
    ColorIntervention("agent_gray", "agent.color", (119, 136, 153), "block"),
    ColorIntervention("agent_green", "agent.color", (144, 238, 144), "goal"),
    ColorIntervention(
        "agent_purple_matched", "agent.color", (145, 70, 183), "none_matched_gray"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", default="pusht-role-color-audit-v1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--phase",
        choices=["exploratory", "confirmatory", "mechanistic"],
        default="exploratory",
    )
    return parser.parse_args()


def select_manifest(path: Path, count: int, seed: str) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if count > len(rows):
        raise ValueError(f"Requested {count} rows from a {len(rows)}-row manifest")
    material = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    rng = np.random.default_rng(material)
    indices = sorted(rng.choice(len(rows), size=count, replace=False).tolist())
    return [rows[index] for index in indices]


def render_sequence(states: torch.Tensor, intervention: ColorIntervention | None):
    import gymnasium as gym
    import stable_worldmodel

    kwargs = {}
    if intervention is not None:
        kwargs["init_value"] = {
            intervention.variation_key: np.asarray(intervention.rgb, dtype=np.uint8)
        }
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", **kwargs)
    frames = []
    realized_states = []
    try:
        for state in states.numpy():
            observation, _ = env.reset(seed=0, options={"state": state})
            frames.append(torch.from_numpy(env.render()).permute(2, 0, 1))
            realized_states.append(torch.from_numpy(observation["state"]).float())
    finally:
        env.close()
    return torch.stack(frames).to(torch.uint8), torch.stack(realized_states)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n: int = 10_000):
    draws = rng.integers(0, len(values), size=(n, len(values)))
    return tuple(np.quantile(values[draws].mean(axis=1), [0.025, 0.975]))


def contact_sheet(examples: list[dict], path: Path) -> None:
    conditions = [
        "stored_clean",
        "rerender_default",
        "block_blue_target",
        "block_orange_matched_target",
        "agent_gray_target",
        "agent_purple_matched_target",
    ]
    tile = 224
    header = 36
    canvas = Image.new(
        "RGB", (tile * len(conditions), (tile + header) * len(examples)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, example in enumerate(examples):
        y = row_index * (tile + header)
        for column, condition in enumerate(conditions):
            frame = example[condition][-1].permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(frame), (column * tile, y + header))
            draw.text((column * tile + 4, y + 4), condition, fill="black")
    canvas.save(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = torch.device(args.device)
    model, snapshot, _config, state_dict_keys = load_model("pusht", device)
    dataset, action_mean, action_std, preprocess = make_dataset("pusht", args.dataset)
    selection = select_manifest(args.selection_manifest, args.samples, args.seed)

    base_rows = []
    scores = []
    predictors = {}
    targets = {}
    visual_examples = []
    maximum_realized_state_difference = 0.0

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
                    predictors[(row["base_id"], row["condition"])] = predicted[
                        index, -1
                    ].cpu()
                    targets[(row["base_id"], row["condition"])] = encoded["emb"][
                        index, -1
                    ].cpu()
                    scores.append(
                        {
                            "task": "pusht",
                            "base_id": row["base_id"],
                            "episode": row["episode"],
                            "start_step": row["start_step"],
                            "condition": row["condition"],
                            "family": row["family"],
                            "scope": row["scope"],
                            "object_key": row["object_key"],
                            "role_collision": row["role_collision"],
                            "prediction_mse": float(mse[index, -1]),
                            "temporal_prediction_mse": json.dumps(
                                [float(value) for value in mse[index].cpu()]
                            ),
                            "target_rgb_l1_from_rerender_01": row[
                                "target_rgb_l1_from_rerender_01"
                            ],
                            "target_changed_pixel_fraction": row[
                                "target_changed_pixel_fraction"
                            ],
                            "target_changed_pixel_rgb_l1_01": row[
                                "target_changed_pixel_rgb_l1_01"
                            ],
                            "context_rgb_l1_from_rerender_01": row[
                                "context_rgb_l1_from_rerender_01"
                            ],
                        }
                    )

    for selection_row in selection:
        dataset_index = int(selection_row["dataset_index"])
        sample = dataset[dataset_index]
        base_id = selection_row["base_id"]
        stored = sample["pixels"].to(torch.uint8)
        default, default_states = render_sequence(sample["state"].float(), None)
        identity = visual_difference(default, stored)
        rendered = {"stored_clean": stored, "rerender_default": default}
        variant_states = {}
        for intervention in INTERVENTIONS:
            variant, realized_states = render_sequence(
                sample["state"].float(), intervention
            )
            rendered[f"{intervention.name}_persistent"] = variant
            target = default.clone()
            target[-1] = variant[-1]
            rendered[f"{intervention.name}_target"] = target
            variant_states[intervention.name] = realized_states
            state_difference = float((realized_states - default_states).abs().max())
            maximum_realized_state_difference = max(
                maximum_realized_state_difference, state_difference
            )
        if maximum_realized_state_difference != 0:
            raise RuntimeError("Color intervention changed realized simulator state")

        base_rows.append(
            {
                "base_id": base_id,
                "dataset_index": dataset_index,
                "episode": int(selection_row["episode"]),
                "start_step": int(selection_row["start_step"]),
                "stored_rerender_rgb_l1_01": identity["rgb_l1_01"],
                "stored_rerender_rgb_rmse_01": identity["rgb_rmse_01"],
                "stored_rerender_changed_pixel_fraction": identity[
                    "changed_pixel_fraction"
                ],
                "source_to_realized_state_max_abs": float(
                    (sample["state"].float() - default_states).abs().max()
                ),
            }
        )
        actions = normalize_action_blocks(
            sample["action"].float(), action_mean, action_std
        )
        condition_metadata = {
            "stored_clean": ("identity", "stored", "none", "none"),
            "rerender_default": ("identity", "rerender", "none", "none"),
        }
        for intervention in INTERVENTIONS:
            condition_metadata[f"{intervention.name}_target"] = (
                intervention.name,
                "target",
                intervention.variation_key,
                intervention.role_collision,
            )
            condition_metadata[f"{intervention.name}_persistent"] = (
                intervention.name,
                "persistent",
                intervention.variation_key,
                intervention.role_collision,
            )
        base_work = []
        for condition_name, pixels in rendered.items():
            family, scope, object_key, role_collision = condition_metadata[
                condition_name
            ]
            target_difference = visual_difference(pixels[-1], default[-1])
            context_difference = visual_difference(pixels[:-1], default[:-1])
            base_work.append(
                {
                    "base_id": base_id,
                    "episode": int(selection_row["episode"]),
                    "start_step": int(selection_row["start_step"]),
                    "condition": condition_name,
                    "family": family,
                    "scope": scope,
                    "object_key": object_key,
                    "role_collision": role_collision,
                    "pixels": preprocess({"pixels": pixels})["pixels"],
                    "actions": actions,
                    "target_rgb_l1_from_rerender_01": target_difference["rgb_l1_01"],
                    "target_changed_pixel_fraction": target_difference[
                        "changed_pixel_fraction"
                    ],
                    "target_changed_pixel_rgb_l1_01": target_difference[
                        "changed_pixel_rgb_l1_01"
                    ],
                    "context_rgb_l1_from_rerender_01": context_difference["rgb_l1_01"],
                }
            )
        score_items(base_work)
        if len(visual_examples) < 4:
            visual_examples.append(rendered)
    if not all(np.isfinite(row["prediction_mse"]) for row in scores):
        raise RuntimeError("Nonfinite prediction score")

    maximum_target_only_predictor_difference = 0.0
    for base in base_rows:
        base_id = base["base_id"]
        baseline = predictors[(base_id, "rerender_default")]
        for intervention in INTERVENTIONS:
            condition = f"{intervention.name}_target"
            difference = float(
                (predictors[(base_id, condition)] - baseline).abs().max()
            )
            maximum_target_only_predictor_difference = max(
                maximum_target_only_predictor_difference, difference
            )
    if maximum_target_only_predictor_difference > INVARIANCE_ATOL:
        raise RuntimeError(
            f"Target-only intervention changed predictor: {maximum_target_only_predictor_difference}"
        )

    maximum_decomposition_reconstruction_error = 0.0
    for row in scores:
        key = (row["base_id"], row["condition"])
        baseline_key = (row["base_id"], "rerender_default")
        predictor_shift = predictors[key] - predictors[baseline_key]
        target_shift = targets[key] - targets[baseline_key]
        baseline_residual = predictors[baseline_key] - targets[baseline_key]
        residual_shift = predictor_shift - target_shift
        linear_term = 2 * (baseline_residual * residual_shift).mean()
        quadratic_term = residual_shift.pow(2).mean()
        baseline_score = baseline_residual.pow(2).mean()
        reconstructed_score = baseline_score + linear_term + quadratic_term
        error = abs(float(reconstructed_score) - row["prediction_mse"])
        maximum_decomposition_reconstruction_error = max(
            maximum_decomposition_reconstruction_error, error
        )
        row.update(
            {
                "target_embedding_shift_mse": float(target_shift.pow(2).mean()),
                "predictor_embedding_shift_mse": float(predictor_shift.pow(2).mean()),
                "residual_shift_mse": float(quadratic_term),
                "residual_shift_linear_term": float(linear_term),
                "decomposed_score_delta": float(linear_term + quadratic_term),
            }
        )
    if maximum_decomposition_reconstruction_error > INVARIANCE_ATOL:
        raise RuntimeError(
            "Prediction-error decomposition failed to reconstruct scores: "
            f"{maximum_decomposition_reconstruction_error}"
        )

    write_csv(args.output_dir / "scores.csv", scores)
    write_csv(args.output_dir / "selection_manifest.csv", base_rows)
    contact_sheet(visual_examples, args.output_dir / "contact_sheet.png")

    import pandas as pd

    frame = pd.DataFrame(scores)
    pivot = frame.pivot(index="base_id", columns="condition", values="prediction_mse")
    rng = np.random.default_rng(20260823)
    summaries = []
    for condition in frame.condition.drop_duplicates():
        delta = (pivot[condition] - pivot.rerender_default).to_numpy()
        low, high = bootstrap_mean(delta, rng)
        metadata = frame[frame.condition == condition].iloc[0]
        summaries.append(
            {
                "condition": condition,
                "family": metadata.family,
                "scope": metadata.scope,
                "object_key": metadata.object_key,
                "role_collision": metadata.role_collision,
                "n_episodes": len(delta),
                "score_mean": pivot[condition].mean(),
                "delta_from_rerender_mean": delta.mean(),
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "delta_median": np.median(delta),
                "positive": int((delta > 0).sum()),
                "target_rgb_l1_from_rerender_01_mean": frame[
                    frame.condition == condition
                ].target_rgb_l1_from_rerender_01.mean(),
                "target_changed_pixel_fraction_mean": frame[
                    frame.condition == condition
                ].target_changed_pixel_fraction.mean(),
                "target_embedding_shift_mse_mean": frame[
                    frame.condition == condition
                ].target_embedding_shift_mse.mean(),
                "predictor_embedding_shift_mse_mean": frame[
                    frame.condition == condition
                ].predictor_embedding_shift_mse.mean(),
                "residual_shift_mse_mean": frame[
                    frame.condition == condition
                ].residual_shift_mse.mean(),
                "residual_shift_linear_term_mean": frame[
                    frame.condition == condition
                ].residual_shift_linear_term.mean(),
            }
        )
    write_csv(args.output_dir / "condition_summary.csv", summaries)

    scope_rows = []
    for intervention in INTERVENTIONS:
        difference = (
            pivot[f"{intervention.name}_target"]
            - pivot[f"{intervention.name}_persistent"]
        ).to_numpy()
        low, high = bootstrap_mean(difference, rng)
        scope_rows.append(
            {
                "intervention": intervention.name,
                "role_collision": intervention.role_collision,
                "target_minus_persistent_mean": difference.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "positive": int((difference > 0).sum()),
            }
        )
    write_csv(args.output_dir / "target_minus_persistent_summary.csv", scope_rows)

    target_shift_pivot = frame.pivot(
        index="base_id", columns="condition", values="target_embedding_shift_mse"
    )
    predictor_shift_pivot = frame.pivot(
        index="base_id", columns="condition", values="predictor_embedding_shift_mse"
    )
    matched_rows = []
    for role_condition, matched_condition in [
        ("block_blue", "block_orange_matched"),
        ("agent_gray", "agent_purple_matched"),
    ]:
        for scope in ("target", "persistent"):
            difference = (
                pivot[f"{role_condition}_{scope}"]
                - pivot[f"{matched_condition}_{scope}"]
            ).to_numpy()
            low, high = bootstrap_mean(difference, rng)
            role_key = f"{role_condition}_{scope}"
            matched_key = f"{matched_condition}_{scope}"
            target_shift_difference = (
                target_shift_pivot[role_key] - target_shift_pivot[matched_key]
            ).to_numpy()
            predictor_shift_difference = (
                predictor_shift_pivot[role_key] - predictor_shift_pivot[matched_key]
            ).to_numpy()
            target_low, target_high = bootstrap_mean(target_shift_difference, rng)
            predictor_low, predictor_high = bootstrap_mean(
                predictor_shift_difference, rng
            )
            matched_rows.append(
                {
                    "role_condition": role_condition,
                    "matched_novel_condition": matched_condition,
                    "scope": scope,
                    "n_episodes": len(difference),
                    "role_minus_matched_score_mean": difference.mean(),
                    "score_ci95_low": low,
                    "score_ci95_high": high,
                    "score_median": np.median(difference),
                    "score_positive": int((difference > 0).sum()),
                    "role_minus_matched_target_shift_mse_mean": target_shift_difference.mean(),
                    "target_shift_ci95_low": target_low,
                    "target_shift_ci95_high": target_high,
                    "role_minus_matched_predictor_shift_mse_mean": predictor_shift_difference.mean(),
                    "predictor_shift_ci95_low": predictor_low,
                    "predictor_shift_ci95_high": predictor_high,
                }
            )
    write_csv(args.output_dir / "role_matched_contrasts.csv", matched_rows)

    provenance = {
        "status": f"complete_{args.phase}",
        "phase": args.phase,
        "task": "pusht",
        "command": " ".join(sys.argv),
        "model_snapshot": str(snapshot),
        "weights_sha256": "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607",
        "state_dict_keys": state_dict_keys,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "source_selection_manifest": str(args.selection_manifest.resolve()),
        "source_selection_manifest_sha256": sha256_file(args.selection_manifest),
        "seed": args.seed,
        "samples": args.samples,
        "interventions": [intervention.__dict__ for intervention in INTERVENTIONS],
        "renderer": "installed stable_worldmodel swm/PushT-v1, Pymunk/Pygame",
        "maximum_realized_state_difference_across_colors": maximum_realized_state_difference,
        "maximum_target_only_predictor_difference": maximum_target_only_predictor_difference,
        "maximum_decomposition_reconstruction_error": maximum_decomposition_reconstruction_error,
        "script_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "wall_seconds": time.time() - started,
        "limitations": (
            [
                "Exploratory role-color screen; selected results require a fresh locked rerun."
            ]
            if args.phase == "exploratory"
            else []
        )
        + [
            "The current renderer is close but not byte-identical to stored training observations; rerender_default is the paired baseline.",
            "Object colors are task-role cues in PushT and are not labeled as pure nuisances.",
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
                "conditions": len(frame.condition.unique()),
                "scores": len(scores),
            },
            indent=2,
        )
        + "\n"
    )
    print((args.output_dir / "status.json").read_text())


if __name__ == "__main__":
    main()
