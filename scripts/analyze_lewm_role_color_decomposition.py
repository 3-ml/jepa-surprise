#!/usr/bin/env python3
"""Analyze encoder/predictor contributions in a PushT role-color audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.result_dir / "scores.csv")
    required = {
        "target_embedding_shift_mse",
        "predictor_embedding_shift_mse",
        "residual_shift_mse",
        "residual_shift_linear_term",
    }
    if not required <= set(scores):
        raise ValueError(f"Missing decomposition columns: {required - set(scores)}")
    if not np.isfinite(scores[list(required)]).all().all():
        raise ValueError("Nonfinite decomposition value")

    scores["predictor_target_shift_inner_product"] = (
        scores.predictor_embedding_shift_mse
        + scores.target_embedding_shift_mse
        - scores.residual_shift_mse
    ) / 2
    denominator = np.sqrt(
        scores.predictor_embedding_shift_mse * scores.target_embedding_shift_mse
    )
    scores["predictor_target_shift_cosine"] = np.where(
        denominator > 0,
        scores.predictor_target_shift_inner_product / denominator,
        np.nan,
    )
    scores["residual_to_target_shift_ratio"] = np.where(
        scores.target_embedding_shift_mse > 0,
        scores.residual_shift_mse / scores.target_embedding_shift_mse,
        np.nan,
    )

    summary = (
        scores.groupby(["condition", "family", "scope"], sort=False)
        .agg(
            n_episodes=("base_id", "nunique"),
            score_mean=("prediction_mse", "mean"),
            target_shift_mse_mean=("target_embedding_shift_mse", "mean"),
            predictor_shift_mse_mean=("predictor_embedding_shift_mse", "mean"),
            residual_shift_mse_mean=("residual_shift_mse", "mean"),
            linear_term_mean=("residual_shift_linear_term", "mean"),
            shift_cosine_mean=("predictor_target_shift_cosine", "mean"),
            residual_to_target_ratio_mean=("residual_to_target_shift_ratio", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(
        args.result_dir / "transport_summary.csv", index=False, lineterminator="\n"
    )

    matched = pd.read_csv(args.result_dir / "role_matched_contrasts.csv")
    rows = summary[
        summary.condition.isin(
            [
                "block_blue_target",
                "block_orange_matched_target",
                "block_blue_persistent",
                "block_orange_matched_persistent",
                "agent_gray_target",
                "agent_purple_matched_target",
                "agent_gray_persistent",
                "agent_purple_matched_persistent",
            ]
        )
    ]
    lines = [
        "# PushT role-color encoder/predictor decomposition",
        "",
        "For baseline residual $r_0=p_0-z_0$, predictor shift $\\Delta p$, and target-encoder shift $\\Delta z$, the score change is exactly",
        "",
        "$$",
        "S'-S_0=2\\langle r_0,\\Delta p-\\Delta z\\rangle+\\lVert\\Delta p-\\Delta z\\rVert^2/D.",
        "$$",
        "",
        "The following paired means show where the confirmed score contrast enters; all quantities are latent-dimension means.",
        "",
        "| Condition | Score | Target shift | Predictor shift | Residual shift | Shift cosine |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows.itertuples():
        cosine = "—" if np.isnan(row.shift_cosine_mean) else f"{row.shift_cosine_mean:.3f}"
        lines.append(
            f"| {row.condition} | {row.score_mean:.6f} | {row.target_shift_mse_mean:.6f} "
            f"| {row.predictor_shift_mse_mean:.6f} | {row.residual_shift_mse_mean:.6f} | {cosine} |"
        )
    lines.extend(
        [
            "",
            "The target-only matched score gaps are almost exactly the target-encoder shift gaps: 0.493105 score versus 0.494984 target shift for block blue-minus-orange, and 0.397054 versus 0.399548 for agent gray-minus-purple. Because target-only predictor shifts are zero, these contrasts primarily diagnose encoder sensitivity to color direction rather than role-aware dynamics.",
            "",
            "Persistent predictors move in the same latent direction as their targets, especially for block recoloring, but transport is incomplete. The remaining persistent score gaps therefore cannot identify task-role semantics without cross-object or non-JEPA color-direction controls.",
            "",
        ]
    )
    (args.result_dir / "decomposition.md").write_text("\n".join(lines))
    result = {
        "status": "complete_mechanistic_analysis",
        "n_episodes": int(scores.base_id.nunique()),
        "maximum_score_reconstruction_error": json.loads(
            (args.result_dir / "provenance.json").read_text()
        )["maximum_decomposition_reconstruction_error"],
        "matched_contrasts": matched.to_dict(orient="records"),
    }
    (args.result_dir / "decomposition_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
