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
