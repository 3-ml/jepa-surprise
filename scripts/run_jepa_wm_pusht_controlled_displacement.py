#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torchvision.transforms import functional as tvf

ACTION_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
HUE_SCALE_MATCHES = {30.0: 0.25, 60.0: 0.75, 120.0: 1.0, 180.0: 1.0}
EXPECTED_JEPA_WM_REVISION = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
EXPECTED_DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
EXPECTED_STABLE_WM_REVISION = "0ef3856875e70a1283e637fcd2ab936eae6c4e6f"
EXPECTED_CHECKPOINT_SHA256 = "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"
EXPECTED_DATASET_SHA256 = "b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb"
INVARIANCE_ATOL = 1e-6
REPEAT_ATOL = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-wm-root", type=Path, required=True)
    parser.add_argument("--stable-worldmodel-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--bundle-provenance", type=Path, required=True)
    parser.add_argument("--reference-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--phase", choices=["smoke", "exploratory"], default="smoke")
    parser.add_argument("--seed", default="jepa-wm-pusht-controlled-displacement-v1")
    parser.add_argument("--hf-revision", default="9b9c41ef249466630dbf1a20e78391865d07b3b9")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def verify_revision(root: Path, expected: str, name: str) -> str:
    actual = git_revision(root)
    if actual != expected:
        raise RuntimeError(f"{name} revision mismatch: {actual} != {expected}")
    return actual


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapping = {
        "state_encoder.weight": "proprio_encoder.weight",
        "state_encoder.bias": "proprio_encoder.bias",
    }
    return {mapping.get(key.replace("module.", ""), key.replace("module.", "")): value for key, value in state.items()}


def validate_loaded_module(module: torch.nn.Module | None, state: dict | None, name: str) -> dict:
    if module is None and state is None:
        return {"name": name, "status": "absent"}
    if module is None or state is None:
        raise RuntimeError(f"Checkpoint/module presence mismatch for {name}")
    result = module.load_state_dict(clean_state_dict(state), strict=False)
    allowed_missing = [key for key in result.missing_keys if "attn_mask" in key or key.endswith(".bias")]
    disallowed_missing = sorted(set(result.missing_keys) - set(allowed_missing))
    if disallowed_missing or result.unexpected_keys:
        raise RuntimeError(
            f"{name} checkpoint mismatch: missing={disallowed_missing}, unexpected={result.unexpected_keys}"
        )
    return {
        "name": name,
        "status": "loaded",
        "checkpoint_keys": len(state),
        "module_keys": len(module.state_dict()),
        "allowed_missing_keys": sorted(allowed_missing),
        "unexpected_keys": list(result.unexpected_keys),
    }


def load_model(args: argparse.Namespace, device: torch.device):
    jepa_revision = verify_revision(args.jepa_wm_root, EXPECTED_JEPA_WM_REVISION, "JEPA-WM")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {checkpoint_sha256}")
    sys.path.insert(0, str(args.jepa_wm_root.resolve()))
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

    config_path = (
        args.jepa_wm_root
        / "configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    model_kwargs = copy.deepcopy(config["model_kwargs"]["pretrain_kwargs"])
    model_kwargs["heads_cfg"] = {"architectures": {}, "pretrain_dec_path": None}
    cfg_data = config["model_kwargs"]["data"]
    cfg_aug = config["model_kwargs"]["data_aug"]
    transform = make_transforms(
        img_size=int(cfg_data["img_size"]),
        normalize=cfg_aug["normalize"],
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    inverse_transform = make_inverse_transforms(img_size=int(cfg_data["img_size"]), **cfg_aug)
    stats = get_data_stats("pusht")
    preprocessor = Preprocessor(
        action_mean=torch.tensor(stats["action_mean"]),
        action_std=torch.tensor(stats["action_std"]),
        state_mean=torch.tensor(stats["state_mean"]),
        state_std=torch.tensor(stats["state_std"]),
        proprio_mean=torch.tensor(stats["proprio_mean"]),
        proprio_std=torch.tensor(stats["proprio_std"]),
        transform=transform,
        inverse_transform=inverse_transform,
    )
    model = init_module(
        folder=args.checkpoint.parent,
        checkpoint=args.checkpoint.name,
        model_kwargs=model_kwargs,
        device=device,
        action_dim=int(stats["action_dim"]),
        proprio_dim=int(stats["proprio_dim"]),
        preprocessor=preprocessor,
        cfgs_data=cfg_data,
        wrapper_kwargs=config["model_kwargs"]["wrapper_kwargs"],
    )
    model.requires_grad_(False)
    model.eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_validation = [
        validate_loaded_module(model.model.predictor, checkpoint.get("predictor"), "predictor"),
        validate_loaded_module(model.model.action_encoder, checkpoint.get("action_encoder"), "action_encoder"),
        validate_loaded_module(model.model.proprio_encoder, checkpoint.get("proprio_encoder"), "proprio_encoder"),
    ]
    return model, preprocessor, config_path, config, checkpoint, load_validation, jepa_revision, checkpoint_sha256


def load_bundle(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict]:
    bundle_provenance = json.loads(args.bundle_provenance.read_text())
    if bundle_provenance["bundle_sha256"] != sha256_file(args.input_bundle):
        raise RuntimeError("Input bundle SHA-256 mismatch")
    if bundle_provenance["dataset_sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Source dataset SHA-256 mismatch")
    loaded = np.load(args.input_bundle, allow_pickle=False)
    bundle = {key: loaded[key] for key in loaded.files}
    loaded.close()
    required = {"base_id", "dataset_index", "episode", "start_step", "pixels", "action", "state", "episode_idx", "step_idx"}
    if set(bundle) != required:
        raise RuntimeError(f"Unexpected bundle arrays: {set(bundle)}")
    count = len(bundle["base_id"])
    expected_shapes = {
        "pixels": (count, 4, 3, 224, 224),
        "action": (count, 4, 10),
        "state": (count, 4, 7),
        "episode_idx": (count, 4),
        "step_idx": (count, 4),
    }
    for key, shape in expected_shapes.items():
        if bundle[key].shape != shape:
            raise RuntimeError(f"Unexpected {key} shape: {bundle[key].shape} != {shape}")
    if args.samples < 1 or args.samples > count:
        raise ValueError(f"samples must be between 1 and {count}")
    return bundle, bundle_provenance


def load_target_references(path: Path) -> dict[tuple[str, str], str]:
    references = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["base_id"], row["condition"])
            target_hash = row["target_sha256"]
            prior = references.setdefault(key, target_hash)
            if prior != target_hash:
                raise RuntimeError(f"Conflicting target hashes for {key}")
    return references


def scale_token(value: float) -> str:
    return str(value).replace(".", "p")


def angle_difference(a: float, b: float) -> float:
    return float(abs((a - b + np.pi) % (2 * np.pi) - np.pi))


def visual_difference(changed: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    absolute = (changed.float() - reference.float()).abs() / 255.0
    changed_pixels = (changed != reference).any(dim=0)
    return {
        "rgb_l1_01": float(absolute.mean()),
        "changed_pixel_fraction": float(changed_pixels.float().mean()),
    }


def simulate_target(state: torch.Tensor, actions: torch.Tensor):
    import gymnasium as gym

    __import__("stable_worldmodel")
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


def normalize_action_blocks(actions: torch.Tensor, preprocessor) -> torch.Tensor:
    if actions.shape[-1] % 2 != 0:
        raise ValueError(f"Unexpected action shape: {tuple(actions.shape)}")
    elementary = actions.reshape(*actions.shape[:-1], -1, 2) / 100.0
    normalized = preprocessor.normalize_actions(elementary)
    return torch.nan_to_num(normalized.reshape(actions.shape).float(), 0.0)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n: int = 10_000) -> tuple[float, float]:
    draws = rng.integers(0, len(values), size=(n, len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.time()
    if args.stable_worldmodel_root.exists():
        verify_revision(args.stable_worldmodel_root, EXPECTED_STABLE_WM_REVISION, "stable-worldmodel")
        sys.path.insert(0, str(args.stable_worldmodel_root.resolve()))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(0)
    np.random.seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    bundle, bundle_provenance = load_bundle(args)
    target_references = load_target_references(args.reference_scores)
    (
        model,
        preprocessor,
        config_path,
        config,
        checkpoint,
        load_validation,
        jepa_revision,
        checkpoint_sha256,
    ) = load_model(args, device)

    selected = []
    scores = []
    target_embeddings = {}
    context_embeddings = {}
    target_hash_mismatches = []
    maximum_start_state_error = 0.0
    maximum_official_endpoint_error = 0.0
    maximum_repeat_score_difference = 0.0
    repeat_checked = False

    def encode_episode_inputs(items: list[dict]):
        reference_visual = items[0]["visual"][:3]
        reference_proprio = items[0]["proprio"][:3]
        for row in items[1:]:
            if not torch.equal(row["visual"][:3], reference_visual):
                raise RuntimeError(f"Context pixel mismatch for {row['base_id']}")
            if not torch.equal(row["proprio"][:3], reference_proprio):
                raise RuntimeError(f"Context proprioception mismatch for {row['base_id']}")
        context = model.encode(
            {
                "visual": reference_visual.unsqueeze(0).to(device),
                "proprio": reference_proprio.unsqueeze(0).to(device),
            }
        )
        representative = {}
        for row in items:
            prior = representative.setdefault(row["target_group"], row)
            if not torch.equal(prior["visual"][-1], row["visual"][-1]):
                raise RuntimeError(f"Target pixel mismatch within {row['target_group']}")
        groups = list(representative)
        target_rows = [representative[group] for group in groups]
        targets = model.encode(
            {
                "visual": torch.stack([row["visual"][-1:] for row in target_rows]).to(device),
                "proprio": torch.stack([row["proprio"][-1:] for row in target_rows]).to(device),
            }
        )["visual"][:, 0]
        return context["visual"], context["proprio"], {
            group: targets[index] for index, group in enumerate(groups)
        }

    def forward_batch(batch: list[dict], context_visual, context_proprio, target_visual):
        actions = torch.stack([row["actions"] for row in batch]).to(device)
        batch_size = len(batch)
        visual_context = context_visual.expand(batch_size, *context_visual.shape[1:])
        proprio_context = context_proprio.expand(batch_size, *context_proprio.shape[1:])
        action_features = model.model.encode_act(actions[:, :3])
        predicted, _, _ = model.model.forward_pred(
            visual_context,
            action_features,
            proprio_context,
        )
        final_targets = torch.stack([target_visual[row["target_group"]] for row in batch]).unsqueeze(1)
        temporal_targets = torch.cat(
            [visual_context[:, 1:3], final_targets],
            dim=1,
        )
        temporal_mse = (predicted - temporal_targets).pow(2).flatten(2).mean(dim=2)
        return temporal_mse

    def score_items(items: list[dict]) -> None:
        nonlocal maximum_repeat_score_difference, repeat_checked
        with torch.inference_mode():
            context_visual, context_proprio, target_visual = encode_episode_inputs(items)
            for offset in range(0, len(items), args.batch_size):
                batch = items[offset : offset + args.batch_size]
                temporal_mse = forward_batch(batch, context_visual, context_proprio, target_visual)
                if not repeat_checked:
                    repeated_mse = forward_batch(batch, context_visual, context_proprio, target_visual)
                    maximum_repeat_score_difference = max(
                        maximum_repeat_score_difference,
                        float((temporal_mse - repeated_mse).abs().max().cpu()),
                    )
                    repeat_checked = True
                for index, row in enumerate(batch):
                    context_embeddings[(row["base_id"], row["condition"])] = context_visual[0].cpu()
                    target_embeddings[(row["base_id"], row["target_group"], row["condition"])] = target_visual[
                        row["target_group"]
                    ].cpu()
                    scores.append(
                        {
                            "task": "pusht",
                            "model": "JEPA-WM",
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
                            "prediction_mse": float(temporal_mse[index, -1].cpu()),
                            "temporal_prediction_mse": json.dumps(
                                [float(value) for value in temporal_mse[index].cpu()]
                            ),
                            "target_rgb_l1_from_static_01": row["target_rgb_l1_from_static_01"],
                            "target_changed_pixel_fraction_from_static": row[
                                "target_changed_pixel_fraction_from_static"
                            ],
                            "target_sha256": row["target_sha256"],
                        }
                    )

    for bundle_index in range(args.samples):
        base_id = str(bundle["base_id"][bundle_index])
        episode = int(bundle["episode"][bundle_index])
        start_step = int(bundle["start_step"][bundle_index])
        stored = torch.from_numpy(bundle["pixels"][bundle_index]).to(torch.uint8)
        stored_states = torch.from_numpy(bundle["state"][bundle_index]).float()
        stored_actions = torch.from_numpy(bundle["action"][bundle_index]).float()
        if not torch.isfinite(stored_actions).all() or not torch.isfinite(stored_states).all():
            raise RuntimeError(f"Nonfinite source data for {base_id}")
        action_block = stored_actions[2].reshape(5, 2)
        source_state = stored_states[2]
        official_state = stored_states[3]
        rendered = {}
        states = {}
        start_errors = []
        for scale in ACTION_SCALES:
            realized_start, final_state, frame = simulate_target(source_state, action_block * scale)
            start_errors.append(float((realized_start - source_state).abs().max()))
            rendered[scale] = frame
            states[scale] = final_state
        start_error = max(start_errors)
        endpoint_error = float((states[1.0] - official_state).abs().max())
        maximum_start_state_error = max(maximum_start_state_error, start_error)
        maximum_official_endpoint_error = max(maximum_official_endpoint_error, endpoint_error)
        if start_error > 2e-5 or endpoint_error > 2e-4:
            raise RuntimeError(f"Simulator identity failure for {base_id}: start={start_error}, endpoint={endpoint_error}")

        static_frame = rendered[0.0]
        natural_agent = float(torch.linalg.vector_norm(official_state[:2] - source_state[:2]))
        natural_block = float(torch.linalg.vector_norm(official_state[2:4] - source_state[2:4]))
        natural_angle = angle_difference(float(official_state[4]), float(source_state[4]))
        natural_identity = visual_difference(rendered[1.0], stored[-1])
        boundary_identity = visual_difference(static_frame, stored[2])
        selected.append(
            {
                "base_id": base_id,
                "dataset_index": int(bundle["dataset_index"][bundle_index]),
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

        items = []

        def add_item(
            condition: str,
            target_group: str,
            action_mode: str,
            action_scale: float,
            intended_scale: float,
            target_frame: torch.Tensor,
            target_state: torch.Tensor,
            final_block: torch.Tensor,
        ) -> None:
            pixels = torch.cat([stored[:3], target_frame.unsqueeze(0)], dim=0)
            states_for_proprio = stored_states.clone()
            states_for_proprio[3] = target_state
            raw_actions = stored_actions.clone()
            raw_actions[2] = final_block.reshape(-1)
            target_hash = hashlib.sha256(target_frame.numpy().tobytes()).hexdigest()
            expected_hash = target_references.get((base_id, condition))
            if expected_hash != target_hash:
                target_hash_mismatches.append(
                    {"base_id": base_id, "condition": condition, "expected": expected_hash, "actual": target_hash}
                )
            difference = visual_difference(target_frame, static_frame)
            items.append(
                {
                    "base_id": base_id,
                    "episode": episode,
                    "start_step": start_step,
                    "condition": condition,
                    "target_group": target_group,
                    "action_mode": action_mode,
                    "action_scale": action_scale,
                    "intended_displacement_scale": intended_scale,
                    "agent_displacement_state": float(
                        torch.linalg.vector_norm(target_state[:2] - states[0.0][:2])
                    ),
                    "block_displacement_state": float(
                        torch.linalg.vector_norm(target_state[2:4] - states[0.0][2:4])
                    ),
                    "block_angle_change_radians": angle_difference(
                        float(target_state[4]), float(states[0.0][4])
                    ),
                    "visual": pixels,
                    "proprio": states_for_proprio[:, [0, 1, 5, 6]],
                    "actions": normalize_action_blocks(raw_actions, preprocessor),
                    "target_rgb_l1_from_static_01": difference["rgb_l1_01"],
                    "target_changed_pixel_fraction_from_static": difference["changed_pixel_fraction"],
                    "target_sha256": target_hash,
                }
            )

        add_item(
            "official_clean",
            "official_stored",
            "official",
            1.0,
            1.0,
            stored[-1],
            official_state,
            action_block,
        )
        for scale in ACTION_SCALES:
            token = scale_token(scale)
            target_frame = rendered[scale]
            target_state = states[scale]
            add_item(
                f"lawful_scale_{token}",
                f"scale_{token}",
                "lawful",
                scale,
                scale,
                target_frame,
                target_state,
                action_block * scale,
            )
            if scale > 0:
                add_item(
                    f"zero_scale_{token}",
                    f"scale_{token}",
                    "zero",
                    0.0,
                    scale,
                    target_frame,
                    target_state,
                    torch.zeros_like(action_block),
                )
                add_item(
                    f"opposite_scale_{token}",
                    f"scale_{token}",
                    "opposite",
                    -scale,
                    scale,
                    target_frame,
                    target_state,
                    -action_block * scale,
                )
        for hue_degrees in HUE_SCALE_MATCHES:
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
        if target_hash_mismatches:
            raise RuntimeError(f"Target byte mismatch: {target_hash_mismatches[0]}")
        score_items(items)
        if (bundle_index + 1) % 8 == 0 or bundle_index + 1 == args.samples:
            print(f"processed {bundle_index + 1}/{args.samples} episodes", flush=True)

    if len(scores) != args.samples * 18:
        raise RuntimeError(f"Incomplete score matrix: {len(scores)} != {args.samples * 18}")
    if not all(np.isfinite(row["prediction_mse"]) for row in scores):
        raise RuntimeError("Nonfinite score")

    maximum_same_target_embedding_difference = 0.0
    maximum_same_context_embedding_difference = 0.0
    for base in selected:
        base_id = base["base_id"]
        conditions = [key[1] for key in context_embeddings if key[0] == base_id]
        context_reference = context_embeddings[(base_id, conditions[0])]
        for condition in conditions[1:]:
            maximum_same_context_embedding_difference = max(
                maximum_same_context_embedding_difference,
                float((context_embeddings[(base_id, condition)] - context_reference).abs().max()),
            )
        for scale in ACTION_SCALES:
            token = scale_token(scale)
            names = [f"lawful_scale_{token}"]
            if scale > 0:
                names.extend([f"zero_scale_{token}", f"opposite_scale_{token}"])
            reference = target_embeddings[(base_id, f"scale_{token}", names[0])]
            for condition in names[1:]:
                maximum_same_target_embedding_difference = max(
                    maximum_same_target_embedding_difference,
                    float((target_embeddings[(base_id, f"scale_{token}", condition)] - reference).abs().max()),
                )
    if maximum_same_target_embedding_difference > INVARIANCE_ATOL:
        raise RuntimeError(f"Same-target visual embedding invariance failed: {maximum_same_target_embedding_difference}")
    if maximum_same_context_embedding_difference > INVARIANCE_ATOL:
        raise RuntimeError(f"Same-context visual embedding invariance failed: {maximum_same_context_embedding_difference}")
    if maximum_repeat_score_difference > REPEAT_ATOL:
        raise RuntimeError(f"Repeated score instability: {maximum_repeat_score_difference}")

    write_csv(args.output_dir / "scores.csv", scores)
    write_csv(args.output_dir / "selection_manifest.csv", selected)
    frame = pd.DataFrame(scores)
    pivot = frame.pivot(index="base_id", columns="condition", values="prediction_mse")
    rng = np.random.default_rng(20260903)
    condition_summary = []
    for condition in frame["condition"].drop_duplicates():
        group = frame[frame.condition == condition]
        values = pivot[condition].to_numpy()
        low, high = bootstrap_mean(values, rng)
        first = group.iloc[0]
        condition_summary.append(
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
    write_csv(args.output_dir / "condition_summary.csv", condition_summary)

    same_target_contrasts = []
    for scale in ACTION_SCALES[1:]:
        token = scale_token(scale)
        lawful = pivot[f"lawful_scale_{token}"]
        for mode in ("zero", "opposite"):
            delta = pivot[f"{mode}_scale_{token}"] - lawful
            low, high = bootstrap_mean(delta.to_numpy(), rng)
            same_target_contrasts.append(
                {
                    "action_mode": mode,
                    "intended_displacement_scale": scale,
                    "n_episodes": len(delta),
                    "inconsistent_minus_lawful_mean": delta.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "median": delta.median(),
                    "positive": int((delta > 0).sum()),
                }
            )
    write_csv(args.output_dir / "same_target_contrasts.csv", same_target_contrasts)

    construct_comparisons = []
    static = pivot["lawful_scale_0p0"]
    for hue_degrees, scale in HUE_SCALE_MATCHES.items():
        hue_condition = f"appearance_hue_{int(hue_degrees):03d}"
        appearance_delta = pivot[hue_condition] - static
        appearance_low, appearance_high = bootstrap_mean(appearance_delta.to_numpy(), rng)
        row = {
            "hue_degrees": hue_degrees,
            "matched_displacement_scale": scale,
            "appearance_minus_static_mean": appearance_delta.mean(),
            "appearance_ci95_low": appearance_low,
            "appearance_ci95_high": appearance_high,
        }
        for mode in ("zero", "opposite"):
            difference = pivot[hue_condition] - pivot[f"{mode}_scale_{scale_token(scale)}"]
            low, high = bootstrap_mean(difference.to_numpy(), rng)
            row[f"appearance_minus_{mode}_mean"] = difference.mean()
            row[f"appearance_minus_{mode}_ci95_low"] = low
            row[f"appearance_minus_{mode}_ci95_high"] = high
        hue_rows = frame[frame.condition == hue_condition]
        physical_rows = frame[frame.condition == f"lawful_scale_{scale_token(scale)}"]
        row["appearance_rgb_l1_mean"] = hue_rows.target_rgb_l1_from_static_01.mean()
        row["physical_rgb_l1_mean"] = physical_rows.target_rgb_l1_from_static_01.mean()
        construct_comparisons.append(row)
    write_csv(args.output_dir / "construct_comparisons.csv", construct_comparisons)

    summary_frame = pd.DataFrame(condition_summary)
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for mode, marker in (("lawful", "o"), ("zero", "s"), ("opposite", "^")):
        group = summary_frame[summary_frame.action_mode == mode].sort_values("intended_displacement_scale")
        axis.plot(group.intended_displacement_scale, group.score_mean, marker=marker, label=mode)
    axis.set_xlabel("Official action-block scale")
    axis.set_ylabel("JEPA-WM visual embedding MSE")
    axis.set_title("PushT identical-target action audit")
    axis.legend()
    figure.savefig(args.output_dir / "controlled_displacement.png", dpi=180)
    plt.close(figure)

    torch_hub_dir = Path(torch.hub.get_dir())
    dinov2_roots = sorted(path for path in torch_hub_dir.glob("*dinov2*") if (path / ".git").exists())
    dinov2_revision = git_revision(dinov2_roots[0]) if dinov2_roots else None
    if dinov2_revision != EXPECTED_DINOV2_REVISION:
        raise RuntimeError(f"DINOv2 revision mismatch: {dinov2_revision}")
    dino_weight_files = sorted((torch_hub_dir / "checkpoints").glob("*dinov2*vits14*"))
    dino_weights = [
        {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in dino_weight_files
    ]
    environment = command_output([sys.executable, "-m", "pip", "freeze"])
    (args.output_dir / "environment.txt").write_text(environment + "\n")
    provenance = {
        "status": f"complete_{args.phase}",
        "phase": args.phase,
        "task": "pusht",
        "model": "JEPA-WM",
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "bootstrap_seed": 20260903,
        "samples": args.samples,
        "batch_size": args.batch_size,
        "action_scales": ACTION_SCALES,
        "hue_scale_matches": HUE_SCALE_MATCHES,
        "action_scale_before_normalization": 100.0,
        "reset_dt_compensation": 0.01,
        "jepa_wm_repo": str(args.jepa_wm_root.resolve()),
        "jepa_wm_revision": jepa_revision,
        "model_hf_revision": args.hf_revision,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint.get("epoch", -1),
        "checkpoint_load_validation": load_validation,
        "official_config": str(config_path.resolve()),
        "official_config_sha256": sha256_file(config_path),
        "dino_v2_roots": [str(path.resolve()) for path in dinov2_roots],
        "dino_v2_revision": dinov2_revision,
        "dino_v2_weights": dino_weights,
        "stable_worldmodel_repo": str(args.stable_worldmodel_root.resolve()),
        "stable_worldmodel_revision": git_revision(args.stable_worldmodel_root),
        "input_bundle": str(args.input_bundle.resolve()),
        "input_bundle_sha256": sha256_file(args.input_bundle),
        "input_bundle_provenance": bundle_provenance,
        "reference_scores": str(args.reference_scores.resolve()),
        "reference_scores_sha256": sha256_file(args.reference_scores),
        "target_hash_mismatches": len(target_hash_mismatches),
        "maximum_start_state_error": maximum_start_state_error,
        "maximum_official_endpoint_error": maximum_official_endpoint_error,
        "maximum_same_context_visual_embedding_difference": maximum_same_context_embedding_difference,
        "maximum_same_target_visual_embedding_difference": maximum_same_target_embedding_difference,
        "maximum_repeated_score_difference": maximum_repeat_score_difference,
        "shared_context_and_target_encoding": True,
        "script_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device": str(device),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "wall_seconds": time.time() - started,
        "score": "final one-step JEPA-WM visual embedding MSE over DINOv2 patch tokens against the encoded target",
        "claim_boundary": "Post-hoc exploratory robustness check using the same PushT episodes, target bytes, action interventions, rendered-effect matches, and bootstrap unit as the LeWorldModel and DINO-WM controlled audits.",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    shutil.copy2(Path(__file__), args.output_dir / Path(__file__).name)
    print(json.dumps({"output_dir": str(args.output_dir), "wall_seconds": provenance["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
