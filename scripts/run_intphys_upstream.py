#!/usr/bin/env python3
"""Run the pinned upstream IntPhys evaluator without editing its checkout."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

PINNED_UPSTREAM_REVISION = "302a2e821d908b4a8667e0c0e48fe733e43c9156"


def load_config(path: Path) -> dict:
    text = os.path.expandvars(path.read_text())
    if "${" in text:
        raise ValueError("Config contains unresolved environment variables")
    return yaml.safe_load(text)


def require_upstream_revision(root: Path, expected: str) -> None:
    command = ["git", "-C", str(root)]
    revision = subprocess.run(
        command + ["rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != expected:
        raise ValueError(f"Expected upstream revision {expected}, found {revision}")
    status = subprocess.run(
        command + ["status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("Upstream checkout has uncommitted changes")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path) -> dict[str, str]:
    command = ["git", "-C", str(root)]
    revision = subprocess.run(
        command + ["rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        command + ["diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "revision": revision,
        "worktree_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def make_dataset_view(
    root: Path, blocks: list[str], scene_limit: int | None
) -> tempfile.TemporaryDirectory | None:
    if scene_limit is None:
        return None
    temporary = tempfile.TemporaryDirectory(prefix="intphys-scenes-")
    view = Path(temporary.name)
    for block in blocks:
        destination = view / block
        destination.mkdir()
        scenes = sorted(path for path in (root / block).iterdir() if path.is_dir())
        if len(scenes) < scene_limit:
            temporary.cleanup()
            raise ValueError(f"{block} has only {len(scenes)} scenes")
        for scene in scenes[:scene_limit]:
            (destination / scene.name).symlink_to(scene)
    return temporary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block", choices=("O1", "O2", "O3"), action="append")
    parser.add_argument("--scene-limit", type=positive_int)
    parser.add_argument("--workers", type=positive_int, default=1)
    parser.add_argument(
        "--losses-only",
        action="store_true",
        help="Serialize surprise losses without computing dataset-level metrics.",
    )
    parser.add_argument(
        "--expected-upstream-revision", default=PINNED_UPSTREAM_REVISION
    )
    parser.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="Use O1, one scene, one context, and FP32 CPU inference.",
    )
    args = parser.parse_args()

    evaluation_root = args.upstream_root / "evaluation_code"
    if not (evaluation_root / "evals/intuitive_physics/eval.py").is_file():
        raise FileNotFoundError(
            f"Not a jepa-intuitive-physics checkout: {args.upstream_root}"
        )
    require_upstream_revision(args.upstream_root, args.expected_upstream_revision)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    params = load_config(args.config)
    blocks = args.block or ["O1", "O2", "O3"]
    scene_limit = args.scene_limit
    if args.cpu_smoke:
        blocks = ["O1"]
        scene_limit = 1
        params["data"]["batch_size"] = 1
        params["data"]["context_lengths"] = [2]
        params["data"]["use_bfloat16"] = False
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params["pretrain"]["folder"] = str(args.output_dir.resolve())
    params["pretrain"]["checkpoint"] = str(args.checkpoint.resolve())
    params["tasks_per_node"] = 1
    params["nodes"] = 1
    if args.losses_only:
        params["mode"] = "losses"

    dataset_view = make_dataset_view(args.dataset_root, blocks, scene_limit)
    active_dataset_root = Path(dataset_view.name) if dataset_view else args.dataset_root

    rendezvous_path = None
    initialized_here = False
    try:
        sys.path.insert(0, str(evaluation_root))
        import torch
        import torch.distributed as dist
        import evals.intuitive_physics.eval as upstream_eval
        import evals.intuitive_physics.utils as upstream_utils

        gpu = None
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            gpu = {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "capability": [properties.major, properties.minor],
            }
        provenance = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "project": git_state(Path(__file__).resolve().parents[1]),
            "upstream_revision": args.expected_upstream_revision,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "gpu": gpu,
            "dataset_root": str(args.dataset_root.resolve()),
            "blocks": blocks,
            "scene_limit": scene_limit,
            "workers": args.workers,
            "checkpoint_sha256": sha256_file(args.checkpoint),
        }
        (args.output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {"provenance": provenance, "parameters": params},
                sort_keys=False,
            )
        )

        cluster = upstream_utils.CLUSTER
        for block in blocks:
            upstream_utils.DATASET_PATHS_BY_CLUSTER[cluster][
                f"IntPhys-dev-{block}"
            ] = str((active_dataset_root / block).resolve())
        upstream_utils.PROPERTIES_BY_DATASET["intphys"] = blocks

        original_init_data = upstream_eval.init_data

        def local_init_data(*call_args, **call_kwargs):
            call_kwargs["num_workers"] = args.workers
            call_kwargs["pin_mem"] = torch.cuda.is_available()
            return original_init_data(*call_args, **call_kwargs)

        upstream_eval.init_data = local_init_data

        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("Runner requires an uninitialized or world-size-one process group")
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            rendezvous = tempfile.NamedTemporaryFile(
                prefix="intphys-dist-", delete=False
            )
            rendezvous.close()
            rendezvous_path = Path(rendezvous.name)
            rendezvous_path.unlink()
            dist.init_process_group(
                backend=backend,
                init_method=f"file://{rendezvous_path}",
                rank=0,
                world_size=1,
            )
            initialized_here = True

        upstream_eval.main(params)
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()
        if rendezvous_path:
            rendezvous_path.unlink(missing_ok=True)
        if dataset_view:
            dataset_view.cleanup()


if __name__ == "__main__":
    main()
