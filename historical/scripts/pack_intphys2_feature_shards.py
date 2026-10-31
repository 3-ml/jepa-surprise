from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

try:
    import intphys2_confirmatory_common as common
except ModuleNotFoundError:
    from scripts import intphys2_confirmatory_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_command() -> str:
    return " ".join([sys.executable, *sys.argv])


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample_id values in feature index")
    return rows


def metadata_columns(rows: list[dict[str, str]]) -> list[str]:
    skip = {"shard", "shard_bytes", "shard_sha256"}
    columns = [column for column in rows[0] if column not in skip]
    if "pair_label" not in columns:
        insert_at = columns.index("physical_label") + 1
        columns.insert(insert_at, "pair_label")
    return columns


def layer_names(native_layers: np.ndarray) -> np.ndarray:
    names = []
    for layer in native_layers.tolist():
        layer = int(layer)
        names.append("final" if layer == 40 else f"block_{layer:02d}")
    return np.asarray(names)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_index(args.index)
    root = args.root.resolve()
    pooled_arrays = []
    reference_layers = None
    reference_depths = None
    for row in rows:
        shard_path = (root / row["shard"]).resolve()
        if common.sha256_file(shard_path) != row["shard_sha256"]:
            raise ValueError(f"Shard SHA-256 mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as payload:
            pooled = payload["pooled"]
            native_layers = payload["native_layers"]
            relative_depths = payload["relative_depths"]
            metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata["sample_id"] != row["sample_id"]:
            raise ValueError(f"Shard sample_id mismatch: {shard_path}")
        if not np.isfinite(pooled).all():
            raise ValueError(f"Non-finite pooled features: {shard_path}")
        if reference_layers is None:
            reference_layers = native_layers.copy()
            reference_depths = relative_depths.copy()
        elif not np.array_equal(reference_layers, native_layers) or not np.array_equal(
            reference_depths, relative_depths
        ):
            raise ValueError(f"Layer/depth mismatch: {shard_path}")
        pooled_arrays.append(pooled.astype(np.float16, copy=False))

    if reference_layers is None or reference_depths is None:
        raise ValueError("No feature shards were packed")
    x = np.stack(pooled_arrays, axis=0)
    columns = metadata_columns(rows)
    arrays: dict[str, np.ndarray] = {
        "X": x,
        "layer_names": layer_names(reference_layers),
        "relative_depths": reference_depths.astype(np.float32),
        "native_layers": reference_layers.astype(np.int64),
        "metadata_columns": np.asarray(columns),
    }
    for column in columns:
        if column == "pair_label":
            values = [row["physical_label"] for row in rows]
        else:
            values = [row.get(column, "") for row in rows]
        arrays[column] = np.asarray(values)

    output_npz = output_dir / "representations.npz"
    representations_sha256 = common.write_atomic_npz(output_npz, arrays)
    samples_path = output_dir / "samples.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            record = {column: row.get(column, "") for column in columns}
            record["pair_label"] = row["physical_label"]
            writer.writerow(record)
    provenance = {
        "schema_version": 1,
        "status": "complete",
        "command": exact_command(),
        "model_label": args.model_label,
        "source_index": str(args.index.resolve()),
        "source_index_sha256": common.sha256_file(args.index.resolve()),
        "source_root": str(root),
        "representations": str(output_npz.resolve()),
        "representations_sha256": representations_sha256,
        "samples": int(x.shape[0]),
        "pairs": len({row["pair_id"] for row in rows}),
        "videos": len({row["name"] for row in rows}),
        "layers": int(x.shape[1]),
        "feature_dim": int(x.shape[2]),
        "dtype": str(x.dtype),
        "metadata_columns": columns,
        "input_index_head_sha256": sha256_text(json.dumps(rows[:5], sort_keys=True)),
        "claim_boundary": "Packing only; no probe, geometry, or paper claim is evaluated here.",
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "samples": int(x.shape[0]),
                "pairs": len({row["pair_id"] for row in rows}),
                "layers": int(x.shape[1]),
                "representations_sha256": representations_sha256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
