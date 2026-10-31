from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


MASS_READOUT_ROOTS = {
    "mass_collision_pp": "Mass-Collision",
    "mass_dominoes_pp": "Mass-Dominoes",
    "mass_waterpush_pp": "Mass-Waterpush",
}

MASS_TEST_ROOTS = {
    "mass_collision_pp-copy0": "Mass-Collision",
    "mass_collision_pp-copy1": "Mass-Collision",
    "mass_dominoes_pp-copy0": "Mass-Dominoes",
    "mass_dominoes_pp-copy1": "Mass-Dominoes",
    "mass_waterpush_pp-copy0": "Mass-Waterpush",
    "mass_waterpush_pp-copy1": "Mass-Waterpush",
}


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def infer_label_from_json(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    # Physion++ stores rich per-frame metadata. Keep this deliberately broad
    # until we lock the exact official OCP field after inspection.
    candidates = [
        "label",
        "target_contacting_zone",
        "target_contacting_zone_bool",
        "will_collide",
        "will_contact",
        "contact",
        "hit",
        "success",
    ]
    for key in candidates:
        if key in payload:
            return str(payload[key])
    return ""


def label_from_metadata(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return "1"
        if lowered == "false":
            return "0"
    return "" if value is None else str(value)


def scenario_from_source_root(source_root: str) -> str:
    clean = re.sub(r"-copy[01]$", "", source_root)
    mapping = MASS_READOUT_ROOTS | {
        key.removesuffix("-copy0").removesuffix("-copy1"): value
        for key, value in MASS_TEST_ROOTS.items()
    }
    return mapping.get(clean, clean)


def parse_official_test_filename(filename: str) -> tuple[str, str, str]:
    match = re.match(r"(?P<root>.+?_pp-copy[01])_(?P<condition>.+)_(?P<trial>\d{4})_img\.mp4$", filename)
    if not match:
        raise ValueError(f"Unexpected Physion++ Mass filename: {filename}")
    return match.group("root"), match.group("condition"), match.group("trial")


def normalized_test_pair_key(source_root: str, condition: str, trial_id: str) -> str:
    normalized_root = re.sub(r"-copy[01]$", "", source_root)
    return f"{normalized_root}/{condition}/{trial_id}"


def build_readout_rows(root: Path, split_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dirname, scenario in MASS_READOUT_ROOTS.items():
        scenario_root = split_root / dirname
        if not scenario_root.exists():
            continue
        for metadata_path in sorted(scenario_root.rglob("metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = []
            if not isinstance(metadata, list):
                continue
            trial_dir = metadata_path.parent
            for item in metadata:
                if not isinstance(item, dict):
                    continue
                stimulus_name = item.get("stimulus_name")
                if not stimulus_name or not item.get("is_trial_valid", False):
                    continue
                trial_id = stimulus_name.rsplit("_", 1)[-1]
                rgb_path = trial_dir / f"{trial_id}_img.mp4"
                if not rgb_path.exists():
                    continue
                id_mp4 = trial_dir / f"{trial_id}_id.mp4"
                id_json = trial_dir / f"{trial_id}_id.json"
                map_png = trial_dir / f"{trial_id}_map.png"
                pkl_path = trial_dir / f"{trial_id}.pkl"
                rows.append(
                    {
                        "split": "readout",
                        "scenario": scenario,
                        "source_root": dirname,
                        "condition_dir": rel(trial_dir, root),
                        "trial_id": trial_id,
                        "pair_key": "",
                        "label": label_from_metadata(item.get("does_target_contact_zone")),
                        "start_frame_for_prediction": "",
                        "rgb_mp4": rel(rgb_path, root),
                        "id_mp4": rel(id_mp4, root) if id_mp4.exists() else "",
                        "id_json": rel(id_json, root) if id_json.exists() else "",
                        "map_png": rel(map_png, root) if map_png.exists() else "",
                        "pkl": rel(pkl_path, root) if pkl_path.exists() else "",
                    }
                )
    return rows


def build_test_rows_from_csv(root: Path, split_root: Path, csv_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_row in csv_rows:
        source_root, condition, trial_id = parse_official_test_filename(csv_row["filenames"])
        scenario = scenario_from_source_root(source_root)
        trial_dir = split_root / source_root / condition
        rgb_path = trial_dir / f"{trial_id}_img.mp4"
        id_mp4 = trial_dir / f"{trial_id}_id.mp4"
        id_json = trial_dir / f"{trial_id}_id.json"
        map_png = trial_dir / f"{trial_id}_map.png"
        pkl_path = trial_dir / f"{trial_id}.pkl"
        rows.append(
            {
                "split": "test",
                "scenario": scenario,
                "source_root": source_root,
                "condition_dir": rel(trial_dir, root),
                "trial_id": trial_id,
                "pair_key": normalized_test_pair_key(source_root, condition, trial_id),
                "label": label_from_metadata(csv_row["target_hit_zone_labels"]),
                "start_frame_for_prediction": csv_row.get("start_frame_for_prediction", ""),
                "rgb_mp4": rel(rgb_path, root) if rgb_path.exists() else "",
                "id_mp4": rel(id_mp4, root) if id_mp4.exists() else "",
                "id_json": rel(id_json, root) if id_json.exists() else "",
                "map_png": rel(map_png, root) if map_png.exists() else "",
                "pkl": rel(pkl_path, root) if pkl_path.exists() else "",
                **{f"official_csv_{key}": value for key, value in csv_row.items()},
            }
        )
    return rows


def build_file_rows(root: Path, split_root: Path, split: str, roots: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dirname, scenario in roots.items():
        scenario_root = split_root / dirname
        if not scenario_root.exists():
            continue
        for rgb_path in sorted(scenario_root.rglob("*_img.mp4")):
            trial_stem = rgb_path.name.removesuffix("_img.mp4")
            trial_dir = rgb_path.parent
            id_mp4 = trial_dir / f"{trial_stem}_id.mp4"
            id_json = trial_dir / f"{trial_stem}_id.json"
            map_png = trial_dir / f"{trial_stem}_map.png"
            pkl_path = trial_dir / f"{trial_stem}.pkl"
            rows.append(
                {
                    "split": split,
                    "scenario": scenario,
                    "source_root": dirname,
                    "condition_dir": rel(trial_dir, root),
                    "trial_id": trial_stem,
                    "pair_key": rel(trial_dir, split_root),
                    "label": infer_label_from_json(id_json),
                    "start_frame_for_prediction": "",
                    "rgb_mp4": rel(rgb_path, root),
                    "id_mp4": rel(id_mp4, root) if id_mp4.exists() else "",
                    "id_json": rel(id_json, root) if id_json.exists() else "",
                    "map_png": rel(map_png, root) if map_png.exists() else "",
                    "pkl": rel(pkl_path, root) if pkl_path.exists() else "",
                }
            )
    return rows


def add_test_csv_columns(rows: list[dict[str, str]], csv_rows: list[dict[str, str]]) -> list[str]:
    csv_fields = sorted({key for row in csv_rows for key in row.keys()})
    for row in rows:
        for field in csv_fields:
            row.setdefault(f"official_csv_{field}", "")
    return [f"official_csv_{field}" for field in csv_fields]


def write_rows(path: Path, rows: list[dict[str, str]], extra_fields: list[str] | None = None) -> None:
    base_fields = [
        "split",
        "scenario",
        "source_root",
        "condition_dir",
        "trial_id",
        "pair_key",
        "label",
        "start_frame_for_prediction",
        "rgb_mp4",
        "id_mp4",
        "id_json",
        "map_png",
        "pkl",
    ]
    fields = base_fields + (extra_fields or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_test_pairs(path: Path, rows: list[dict[str, str]]) -> None:
    pairs: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["split"] == "test":
            pairs.setdefault(row["pair_key"], []).append(row)

    pair_rows = []
    for pair_id, (first, second) in enumerate(sorted((key, value) for key, value in pairs.items())):
        members = sorted(second, key=lambda row: row["source_root"])
        pair_rows.append(
            {
                "pair_id": f"mass_pair_{pair_id:03d}",
                "pair_key": first,
                "scenario": members[0]["scenario"],
                "n_trials": len(members),
                "labels": ";".join(row["label"] for row in members),
                "trial_paths": ";".join(row["rgb_mp4"] for row in members),
            }
        )

    fields = ["pair_id", "pair_key", "scenario", "n_trials", "labels", "trial_paths"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_rows)


def write_debug_subset(path: Path, rows: list[dict[str, str]], n_pairs: int = 20) -> None:
    selected_pair_keys = []
    for row in rows:
        if row["split"] != "test":
            continue
        if row["pair_key"] not in selected_pair_keys:
            selected_pair_keys.append(row["pair_key"])
        if len(selected_pair_keys) >= n_pairs:
            break
    selected = set(selected_pair_keys)
    subset = [row for row in rows if row["split"] == "readout" or row["pair_key"] in selected]
    extra_fields = [field for field in rows[0].keys() if field.startswith("official_csv_")]
    write_rows(path, subset, extra_fields)


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    by_split: dict[str, int] = {}
    by_scenario: dict[str, int] = {}
    labels: dict[str, int] = {}
    pair_keys = set()
    for row in rows:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
        by_scenario[row["scenario"]] = by_scenario.get(row["scenario"], 0) + 1
        labels[row["label"]] = labels.get(row["label"], 0) + 1
        pair_keys.add((row["split"], row["pair_key"]))
    return {
        "rows": len(rows),
        "by_split": by_split,
        "by_scenario": by_scenario,
        "labels": labels,
        "unique_split_pair_keys": len(pair_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/physionpp/mass_only")
    parser.add_argument("--output", default="data/physionpp/mass_only/manifest_mass.csv")
    args = parser.parse_args()

    root = Path(args.root)
    readout_root = root / "readout_data_v1"
    test_root = root / "testdata_v1"

    mass_csv = test_root / "physionpp-mass_merge_221111.csv"
    csv_rows = read_csv_rows(mass_csv)
    rows = []
    rows.extend(build_readout_rows(root, readout_root))
    rows.extend(build_test_rows_from_csv(root, test_root, csv_rows))
    extra_fields = add_test_csv_columns(rows, csv_rows)

    output = Path(args.output)
    write_rows(output, rows, extra_fields)

    summary_path = output.with_suffix(".summary.json")
    summary = summarize(rows)
    summary["official_mass_csv_rows"] = len(csv_rows)
    summary["official_mass_csv_fields"] = sorted({key for row in csv_rows for key in row.keys()})
    write_test_pairs(output.with_name("manifest_mass_test_pairs.csv"), rows)
    write_debug_subset(output.with_name("manifest_mass_debug20.csv"), rows, n_pairs=20)
    summary["outputs"] = [
        output.as_posix(),
        output.with_name("manifest_mass_test_pairs.csv").as_posix(),
        output.with_name("manifest_mass_debug20.csv").as_posix(),
        summary_path.as_posix(),
    ]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
