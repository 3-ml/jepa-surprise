from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


VARIANTS = ("metal", "foam")


def scenario_matches(row: dict[str, str], scenario: str) -> bool:
    if scenario.lower() in {"all", "mass", "*"}:
        return row.get("scenario", "").startswith("Mass-")
    return row.get("scenario") == scenario


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_complete_pairs(
    rows: list[dict[str, str]],
    split: str,
    scenario: str,
    pair_count: int,
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["split"] == split and scenario_matches(row, scenario)]
    by_pair: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        by_pair.setdefault(row["pair_key"], []).append(row)

    out: list[dict[str, str]] = []
    for pair_key in sorted(by_pair):
        pair_rows = by_pair[pair_key]
        if len(pair_rows) != 2:
            continue
        out.extend(sorted(pair_rows, key=lambda row: (row["source_root"], row["trial_id"])))
        if pair_count > 0 and len({row["pair_key"] for row in out}) >= pair_count:
            break
    return out


def select_rows(
    rows: list[dict[str, str]],
    split: str,
    scenario: str,
    pair_count: int,
    include_readout: bool,
    readout_count: int,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if include_readout or split == "readout":
        readout_rows = sorted(
            [row for row in rows if row["split"] == "readout" and scenario_matches(row, scenario)],
            key=lambda row: (row["condition_dir"], row["trial_id"]),
        )
        if readout_count > 0:
            readout_rows = readout_rows[:readout_count]
        out.extend(readout_rows)
    if split in {"test", "all"}:
        out.extend(select_complete_pairs(rows, "test", scenario, pair_count))
    if split == "readout":
        return out
    if split not in {"test", "all"}:
        raise ValueError(f"Unsupported split: {split}")
    return out


def decode_mask(entry: dict[str, object]) -> np.ndarray:
    rle = {"size": entry["size"], "counts": str(entry["counts"]).encode("ascii")}
    return mask_utils.decode(rle).astype(bool)


def load_masks(path: Path) -> dict[int, dict[int, np.ndarray]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    masks: dict[int, dict[int, np.ndarray]] = {}
    for frame_key, entries in raw.items():
        frame_index = int(frame_key)
        masks[frame_index] = {}
        for entry in entries:
            masks[frame_index][int(entry["idx"])] = decode_mask(entry)
    return masks


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def moving_object_idx(masks: dict[int, dict[int, np.ndarray]]) -> int:
    by_idx: dict[int, list[tuple[int, float, float, float]]] = {}
    for frame_index, frame_masks in masks.items():
        for idx, mask in frame_masks.items():
            centroid = mask_centroid(mask)
            if centroid is None:
                continue
            by_idx.setdefault(idx, []).append((frame_index, centroid[0], centroid[1], float(mask.sum())))

    best_idx = None
    best_score = -1.0
    for idx, points in by_idx.items():
        if len(points) < 2:
            continue
        points = sorted(points)
        start = points[0]
        end = points[-1]
        displacement = ((end[1] - start[1]) ** 2 + (end[2] - start[2]) ** 2) ** 0.5
        median_area = float(np.median([p[3] for p in points]))
        # Prefer compact moving objects over large static background-ish masks.
        score = displacement / max(median_area**0.25, 1.0)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None:
        raise RuntimeError("Could not identify a moving object mask.")
    return best_idx


def load_static_metadata(path: Path) -> dict[str, object]:
    payload = pickle.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected pkl payload type for {path}: {type(payload)}")
    static = payload.get("static")
    if not isinstance(static, dict):
        static = payload.get("static0")
    if not isinstance(static, dict):
        raise RuntimeError(f"No static metadata dictionary found in {path}")
    return static


def scalar_metadata(value: object) -> object:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def metadata_object_id(static: dict[str, object], key: str) -> int | None:
    value = static.get(key)
    value = scalar_metadata(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mask_area_stats(masks: dict[int, dict[int, np.ndarray]], target_idx: int) -> dict[str, object]:
    areas = [int(frame_masks[target_idx].sum()) for frame_masks in masks.values() if target_idx in frame_masks]
    total_frames = len(masks)
    present_frames = len(areas)
    if not areas:
        return {
            "mask_total_frames": total_frames,
            "mask_present_frames": 0,
            "mask_presence_ratio": 0.0,
            "mask_area_min": 0,
            "mask_area_median": 0.0,
            "mask_area_max": 0,
        }
    return {
        "mask_total_frames": total_frames,
        "mask_present_frames": present_frames,
        "mask_presence_ratio": float(present_frames / max(total_frames, 1)),
        "mask_area_min": int(min(areas)),
        "mask_area_median": float(np.median(areas)),
        "mask_area_max": int(max(areas)),
    }


def normalized_shading(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lum = (
        0.2126 * rgb[:, :, 0].astype(np.float32)
        + 0.7152 * rgb[:, :, 1].astype(np.float32)
        + 0.0722 * rgb[:, :, 2].astype(np.float32)
    )
    vals = lum[mask]
    if vals.size == 0:
        return np.zeros_like(lum, dtype=np.float32)
    mean = float(vals.mean())
    std = float(vals.std() + 1e-6)
    return np.clip((lum - mean) / std, -2.0, 2.0)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def material_rgb(rgb: np.ndarray, mask: np.ndarray, variant: str, rng: np.random.Generator) -> np.ndarray:
    h, w = mask.shape
    shade = normalized_shading(rgb, mask)
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    box = bbox(mask)
    if box is None:
        return rgb
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    sx = max((x1 - x0) / 2.0, 1.0)
    sy = max((y1 - y0) / 2.0, 1.0)
    radial = np.exp(-(((x_grid - cx) / sx) ** 2 + ((y_grid - cy) / sy) ** 2))
    diagonal = ((x_grid - x0) / max(x1 - x0, 1) + (y_grid - y0) / max(y1 - y0, 1)) / 2.0

    if variant == "metal":
        base = np.array([180.0, 185.0, 190.0], dtype=np.float32)
        cool = np.array([18.0, 10.0, 0.0], dtype=np.float32)
        texture = 42.0 * shade[:, :, None] + 38.0 * radial[:, :, None] + 24.0 * diagonal[:, :, None]
        material = base + cool + texture
    elif variant == "foam":
        base = np.array([218.0, 218.0, 198.0], dtype=np.float32)
        speckle = rng.normal(0.0, 10.0, size=rgb.shape).astype(np.float32)
        pore = rng.random(mask.shape, dtype=np.float32)
        pore = (pore < 0.045).astype(np.float32)
        pore = cv2.GaussianBlur(pore, (3, 3), 0)[:, :, None] * -34.0
        material = base + 17.0 * shade[:, :, None] + speckle + pore
    else:
        raise ValueError(f"Unknown material variant: {variant}")

    alpha = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)[:, :, None]
    out = rgb.astype(np.float32) * (1.0 - alpha) + np.clip(material, 0, 255) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def transform_video(
    rgb_path: Path,
    id_json_path: Path,
    pkl_path: Path,
    output_path: Path,
    variant: str,
    seed: int,
    target_idx: int | None,
) -> dict[str, object]:
    masks = load_masks(id_json_path)
    static = load_static_metadata(pkl_path)
    metadata_target_id = metadata_object_id(static, "target_id")
    selected_target_idx = target_idx
    target_strategy = "explicit_idx"
    if selected_target_idx is None:
        selected_target_idx = metadata_target_id
        target_strategy = "metadata_target_id"
    if selected_target_idx is None:
        selected_target_idx = moving_object_idx(masks)
        target_strategy = "auto_moving_object"

    cap = cv2.VideoCapture(str(rgb_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {rgb_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video: {output_path}")

    rng = np.random.default_rng(seed)
    frame_count = 0
    changed_pixels = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = masks.get(frame_count, {}).get(selected_target_idx)
            if mask is not None:
                if mask.shape != rgb.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                rgb = material_rgb(rgb, mask, variant, rng)
                changed_pixels += int(mask.sum())
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    return {
        "target_idx": target_idx,
        "target_strategy": target_strategy,
        "metadata_target_id": metadata_target_id,
        "metadata_zone_id": metadata_object_id(static, "zone_id"),
        "metadata_probe_id": metadata_object_id(static, "probe_id"),
        "selected_target_idx": selected_target_idx,
        "frames": frame_count,
        "changed_mask_pixels": changed_pixels,
        "fps": fps,
        "width": width,
        "height": height,
        **mask_area_stats(masks, selected_target_idx),
    }


def copy_original_preview(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def make_contact_sheet(video_paths: list[tuple[str, Path]], output_path: Path) -> None:
    panels = []
    for label, path in video_paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = [0, max(count // 2, 0), max(count - 1, 0)]
        frames = []
        try:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                cv2.putText(frame, f"{label} t={idx}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
                cv2.putText(frame, f"{label} t={idx}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)
                frames.append(frame)
        finally:
            cap.release()
        if frames:
            panels.append(np.vstack(frames))
    if not panels:
        raise RuntimeError("No frames available for contact sheet.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.hstack(panels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate material-appearance counterfactuals for Physion++ Mass.")
    parser.add_argument("--manifest", default="data/physionpp/mass_only/manifest_mass.csv")
    parser.add_argument("--root", default="data/physionpp/mass_only")
    parser.add_argument("--output-root", default="data/physionpp/mass_material_pilot")
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenario", default="Mass-Collision")
    parser.add_argument("--pair-count", type=int, default=4, help="Number of complete test pairs; <=0 selects all.")
    parser.add_argument("--include-readout", action="store_true", help="Also transform readout-fitting rows.")
    parser.add_argument("--readout-count", type=int, default=0, help="Number of readout rows; <=0 selects all.")
    parser.add_argument("--preview-count", type=int, default=1)
    parser.add_argument("--target-idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.output_root)
    rows = select_rows(
        read_manifest(Path(args.manifest)),
        args.split,
        args.scenario,
        args.pair_count,
        args.include_readout,
        args.readout_count,
    )
    if not rows:
        raise RuntimeError("No rows selected.")

    manifest_rows = []
    selected_test_pairs = sorted({row["pair_key"] for row in rows if row["split"] == "test"})
    summary = {
        "source_manifest": args.manifest,
        "source_root": args.root,
        "output_root": str(output_root),
        "split": args.split,
        "scenario": args.scenario,
        "pair_count": args.pair_count,
        "include_readout": args.include_readout,
        "readout_count": args.readout_count,
        "selected_videos": len(rows),
        "selected_test_pairs": len(selected_test_pairs),
        "selected_by_split": {
            split_name: sum(1 for row in rows if row["split"] == split_name)
            for split_name in sorted({row["split"] for row in rows})
        },
        "variants": list(VARIANTS),
        "videos": [],
    }

    for row_i, row in enumerate(rows):
        rgb_path = root / row["rgb_mp4"]
        id_json_path = root / row["id_json"]
        pkl_path = root / row["pkl"]
        for required in (rgb_path, id_json_path, pkl_path):
            if not required.exists():
                raise FileNotFoundError(required)
        source_stem = Path(row["rgb_mp4"]).with_suffix("").as_posix().replace("/", "__").replace("_img", "")
        variant_outputs = []
        for variant_i, variant in enumerate(VARIANTS):
            relative_output = Path(variant) / f"{source_stem}_{variant}_img.mp4"
            output_path = output_root / relative_output
            stats = transform_video(
                rgb_path,
                id_json_path,
                pkl_path,
                output_path,
                variant,
                seed=args.seed + row_i * 1009 + variant_i * 917,
                target_idx=args.target_idx,
            )
            out_row = dict(row)
            out_row["material_variant"] = variant
            out_row["material_target"] = str(stats["target_strategy"])
            out_row["material_target_idx"] = str(stats["selected_target_idx"])
            out_row["material_metadata_target_id"] = str(stats["metadata_target_id"])
            out_row["material_mask_present_frames"] = str(stats["mask_present_frames"])
            out_row["material_mask_total_frames"] = str(stats["mask_total_frames"])
            out_row["material_mask_presence_ratio"] = f"{stats['mask_presence_ratio']:.6f}"
            out_row["material_changed_mask_pixels"] = str(stats["changed_mask_pixels"])
            out_row["source_rgb_mp4"] = row["rgb_mp4"]
            out_row["source_id_json"] = row["id_json"]
            out_row["source_id_mp4"] = row.get("id_mp4", "")
            out_row["source_map_png"] = row.get("map_png", "")
            out_row["source_pkl"] = row["pkl"]
            out_row["rgb_mp4"] = relative_output.as_posix()
            manifest_rows.append(out_row)
            variant_outputs.append((variant, output_path))
            summary["videos"].append(
                {
                    "source_rgb_mp4": row["rgb_mp4"],
                    "output_rgb_mp4": relative_output.as_posix(),
                    "variant": variant,
                    "label": row["label"],
                    "pair_key": row["pair_key"],
                    "split": row["split"],
                    **stats,
                }
            )

        if row_i < args.preview_count:
            original_preview = output_root / "previews" / f"{source_stem}_original_img.mp4"
            copy_original_preview(rgb_path, original_preview)
            make_contact_sheet(
                [("original", original_preview), *variant_outputs],
                output_root / "previews" / f"{source_stem}_material_contact_sheet.png",
            )

    write_manifest(output_root / "manifest_material_mass_collision.csv", manifest_rows)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
