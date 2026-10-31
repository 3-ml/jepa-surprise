"""Proposed Vasi V2 manifest contract and score-blind table validator.

This is tooling for discussion/development only. An accepted table does not
validate the physical labels themselves; simulator/contact audits remain needed.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

HASH = re.compile(r"^[0-9a-f]{64}$")
BASE_COLUMNS = {
    "scene_id", "base_trajectory_id", "split", "physical_condition",
    "appearance_condition", "appearance_dose", "appearance_dose_unit",
    "clip_id", "scene_seed", "event_frame_0based", "source_frame_indices_json",
    "source_frame_paths_json", "source_frame_sha256_json", "state_trace_sha256", "state_sha",
    "context_input_sha256", "context_sha", "target_input_sha256", "checkpoint_sha256",
    "score_protocol_id", "output_status", "score", "tubelet_scores_json",
    "validity_verdict", "score_blind_exclusion_reason", "target_visibility_min_fraction",
    "target_area_min_fraction", "target_rgb_l1", "changed_pixel_fraction",
    "flow_summary_json",
}
PHYSICS = {"lawful", "inconsistent"}
APPEARANCES = {"clean", "persistent_hue", "abrupt_event"}
SPLITS = {"development", "confirmatory"}
STATUSES = {"complete", "failed_render", "failed_score", "excluded_score_blind"}


def validate(rows, *, expected_appearances=APPEARANCES, expected_scenes=None):
    """Return audit errors. Failed cells stay present and missing, never imputed as zero.

    Enforces a complete scene x physical branch x requested-appearance grid;
    scene and base-trajectory split separation; paired context per appearance;
    same simulator state over appearance variants within a physical branch; unique
    clip IDs; valid hashes; and explicit score/missingness status.
    """
    errors = []
    if not rows:
        return ["empty table"]
    missing_columns = sorted(BASE_COLUMNS - set(rows[0]))
    if missing_columns:
        return [f"missing column {name}" for name in missing_columns]

    scenes = defaultdict(list)
    ids, cells = set(), set()
    scene_splits = defaultdict(set)
    trajectory_splits = defaultdict(set)
    contexts = defaultdict(set)
    states = defaultdict(set)
    trajectories = defaultdict(set)
    for row in rows:
        scene = row["scene_id"]
        physical = row["physical_condition"]
        appearance = row["appearance_condition"]
        clip = row["clip_id"]
        scenes[scene].append(row)
        if not scene or not row["base_trajectory_id"]:
            errors.append(f"blank scene/trajectory id in {clip!r}")
        if clip in ids:
            errors.append(f"duplicate clip_id {clip}")
        ids.add(clip)
        if row["split"] not in SPLITS:
            errors.append(f"bad split {row['split']!r} in {clip}")
        if physical not in PHYSICS:
            errors.append(f"bad physical_condition {physical!r} in {clip}")
        if appearance not in expected_appearances:
            errors.append(f"unexpected appearance_condition {appearance!r} in {clip}")
        cell = (scene, physical, appearance)
        if cell in cells:
            errors.append(f"duplicate cell {cell}")
        cells.add(cell)
        scene_splits[scene].add(row["split"])
        trajectory_splits[row["base_trajectory_id"]].add(row["split"])
        trajectories[scene].add(row["base_trajectory_id"])
        contexts[(scene, appearance)].add(row["context_input_sha256"])
        states[(scene, physical)].add(row["state_trace_sha256"])
        for name in ("state_trace_sha256", "context_input_sha256", "target_input_sha256", "checkpoint_sha256"):
            if not HASH.fullmatch(str(row[name])):
                errors.append(f"invalid {name} in {clip}")
        if row["context_sha"] != row["context_input_sha256"]:
            errors.append(f"context_sha alias mismatch in {clip}")
        if row["state_sha"] != row["state_trace_sha256"]:
            errors.append(f"state_sha alias mismatch in {clip}")
        status = row["output_status"]
        if status not in STATUSES:
            errors.append(f"bad output_status {status!r} in {clip}")
        score = row["score"]
        if status == "complete":
            try:
                numeric = float(score)
            except (TypeError, ValueError):
                errors.append(f"complete cell lacks numeric score in {clip}")
            else:
                if not math.isfinite(numeric):
                    errors.append(f"nonfinite score in {clip}")
        elif score not in (None, ""):
            errors.append(f"failed/excluded cell must have blank score in {clip}")
        if status == "excluded_score_blind" and not row["score_blind_exclusion_reason"]:
            errors.append(f"score-blind exclusion lacks reason in {clip}")
        if status != "excluded_score_blind" and row["score_blind_exclusion_reason"]:
            errors.append(f"exclusion reason present for non-excluded cell in {clip}")

    for scene, values in scene_splits.items():
        if len(values) != 1:
            errors.append(f"scene {scene} appears in several splits {sorted(values)}")
    for trajectory, values in trajectory_splits.items():
        if len(values) != 1:
            errors.append(f"trajectory {trajectory} appears in several splits {sorted(values)}")
    for scene, values in trajectories.items():
        if len(values) != 1:
            errors.append(f"scene {scene} has several base_trajectory_id values")
    for key, values in contexts.items():
        if len(values) != 1:
            errors.append(f"lawful/inconsistent context mismatch {key}")
    for key, values in states.items():
        if len(values) != 1:
            errors.append(f"persistent/abrupt appearance changed simulator state {key}")

    required = set(expected_scenes) if expected_scenes is not None else set(scenes)
    expected_cells = {
        (scene, physical, appearance)
        for scene in required for physical in PHYSICS for appearance in expected_appearances
    }
    for cell in sorted(expected_cells - cells):
        errors.append(f"missing manifest cell {cell}; record explicit missing status")
    for cell in sorted(cells - expected_cells):
        errors.append(f"cell outside frozen grid {cell}")
    return errors
