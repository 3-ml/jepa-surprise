import numpy as np
import torch

from scripts.run_lewm_prediction_error_audit import (
    Condition,
    apply_condition_actions,
    apply_condition_pixels,
    conditions,
    normalize_action_blocks,
    select_clip_indices,
    state_metrics,
    visual_difference,
)


def test_conditions_are_unique_and_keep_appearance_scopes_separate():
    rows = conditions()
    assert len(rows) == 38
    assert len({row.name for row in rows}) == len(rows)
    assert {row.scope for row in rows if row.family == "hue"} == {
        "target",
        "persistent",
    }


def test_action_interventions_change_only_final_target_leading_block():
    actions = torch.arange(40, dtype=torch.float32).reshape(4, 10)
    zero = apply_condition_actions(
        actions, Condition("zero", "action", "final_block", 0, "scale_final")
    )
    opposite = apply_condition_actions(
        actions,
        Condition("opposite", "action", "final_block", -1, "scale_final"),
    )
    assert torch.equal(zero[:2], actions[:2])
    assert torch.equal(zero[3:], actions[3:])
    assert torch.equal(zero[2], torch.zeros(10))
    assert torch.equal(opposite[2], -actions[2])
    assert torch.equal(actions, torch.arange(40, dtype=torch.float32).reshape(4, 10))


def test_action_block_normalization_repeats_stats_and_zeroes_boundary_nans():
    actions = torch.tensor([[1.0, 4.0, 3.0, 8.0], [float("nan"), 2.0, 1.0, 4.0]])
    normalized = normalize_action_blocks(
        actions, torch.tensor([1.0, 2.0]), torch.tensor([2.0, 2.0])
    )
    assert torch.equal(
        normalized,
        torch.tensor([[0.0, 1.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]]),
    )


def test_visual_difference_separates_extent_from_changed_pixel_magnitude():
    reference = torch.zeros((3, 2, 2), dtype=torch.uint8)
    changed = reference.clone()
    changed[:, 0, 0] = 255
    metrics = visual_difference(changed, reference)
    assert metrics["rgb_l1_01"] == 0.25
    assert metrics["changed_pixel_fraction"] == 0.25
    assert metrics["changed_pixel_rgb_l1_01"] == 1.0


def test_zero_visual_interventions_are_bitwise_identity():
    frames = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
    assert torch.equal(
        frames,
        apply_condition_pixels(frames, Condition("hue", "hue", "target", 0)),
    )
    assert torch.equal(
        frames,
        apply_condition_pixels(
            frames, Condition("brightness", "brightness", "persistent", 1)
        ),
    )


def test_target_scope_leaves_context_unchanged_and_persistent_changes_it():
    frames = torch.zeros((4, 3, 8, 8), dtype=torch.uint8)
    frames[:, 0] = 255
    target = apply_condition_pixels(frames, Condition("target", "hue", "target", 120))
    persistent = apply_condition_pixels(
        frames, Condition("persistent", "hue", "persistent", 120)
    )
    assert torch.equal(target[:-1], frames[:-1])
    assert not torch.equal(target[-1], frames[-1])
    assert not torch.equal(persistent[0], frames[0])


def test_clip_selection_uses_one_clip_per_episode_deterministically():
    class Dataset:
        clip_indices = [(episode, start) for episode in range(10) for start in range(3)]

    first = select_clip_indices(Dataset(), 5, "seed", excluded_episodes={0, 1})
    second = select_clip_indices(Dataset(), 5, "seed", excluded_episodes={0, 1})
    assert first == second
    episodes = [Dataset.clip_indices[index][0] for index in first]
    assert len(episodes) == len(set(episodes)) == 5
    assert not set(episodes) & {0, 1}


def test_state_metrics_report_task_specific_displacement():
    tworoom = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [15.0, 2.0]])
    assert state_metrics("tworoom", tworoom)["agent_displacement_diameters"] == 1.0

    pusht = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, np.pi - 0.1],
            [30.0, 0.0, 4.0, 5.0, -np.pi + 0.1],
        ]
    )
    metrics = state_metrics("pusht", pusht)
    assert metrics["agent_displacement_diameters"] == 1.0
    assert np.isclose(metrics["block_displacement_state"], 5.0)
    assert np.isclose(metrics["block_angle_change_radians"], 0.2, atol=1e-5)
