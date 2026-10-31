from scripts.run_lewm_tworoom_controlled_displacement import (
    ACTION_SCALES,
    HUE_SCALE_MATCHES,
    candidate_manifest,
    scale_token,
)


def test_action_scale_grid_and_tokens_are_locked():
    assert ACTION_SCALES == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert HUE_SCALE_MATCHES == {30.0: 0.25, 60.0: 0.5, 120.0: 0.75, 180.0: 1.0}
    assert scale_token(0.25) == "0p25"


def test_candidate_manifest_permutation_is_deterministic(tmp_path):
    path = tmp_path / "pool.csv"
    path.write_text(
        "base_id,dataset_index,episode,start_step\n"
        + "".join(f"tworoom:{i}:0,{i},{i},0\n" for i in range(10))
    )
    first = candidate_manifest(path, "seed")
    second = candidate_manifest(path, "seed")
    assert first == second
    assert {row["base_id"] for row in first} == {f"tworoom:{i}:0" for i in range(10)}
