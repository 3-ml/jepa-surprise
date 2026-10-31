import numpy as np

from scripts.run_lewm_pusht_role_color_audit import INTERVENTIONS, select_manifest


def test_role_colors_have_matched_novel_rgb_controls():
    colors = {row.name: np.asarray(row.rgb, dtype=np.int16) for row in INTERVENTIONS}
    block_default = np.asarray([119, 136, 153])
    agent_default = np.asarray([65, 105, 225])
    block_role_distance = np.abs(colors["block_blue"] - block_default).sum()
    block_novel_distance = np.abs(colors["block_orange_matched"] - block_default).sum()
    agent_role_distance = np.abs(colors["agent_gray"] - agent_default).sum()
    agent_novel_distance = np.abs(colors["agent_purple_matched"] - agent_default).sum()

    assert abs(block_role_distance - block_novel_distance) <= 10
    assert agent_role_distance == agent_novel_distance


def test_manifest_selection_is_deterministic_and_without_replacement(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text(
        "base_id,dataset_index,episode,start_step\n"
        + "".join(f"pusht:{i}:0,{i},{i},0\n" for i in range(10))
    )
    first = select_manifest(path, 5, "seed")
    second = select_manifest(path, 5, "seed")
    assert first == second
    assert len({row["base_id"] for row in first}) == 5
