import numpy as np
import torch

from scripts.run_vjepa2_section_surprise import (
    fixed_segment_bounds,
    sample_indices,
    section_bounds,
    target_frame_ranges,
)


def test_section_bounds_cover_frame_range_without_overlap():
    assert section_bounds(100, 5) == [
        (0, 20),
        (20, 40),
        (40, 60),
        (60, 80),
        (80, 100),
    ]


def test_fixed_segment_bounds_keep_only_complete_segments():
    assert fixed_segment_bounds(150, 64) == [(0, 64), (64, 128)]


def test_sample_indices_keep_section_endpoints():
    indices = sample_indices(20, 40, 5)
    assert indices.tolist() == [20, 25, 30, 34, 39]


def test_target_frame_ranges_map_tubelet_ids_to_sampled_frames():
    sampled = np.arange(0, 64)
    time_ids = torch.tensor([24, 25, 26])
    ranges = target_frame_ranges(sampled, tubelet_size=2, time_ids=time_ids)
    assert ranges == [[48, 49], [50, 51], [52, 53]]
