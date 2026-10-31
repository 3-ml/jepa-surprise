import numpy as np

from scripts.run_lewm_pusht_controlled_displacement import (
    ACTION_SCALES,
    HUE_SCALE_MATCHES,
    angle_difference,
    scale_token,
)


def test_pusht_action_scale_grid_is_locked():
    assert ACTION_SCALES == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert HUE_SCALE_MATCHES == {30.0: 0.25, 60.0: 0.75, 120.0: 1.0, 180.0: 1.0}
    assert scale_token(0.75) == "0p75"


def test_angle_difference_wraps_at_two_pi():
    assert np.isclose(angle_difference(np.pi - 0.1, -np.pi + 0.1), 0.2)
