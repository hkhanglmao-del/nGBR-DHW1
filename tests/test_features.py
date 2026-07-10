import numpy as np

from coral_bleaching_pipeline.features import compute_alert_level


def test_extended_alert_levels():
    dhw = np.array([0.0, 0.0, 2.0, 5.0, 9.0, 13.0, 17.0, 21.0])
    hotspot = np.array([-0.1, 0.5, 1.1, 1.2, 1.2, 1.2, 1.2, 1.2])
    assert compute_alert_level(dhw, hotspot).tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_legacy_alert_caps_extreme_dhw_at_level_2():
    dhw = np.array([13.0, 21.0])
    hotspot = np.array([1.2, 1.2])
    assert compute_alert_level(dhw, hotspot, scheme="legacy_baa").tolist() == [4, 4]

