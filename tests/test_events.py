import numpy as np

from blueraven_visualizer.events import first_true_time, nearest, detect_shred


def test_first_true_time_found():
    t = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    flag = np.array([0, 0, 1, 1, 0])
    assert first_true_time(t, flag) == 0.2


def test_first_true_time_never_true():
    t = np.array([0.0, 0.1, 0.2])
    flag = np.array([0, 0, 0])
    assert np.isnan(first_true_time(t, flag))


def test_nearest():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    assert nearest(t, 1.9) == 2
    assert nearest(t, -5.0) == 0
    assert nearest(t, 100.0) == 3


def test_detect_shred():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    accel_mag = np.array([1.0, 2.0, 50.0, 3.0, 1.0])
    tShred, gPk, idx = detect_shred(t, accel_mag)
    assert idx == 2
    assert tShred == 2.0
    assert gPk == 50.0
