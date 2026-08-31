import numpy as np

from blueraven_visualizer.events import first_true_time, nearest, detect_shred, detect_tumble_onset


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


def test_detect_tumble_onset():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    gyro_mag = np.array([10.0, 20.0, 30.0, 900.0, 40.0])
    tSpin, wPeak, idx = detect_tumble_onset(t, gyro_mag)
    assert idx == 3
    assert tSpin == 3.0
    assert wPeak == 900.0


def test_detect_tumble_onset_distinct_from_shred():
    """Tumble onset and shred can land on different frames - the whole
    reason to report them separately."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    accel_mag = np.array([1.0, 300.0, 2.0, 1.0])
    gyro_mag = np.array([5.0, 10.0, 1500.0, 8.0])
    tShred, _, _ = detect_shred(t, accel_mag)
    tSpin, _, _ = detect_tumble_onset(t, gyro_mag)
    assert tShred == 1.0
    assert tSpin == 2.0
