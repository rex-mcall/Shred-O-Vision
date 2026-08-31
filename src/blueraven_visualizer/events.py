"""Flight-event detection helpers shared by both rendering backends."""

import numpy as np


def first_true_time(t, flag):
    """Time of the first sample where flag (a 0/1 or bool array) goes true."""
    i = np.argmax(flag) if flag.any() else None
    return float(t[i]) if i is not None and flag[i] else np.nan


def nearest(t, tq):
    """Index into t nearest to query time tq."""
    return int(np.argmin(np.abs(t - tq)))


def detect_shred(t_hr, accel_mag):
    """Structural-failure ("shred") instant: the peak IMU acceleration magnitude.
    Returns (time, peak_g, index)."""
    i = int(np.argmax(accel_mag))
    return float(t_hr[i]), float(accel_mag[i]), i


def detect_tumble_onset(t_hr, gyro_mag):
    """Tumble onset: the peak angular rate. Distinct from detect_shred - a
    rocket can start tumbling slightly before or after peak acceleration.
    Returns (time, peak_deg_per_s, index)."""
    i = int(np.argmax(gyro_mag))
    return float(t_hr[i]), float(gyro_mag[i]), i
