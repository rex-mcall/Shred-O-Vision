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
