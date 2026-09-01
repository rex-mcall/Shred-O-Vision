"""Flight-event detection helpers shared by both rendering backends.

These are deliberately neutral about *why* a peak happened. On a nominal
flight the peak acceleration is just max thrust off the pad, and the peak
angular rate is usually spin-up or deployment; on a failure it's the
structural break ("shred"). The tool reports the instants either way and
leaves the interpretation to whoever is reading it.
"""

import numpy as np


def first_true_time(t, flag):
    """Time of the first sample where flag (a 0/1 or bool array) goes true."""
    i = np.argmax(flag) if flag.any() else None
    return float(t[i]) if i is not None and flag[i] else np.nan


def nearest(t, tq):
    """Index into t nearest to query time tq."""
    return int(np.argmin(np.abs(t - tq)))


def detect_peak_accel(t_hr, accel_mag):
    """Instant of peak IMU acceleration magnitude. Max thrust on a nominal
    flight; the break on a structural failure. Returns (time, peak_g, index)."""
    i = int(np.argmax(accel_mag))
    return float(t_hr[i]), float(accel_mag[i]), i


def detect_peak_spin(t_hr, gyro_mag):
    """Instant of peak angular rate. Distinct from detect_peak_accel - the two
    rarely coincide exactly. Returns (time, peak_deg_per_s, index)."""
    i = int(np.argmax(gyro_mag))
    return float(t_hr[i]), float(gyro_mag[i]), i


# Back-compat aliases for the original failure-analysis-specific names.
detect_shred = detect_peak_accel
detect_tumble_onset = detect_peak_spin
