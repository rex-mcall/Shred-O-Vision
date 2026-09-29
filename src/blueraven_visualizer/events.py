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


def detect_roll_axis(t_hr, accel):
    """Which body axis of the flight computer points out the rocket's nose,
    as a '+x'-style string.

    The quaternions describe the BOARD's attitude, so drawing the rocket
    right depends on knowing how the board sits in the airframe. The
    standard Blue Raven mounting puts body +X along the nose, but boards get
    mounted other ways - and then roll about the real long axis gets drawn
    as the rocket cartwheeling end over end.

    Read it off the accelerometer instead of assuming. Sitting on the pad the
    nose axis carries all of gravity, and under boost it carries all of the
    thrust; both read NEGATIVE along the nose (a normally mounted board
    shows Accel_X = -0.99 g on the pad and about -12 g in boost). Boost is
    used first since thrust dwarfs any rail angle or pad vibration, with
    the pad as a fallback for a log with little or no powered flight.
    Returns '+x' (the standard mounting) when neither is conclusive.
    """
    t = np.asarray(t_hr, float)
    accel = np.asarray(accel, float)
    for mask, min_g in (((t > 0.1) & (t < 1.0), 3.0),      # boost
                        (t < -0.2, 0.7)):                     # on the pad
        if mask.sum() < 20:
            continue
        mean = np.nanmean(accel[mask], axis=0)
        k = int(np.nanargmax(np.abs(mean)))
        others = np.delete(np.abs(mean), k)
        if abs(mean[k]) >= min_g and np.all(others < 0.5 * abs(mean[k])):
            return ("+" if mean[k] < 0 else "-") + "xyz"[k]
    return "+x"


# Back-compat aliases for the original failure-analysis-specific names.
detect_shred = detect_peak_accel
detect_tumble_onset = detect_peak_spin
