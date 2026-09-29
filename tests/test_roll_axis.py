"""Flight-computer mounting: which board axis points out the rocket's nose.

The quaternions describe the BOARD's attitude. If it's mounted with some axis
other than +X along the nose (a real external log, SN1843, has +Y), treating
+X as the roll axis draws the rocket's roll as an end-over-end cartwheel.

The end-to-end test re-mounts the bundled fixture flight in software - rotating
its accelerometer, gyro and attitude into a board frame whose +Y is the nose -
so the same physical flight must look the same however the board was bolted in.
"""

import csv
import os
import re

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent

from blueraven_visualizer.events import detect_roll_axis
from blueraven_visualizer.io import load_blueraven
from blueraven_visualizer.quaternion import quat_rotmat
from blueraven_visualizer.render_matplotlib import visualize

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
HR = os.path.join(FIXTURES, "hr_sample.csv")
LR = os.path.join(FIXTURES, "lr_sample.csv")

# New board coords of a physical vector = P @ old board coords. Cyclic:
# old +X (the nose) becomes +Y, old +Y becomes +Z, old +Z becomes +X.
P = np.array([[0, 0, 1],
              [1, 0, 0],
              [0, 1, 0]], dtype=float)


def _quat_from_rotmat(R):
    """Scalar-first unit quaternion for rotation matrix R (Shepperd's method)."""
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(1 + tr)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def remount_y_nose(src, dst):
    """Copy an HR log as if the same flight were flown on a board mounted with
    +Y toward the nose instead of +X."""
    with open(src, newline="") as fh:
        rows = list(csv.reader(fh))
    head = rows[0]
    idx = {n: head.index(n) for n in
           ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z",
            "Quat_1", "Quat_2", "Quat_3", "Quat_4"]}
    for row in rows[1:]:
        for kind in ("Accel", "Gyro"):
            v = np.array([float(row[idx[f"{kind}_{a}"]]) for a in "XYZ"])
            for a, val in zip("XYZ", P @ v):
                row[idx[f"{kind}_{a}"]] = f"{val:.6f}"
        q = np.array([float(row[idx[f"Quat_{n}"]]) for n in (1, 2, 3, 4)])
        R_new = quat_rotmat(q / np.linalg.norm(q)) @ P.T      # new board -> world
        for n, val in zip((1, 2, 3, 4), _quat_from_rotmat(R_new)):
            row[idx[f"Quat_{n}"]] = f"{val:.8f}"
    with open(dst, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return str(dst)


def _accel(path):
    _, c = load_blueraven(path)
    return c("Flight_Time_(s)"), np.c_[c("Accel_X"), c("Accel_Y"), c("Accel_Z")]


# ---------------------------------------------------------------- detection

def _synthetic(nose_axis, sign=-1, pad=True, boost=True):
    t = np.arange(-2.0, 3.0, 0.002)
    a = np.zeros((len(t), 3)) + np.random.default_rng(0).normal(0, 0.05, (len(t), 3))
    k = "xyz".index(nose_axis)
    if pad:
        a[t < 0, k] += sign * 1.0
    if boost:
        a[(t > 0) & (t < 1.5), k] += sign * 12.0
    return t, a


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_detects_each_mounting_axis(axis):
    assert detect_roll_axis(*_synthetic(axis)) == "+" + axis


def test_detects_an_upside_down_mounting():
    assert detect_roll_axis(*_synthetic("y", sign=+1)) == "-y"


def test_falls_back_to_the_pad_with_no_boost_data():
    assert detect_roll_axis(*_synthetic("z", boost=False)) == "+z"


def test_inconclusive_data_keeps_the_standard_mounting():
    t = np.arange(-2.0, 3.0, 0.002)
    assert detect_roll_axis(t, np.zeros((len(t), 3))) == "+x"


def test_a_normally_mounted_real_flight_is_detected_as_plus_x():
    assert detect_roll_axis(*_accel(HR)) == "+x"


def test_a_remounted_copy_of_the_same_flight_is_detected_as_plus_y(tmp_path):
    assert detect_roll_axis(*_accel(remount_y_nose(HR, tmp_path / "hr_y.csv"))) == "+y"


# ---------------------------------------------------------------- end to end

def _boost_tilts(hr, **kw):
    """HUD tilt readings stepped through boost and early coast."""
    fig = visualize(hr, LR, obj=None, window=[0.0, 3.0], fps=10, record=None, **kw)
    try:
        tilts = []
        for _ in range(29):
            m = re.search(r"tilt\s+(\d+)", fig.axes[0].texts[-1].get_text())
            tilts.append(int(m.group(1)))
            KeyEvent("key_press_event", fig.canvas, "right")._process()
        return np.array(tilts)
    finally:
        plt.close(fig)


def test_remounted_board_draws_the_same_flight_as_the_original(tmp_path):
    """Regression guard for the reported bug: a +Y-mounted board's roll was
    drawn as a cartwheel. The same physical flight must now read the same
    tilt however the board was mounted - and forcing the old +X assumption
    onto the remounted log must reproduce the cartwheel."""
    remounted = remount_y_nose(HR, tmp_path / "hr_y.csv")
    original = _boost_tilts(HR)
    auto = _boost_tilts(remounted)
    forced_x = _boost_tilts(remounted, roll_axis="+x")

    assert original.max() < 30, f"reference flight should stay near vertical: {original}"
    np.testing.assert_allclose(auto, original, atol=1)
    assert forced_x.max() > 60, "forcing +X on a +Y board should reproduce the cartwheel"
