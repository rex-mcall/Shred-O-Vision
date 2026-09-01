import os

from blueraven_visualizer.io import load_blueraven
from blueraven_visualizer.report import build_report

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
HR_SAMPLE = os.path.join(FIXTURES, "hr_sample.csv")
LR_SAMPLE = os.path.join(FIXTURES, "lr_sample.csv")


def _load_hr():
    _, hc = load_blueraven(HR_SAMPLE)
    t_hr = hc("Flight_Time_(s)")
    accel_mag = (hc("Accel_X") ** 2 + hc("Accel_Y") ** 2 + hc("Accel_Z") ** 2) ** 0.5
    gyro_mag = (hc("Gyro_X") ** 2 + hc("Gyro_Y") ** 2 + hc("Gyro_Z") ** 2) ** 0.5
    return t_hr, accel_mag, gyro_mag


def test_report_hr_only_has_peaks_but_no_lr_sections():
    t_hr, accel_mag, gyro_mag = _load_hr()
    text = build_report(t_hr, accel_mag, gyro_mag)
    assert "BLUE RAVEN FLIGHT REPORT" in text
    assert "Peak acceleration" in text
    assert "Peak angular rate" in text
    assert "Liftoff" not in text
    assert "Baro apogee" not in text


def test_report_uses_neutral_wording_not_failure_wording():
    """The tool replays any flight, not just breakups: a nominal flight's
    peak acceleration is just max thrust, so the report must not label it
    with failure-specific language."""
    t_hr, accel_mag, gyro_mag = _load_hr()
    text = build_report(t_hr, accel_mag, gyro_mag)
    assert "SHRED" not in text.upper()
    assert "tumble" not in text.lower()


def test_report_with_lr_includes_liftoff_and_mach():
    t_hr, accel_mag, gyro_mag = _load_hr()
    _, lc = load_blueraven(LR_SAMPLE)
    t_lr = lc("Flight_Time_(s)")
    text = build_report(t_hr, accel_mag, gyro_mag, t_lr=t_lr, lc=lc)
    assert "Liftoff" in text
    assert "Peak altitude AGL" in text
    assert "Baro apogee" in text
    assert "Mach" in text


def test_report_is_stable_and_nonempty():
    t_hr, accel_mag, gyro_mag = _load_hr()
    text = build_report(t_hr, accel_mag, gyro_mag)
    lines = text.splitlines()
    assert len(lines) > 3
    assert text == build_report(t_hr, accel_mag, gyro_mag)


def test_report_has_no_diagnosis_or_interpretation():
    """The report shows data only - no judgment calls about what caused a
    shred, even when the LR data would match the old transonic-burnout
    heuristic."""
    t_hr, accel_mag, gyro_mag = _load_hr()
    _, lc = load_blueraven(LR_SAMPLE)
    t_lr = lc("Flight_Time_(s)")
    text = build_report(t_hr, accel_mag, gyro_mag, t_lr=t_lr, lc=lc)
    assert "DIAGNOSIS" not in text
    assert "classic" not in text.lower()


EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "mothman_avenged")
EXAMPLE_HR = os.path.join(EXAMPLE_DIR, "BlRv_wvuer1_HR_06-17-2026_09_25_14.csv")
EXAMPLE_LR = os.path.join(EXAMPLE_DIR, "BlRv_wvuer1_LR_06-17-2026_09_25_14.csv")


def test_mach_is_reported_at_the_max_velocity_instant():
    """Regression guard: the report paired max velocity (T+6.28 s, 1602 ft/s)
    with a Mach taken as the maximum over the WHOLE flight, which landed on
    the final sample (T+103.28 s) where the inertial solution had diverged
    to |v| = 3104 ft/s - printing "1602 ft/s (Mach 2.67)", two numbers from
    different moments. 1602 ft/s against a ~1167 ft/s speed of sound is
    Mach ~1.4, which is also what the OpenRocket sim for this flight says."""
    import re
    import numpy as np
    # Deliberately the FULL flight, not the trimmed fixture: the fixture stops
    # at T+17 s and never contains the late diverged samples that caused this.
    _, hc = load_blueraven(EXAMPLE_HR)
    t_hr = hc("Flight_Time_(s)")
    accel_mag = np.linalg.norm(np.c_[hc("Accel_X"), hc("Accel_Y"), hc("Accel_Z")], axis=1)
    gyro_mag = np.linalg.norm(np.c_[hc("Gyro_X"), hc("Gyro_Y"), hc("Gyro_Z")], axis=1)
    _, lc = load_blueraven(EXAMPLE_LR)
    text = build_report(t_hr, accel_mag, gyro_mag, t_lr=lc("Flight_Time_(s)"), lc=lc)

    line = next(l for l in text.splitlines() if "Max velocity" in l)
    ft_s = float(re.search(r"([\d.]+) ft/s", line).group(1))
    mach = float(re.search(r"Mach ([\d.]+)", line).group(1))

    # speed of sound is 1050-1200 ft/s across any plausible flight temperature
    implied_a = ft_s / mach
    assert 1000 < implied_a < 1250, (
        f"{ft_s:.0f} ft/s reported as Mach {mach:.2f} implies a speed of sound of "
        f"{implied_a:.0f} ft/s - the velocity and Mach figures describe different instants"
    )
