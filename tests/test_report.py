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
