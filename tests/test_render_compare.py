"""Two-flight comparison: sync, HUD content, windowing, export, and the
interactive player.

Most tests build the second flight by rewriting the bundled fixture - shifting
its clock, moving a flag, or trimming rows - so every expectation is exact:
a flight compared against a time-shifted copy of itself must line up sample
for sample when (and only when) syncing is correct.
"""

import csv
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent

from blueraven_visualizer.atmosphere import mach_number
from blueraven_visualizer.io import load_blueraven
from blueraven_visualizer.render_compare import (
    Flight, compare_flights, flight_label, hud_text, sync_window,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
HR = os.path.join(FIXTURES, "hr_sample.csv")
LR = os.path.join(FIXTURES, "lr_sample.csv")
TINY_OBJ = os.path.join(FIXTURES, "tiny.obj")

APOGEE = 13.88          # fixture LR Apogee flag, log time
LR_END = 17.00          # fixture LR last sample


def rewrite(src, dst, *, shift=0.0, flags=None, keep_until=None, drop_before=None):
    """Copy a Blue Raven CSV with its clock shifted by ``shift`` seconds.

    ``flags`` maps a flag column to the ORIGINAL log time it first goes true
    (None = never). ``keep_until`` / ``drop_before`` trim rows by original time.
    """
    with open(src, newline="") as fh:
        rows = list(csv.reader(fh))
    head = rows[0]
    ti = head.index("Flight_Time_(s)")
    out = [head]
    for row in rows[1:]:
        t = float(row[ti])
        if keep_until is not None and t > keep_until:
            continue
        if drop_before is not None and t < drop_before:
            continue
        row = list(row)
        row[ti] = f"{t + shift:.3f}"
        for name, t_true in (flags or {}).items():
            row[head.index(name)] = "0" if (t_true is None or t < t_true) else "1"
        out.append(row)
    with open(dst, "w", newline="") as fh:
        csv.writer(fh).writerows(out)
    return str(dst)


def flight_pair(tmp_path, **kw):
    return (rewrite(HR, tmp_path / "hr_b.csv", **{k: v for k, v in kw.items() if k != "flags"}),
            rewrite(LR, tmp_path / "lr_b.csv", **kw))


# ---------------------------------------------------------------- labels

@pytest.mark.parametrize("name,expected", [
    ("BlRv_wvuer1 HR_06-17-2026_09_25_14.csv", "wvuer1  06-17-2026 09:25"),
    ("BlRv_wvuer1_LR_11-05-2024_17_41_04.csv", "wvuer1  11-05-2024 17:41"),
    ("my_flight.csv", "my_flight"),
])
def test_flight_label_comes_from_blue_raven_filenames(name, expected):
    assert flight_label(os.path.join("somewhere", name)) == expected


# ---------------------------------------------------------------- sync

def test_a_time_shifted_copy_lines_up_sample_for_sample(tmp_path):
    """THE sync test: the same flight with its clock moved 5 s later must show
    identical altitude, Mach and attitude at every synced instant, with its own
    log time running exactly 5 s ahead."""
    a = Flight(HR, LR)
    b = Flight(*flight_pair(tmp_path, shift=5.0))
    assert b.liftoff == pytest.approx(a.liftoff + 5.0, abs=1e-6)

    for ts in (-1.5, 0.0, 2.34, 6.28, 9.99, 13.0):
        sa, sb = a.sample(ts), b.sample(ts)
        assert sb["t_log"] == pytest.approx(sa["t_log"] + 5.0, abs=1e-6)
        assert sb["alt"] == sa["alt"]
        np.testing.assert_array_equal(sb["mach"], sa["mach"])
        np.testing.assert_array_equal(b.Q[sb["ih"]], a.Q[sa["ih"]])


def test_sync_uses_the_liftoff_flag_not_log_time_zero(tmp_path):
    """A log whose Liftoff flag isn't at t=0 must still sync at liftoff."""
    b = Flight(*flight_pair(tmp_path, flags={"Liftoff": 2.0}))
    assert b.liftoff == pytest.approx(2.0, abs=0.02)
    assert b.sample(0.0)["t_log"] == pytest.approx(2.0, abs=0.02)


def test_missing_liftoff_flag_falls_back_to_log_zero_and_says_so(tmp_path):
    b = Flight(*flight_pair(tmp_path, flags={"Liftoff": None}))
    assert b.liftoff == 0.0
    assert any("Liftoff" in n for n in b.notes)


# ---------------------------------------------------------------- Mach

def test_mach_is_live_before_apogee_and_blank_after():
    f = Flight(HR, LR)
    before, after = f.sample(5.0), f.sample(APOGEE + 1.0)
    assert np.isfinite(before["mach"]) and not before["past_apogee"]
    assert np.isnan(after["mach"]) and after["past_apogee"]
    assert "(past apogee)" in hud_text(after)


def test_mach_matches_the_atmosphere_calculation_at_that_sample():
    f = Flight(HR, LR)
    _, lc = load_blueraven(LR)
    t = lc("Flight_Time_(s)")
    i = int(np.argmin(np.abs(t - 6.28)))
    expected = mach_number(lc("Velocity_Up")[i], lc("Velocity_DR")[i],
                           lc("Velocity_CR")[i], lc("Temperature_(F)")[i])
    assert f.sample(6.28)["mach"] == pytest.approx(float(expected))


def test_without_an_apogee_flag_mach_stays_visible_and_that_is_noted(tmp_path):
    f = Flight(*flight_pair(tmp_path, flags={"Apogee": None}))
    assert np.isfinite(f.sample(APOGEE + 1.0)["mach"])
    assert any("Apogee" in n for n in f.notes)


# ---------------------------------------------------------------- window & data edges

def test_default_window_runs_until_the_later_apogee(tmp_path):
    a = Flight(HR, LR)
    b = Flight(*flight_pair(tmp_path, flags={"Apogee": 10.0}))
    for fa, fb in ((a, b), (b, a)):
        start, end = sync_window(fa, fb, None)
        assert end == pytest.approx(APOGEE, abs=0.02)
        assert start < 0


def test_full_window_runs_to_the_end_of_the_longer_recording(tmp_path):
    a = Flight(HR, LR)
    b = Flight(*flight_pair(tmp_path, keep_until=8.0))
    assert sync_window(a, b, "full")[1] == pytest.approx(LR_END, abs=0.02)


def test_peak_window_is_refused_for_comparisons():
    a = Flight(HR, LR)
    with pytest.raises(ValueError, match="peak"):
        sync_window(a, a, "peak")


def test_hud_reports_before_and_after_each_flights_own_data(tmp_path):
    late_start = Flight(*flight_pair(tmp_path, drop_before=2.0))
    early_end = Flight(rewrite(HR, tmp_path / "h2.csv", keep_until=8.0),
                       rewrite(LR, tmp_path / "l2.csv", keep_until=8.0))
    assert late_start.sample(-1.0)["status"] == "before"
    assert "(no data yet)" in hud_text(late_start.sample(-1.0))
    assert early_end.sample(12.0)["status"] == "after"
    assert "(end of data)" in hud_text(early_end.sample(12.0))
    assert early_end.sample(5.0)["status"] == "live"


def test_hud_values_share_one_right_aligned_column():
    lines = hud_text(Flight(HR, LR).sample(6.28)).splitlines()
    assert len(lines) == 3
    assert len({len(line) for line in lines}) == 1, lines


def test_lr_file_is_required():
    with pytest.raises(ValueError, match="LOW-RATE"):
        Flight(HR, None)


def test_bad_output_extension_fails_before_touching_any_file():
    with pytest.raises(ValueError, match=r"\.mp4 or \.gif"):
        compare_flights("nope.csv", "nope.csv", "nope.csv", "nope.csv", record="clip")


# ---------------------------------------------------------------- models & labels

def test_second_flight_reuses_the_first_flights_model_by_default():
    fig = compare_flights(HR, LR, HR, LR, obj_a=TINY_OBJ, window=(0.0, 0.5), record=None)
    try:
        ax_a, ax_b = fig.axes[0], fig.axes[1]
        assert len(ax_a.collections[0].get_paths()) == len(ax_b.collections[0].get_paths())
    finally:
        plt.close(fig)

    fig = compare_flights(HR, LR, HR, LR, obj_a=TINY_OBJ, obj_b=None, window=(0.0, 0.5), record=None)
    try:
        assert len(fig.axes[0].collections[0].get_paths()) != len(fig.axes[1].collections[0].get_paths())
    finally:
        plt.close(fig)


def test_identical_labels_are_disambiguated():
    fig = compare_flights(HR, LR, HR, LR, window=(0.0, 0.5), record=None)
    try:
        titles = [fig.axes[0].get_title(), fig.axes[1].get_title()]
        assert titles[0] != titles[1]
        assert titles[0].endswith("(A)") and titles[1].endswith("(B)")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------- export

@pytest.mark.parametrize("ext", ["mp4", "gif"])
def test_export_writes_a_video(tmp_path, ext):
    out = tmp_path / f"cmp.{ext}"
    compare_flights(HR, LR, HR, LR, window=(0.0, 1.0), fps=5, record=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_mp4_frames_are_padded_not_rescaled_to_the_macroblock_size(tmp_path):
    """H.264 wants dimensions divisible by 16; the encoder used to rescale
    1400x800 to 1408x800 (blurring every frame) and warn about it."""
    import imageio_ffmpeg
    out = tmp_path / "cmp.mp4"
    compare_flights(HR, LR, HR, LR, window=(0.0, 0.6), fps=5, record=str(out))
    reader = imageio_ffmpeg.read_frames(str(out))
    meta = next(reader)
    reader.close()
    w, h = meta["size"]
    assert w % 16 == 0 and h % 16 == 0


# ---------------------------------------------------------------- interactive (Agg)

def _title_seconds(fig):
    return float(fig.texts[0].get_text().split()[1])


def test_first_frame_paints_both_rockets():
    fig = compare_flights(HR, LR, HR, LR, window=(-1.0, 2.0), record=None)
    try:
        for ax in fig.axes[:2]:
            shown = np.array(fig.canvas.copy_from_bbox(ax.bbox))
            mesh = ax.collections[0]
            mesh.set_visible(False)
            fig.canvas.draw()
            hidden = np.array(fig.canvas.copy_from_bbox(ax.bbox))
            mesh.set_visible(True)
            fig.canvas.draw()
            assert not np.array_equal(shown, hidden), f"{ax.get_title()} not painted"
    finally:
        plt.close(fig)


def test_stepping_advances_the_shared_clock_and_both_huds_together(tmp_path):
    fig = compare_flights(HR, LR, *flight_pair(tmp_path, shift=5.0),
                          window=(-1.0, 2.0), fps=10, record=None)
    try:
        t0 = _title_seconds(fig)
        for _ in range(4):
            KeyEvent("key_press_event", fig.canvas, "right")._process()
        t1 = _title_seconds(fig)
        assert t1 > t0

        log_a = float(fig.axes[0].texts[-1].get_text().split()[2])
        log_b = float(fig.axes[1].texts[-1].get_text().split()[2])
        assert log_a == pytest.approx(t1, abs=0.02)
        assert log_b == pytest.approx(t1 + 5.0, abs=0.02)
    finally:
        plt.close(fig)


def test_cameras_are_linked():
    fig = compare_flights(HR, LR, HR, LR, window=(0.0, 1.0), record=None)
    try:
        ax_a, ax_b = fig.axes[0], fig.axes[1]
        ax_a.view_init(elev=40, azim=10, share=True)
        assert (ax_b.elev, ax_b.azim) == (40, 10)
    finally:
        plt.close(fig)
