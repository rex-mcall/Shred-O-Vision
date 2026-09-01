import os
import re
import time

import matplotlib.pyplot as plt
import pytest
from matplotlib.backend_bases import KeyEvent

from blueraven_visualizer.render_matplotlib import visualize

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
HR_SAMPLE = os.path.join(FIXTURES, "hr_sample.csv")
LR_SAMPLE = os.path.join(FIXTURES, "lr_sample.csv")


def _press(canvas, key):
    # FigureCanvasAgg (headless/testing) has no key_press_event() convenience
    # method - synthesize and dispatch the event the way matplotlib's own
    # test suite does.
    KeyEvent("key_press_event", canvas, key)._process()


def _hud_time(ax3d):
    text = ax3d.texts[-1].get_text()
    first_line = text.splitlines()[0]
    return float(first_line.replace("T+", "").replace("s", "").strip())


_MATHDEFAULT_NUMBER = re.compile(r"\$\\mathdefault\{(.+)\}\$")


def _slider_value_text(fig):
    """Slider.valtext renders through matplotlib's default mathtext number
    formatting (e.g. '$\\mathdefault{-1}$', with a Unicode minus), not a
    bare float string - this is the only Text on the whole figure using
    that format, so it uniquely identifies the slider's displayed value."""
    for ax in fig.axes:
        for t in ax.texts:
            m = _MATHDEFAULT_NUMBER.fullmatch(t.get_text())
            if m:
                return float(m.group(1).replace("−", "-"))
    return None


def test_keyboard_step_restart_and_play_pause():
    fig = visualize(HR_SAMPLE, LR_SAMPLE, obj=None, window=[-1.0, 1.0],
                     fps=10, record=None)
    try:
        canvas = fig.canvas
        ax3d = fig.axes[0]

        t0 = _hud_time(ax3d)

        _press(canvas, "right")
        _press(canvas, "right")
        t_after_steps = _hud_time(ax3d)
        assert t_after_steps > t0

        _press(canvas, "r")
        assert _hud_time(ax3d) == pytest.approx(t0, abs=0.05)

        # play/pause shouldn't raise, and end/home jump to the frame extremes
        _press(canvas, " ")
        _press(canvas, " ")

        _press(canvas, "end")
        t_end = _hud_time(ax3d)
        assert t_end > t_after_steps

        _press(canvas, "home")
        assert _hud_time(ax3d) == pytest.approx(t0, abs=0.05)

        _press(canvas, "left")  # stepping past the start clamps, doesn't crash
        assert _hud_time(ax3d) == pytest.approx(t0, abs=0.05)
    finally:
        plt.close(fig)


def test_slider_visually_tracks_the_displayed_time():
    """Regression guard: the slider widget's own artists weren't included
    in the blitted redraw, and Slider.set_val() schedules its own separate
    (and, under blitting, stale/overwritten) redraw - so the slider handle
    could visually lag behind the 3D view/telemetry cursors it's supposed
    to represent, even though its internal .val was already correct."""
    fig = visualize(HR_SAMPLE, LR_SAMPLE, obj=None, window=[-1.0, 1.0],
                     fps=10, record=None)
    try:
        canvas = fig.canvas
        ax3d = fig.axes[0]

        for _ in range(5):
            _press(canvas, "right")
        assert _slider_value_text(fig) == pytest.approx(_hud_time(ax3d), abs=0.02)

        _press(canvas, "r")
        assert _slider_value_text(fig) == pytest.approx(_hud_time(ax3d), abs=0.02)
    finally:
        plt.close(fig)


def test_slider_does_not_trigger_its_own_competing_redraw():
    """Regression guard for the actual root cause: Slider.set_val() has its
    own unconditional canvas.draw_idle() call (gated on `drawon`, not on
    `eventson`), which - competing with our own blit-based redraw - made a
    single keyboard step take ~450ms instead of ~15ms in measurement during
    development. drawon=False is what stops it."""
    fig = visualize(HR_SAMPLE, LR_SAMPLE, obj=None, window=[-1.0, 1.0],
                     fps=10, record=None)
    try:
        # Slider objects aren't directly reachable from the returned figure,
        # so this drives the same synthetic-keypress path a real user would
        # and asserts on wall-clock time instead - a fast, focused perf
        # guard against this exact regression, not a broad benchmark.
        canvas = fig.canvas
        t0 = time.perf_counter()
        for _ in range(20):
            KeyEvent("key_press_event", canvas, "right")._process()
        per_step_ms = (time.perf_counter() - t0) / 20 * 1000
        assert per_step_ms < 150, (
            f"{per_step_ms:.1f}ms/step - the slider's own competing redraw "
            f"regressed (see Slider.drawon in render_matplotlib.py)"
        )
    finally:
        plt.close(fig)
