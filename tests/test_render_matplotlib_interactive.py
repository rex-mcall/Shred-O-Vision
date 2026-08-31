import os

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
