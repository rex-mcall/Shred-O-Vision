import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent

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


def test_manual_camera_rotation_survives_a_subsequent_blit_update():
    """Regression guard: Axes3D's own mouse-drag rotate/pan/zoom
    (Axes3D._on_move) ends with its own independent, full canvas.draw_idle()
    - it knows nothing about our blit setup. Without recapturing the
    background afterward, the very next slider/step/play update calls
    fast_redraw(), which restores the now-STALE cached background (from
    before the rotation) and silently snaps the view back to wherever it
    was - the rotation appears to just not "stick". Recapturing the blit
    background on every button_release_event (covering the end of a
    rotate/pan/zoom drag, alongside window resize) is what fixes it."""
    fig = visualize(HR_SAMPLE, LR_SAMPLE, obj=None, window=[-1.0, 1.0],
                     fps=10, record=None)
    try:
        ax3d = fig.axes[0]
        canvas = fig.canvas

        # Simulate what Axes3D._on_move does at the end of a rotate drag:
        # change the view angle, then its own full redraw.
        ax3d.view_init(elev=60, azim=170)
        fig.canvas.draw()
        img_rotated = np.array(fig.canvas.buffer_rgba())

        # The button release that should end the drag gesture.
        MouseEvent("button_release_event", canvas, 100, 100, button=1)._process()

        # A normal fast_redraw()-driven update, like the user stepping to
        # the next frame - should NOT undo the rotation.
        KeyEvent("key_press_event", canvas, "right")._process()
        img_after_step = np.array(fig.canvas.buffer_rgba())

        diff = np.abs(img_rotated.astype(int) - img_after_step.astype(int))
        changed_fraction = (diff.sum(axis=2) > 10).sum() / diff.shape[0] / diff.shape[1]
        # A real snap-back changes ~11% of pixels (measured); a normal
        # one-frame step (mesh pose + cursor) changes ~1%. Cut well between
        # the two.
        assert changed_fraction < 0.05, (
            f"{changed_fraction:.1%} of pixels changed after stepping post-rotation - "
            f"looks like the camera snapped back to its pre-rotation angle"
        )
    finally:
        plt.close(fig)


def test_first_frame_is_painted_before_any_user_interaction():
    """Regression guard: capture_blit_background() hides the animated
    artists, does a full draw to snapshot the static background, then
    unhides them - but that background draw is what's left on screen. Without
    painting them back, the window opened with a completely EMPTY 3D panel
    (no rocket at all) until the user happened to touch a control.

    Compares the panel as-shown against the same panel with the mesh
    explicitly hidden: if the rocket was never painted, the two are
    identical."""
    fig = visualize(HR_SAMPLE, LR_SAMPLE, obj=None, window=[-1.0, 1.0],
                     fps=10, record=None)
    try:
        ax3d = fig.axes[0]
        as_shown = np.array(fig.canvas.copy_from_bbox(ax3d.bbox))

        mesh = ax3d.collections[0]
        mesh.set_visible(False)
        fig.canvas.draw()
        without_mesh = np.array(fig.canvas.copy_from_bbox(ax3d.bbox))

        assert as_shown.shape == without_mesh.shape
        assert not np.array_equal(as_shown, without_mesh), (
            "the 3D panel as first shown is pixel-identical to one rendered with "
            "the rocket hidden - the mesh was never painted after the blit "
            "background capture"
        )
    finally:
        plt.close(fig)
