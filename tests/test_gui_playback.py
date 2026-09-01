"""Playback tests that need a REAL GUI backend and event loop.

Every headless (Agg) test passed while Play was completely broken in the
actual app: Agg has no running event loop, so nothing there exercises
"does the clock actually advance after the user clicks Play". These drive a
real window, a real mainloop, and a real mouse click instead. Skipped
automatically wherever a GUI backend isn't usable (headless CI).
"""

import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("tkinter")


@pytest.fixture
def tk_backend():
    """Switch to TkAgg for the test and restore the previous backend after.

    Tk initialization is environment-sensitive (a broken/missing Tcl install
    raises TclError, and headless machines have no display at all), so every
    step here degrades to a skip rather than a failure - this test is here to
    catch a real playback regression, not to police the local Tcl setup.
    """
    import matplotlib.pyplot as plt
    previous = matplotlib.get_backend()
    plt.close("all")
    try:
        matplotlib.use("TkAgg", force=True)
        probe = plt.figure()                              # forces real Tk init
        plt.close(probe)
    except Exception as exc:                              # pragma: no cover
        matplotlib.use(previous, force=True)
        pytest.skip(f"no usable Tk/display in this environment: {exc}")
    try:
        yield
    finally:
        plt.close("all")
        matplotlib.use(previous, force=True)


def _find_button(fig, *labels):
    for ax in fig.axes:
        for text in ax.texts:
            if text.get_text() in labels:
                return ax, text
    return None, None


def _hud_seconds(ax3d):
    head = ax3d.texts[-1].get_text().splitlines()[0]
    return float(head.replace("T+", "").replace("s", "").strip())


def test_clicking_play_actually_advances_the_clock(tk_backend, tmp_path):
    """Regression guard: playback used FuncAnimation, which defers starting
    its timer until the next draw_event - and every redraw here is a blit
    (restore_region/draw_artist/blit), which never emits one. The animation
    was created and never started, so the button label flipped to "Pause"
    while the flight clock sat still. Verified to fail against the
    FuncAnimation implementation and pass with a plain backend timer."""
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseEvent
    from blueraven_visualizer.render_matplotlib import visualize

    import os
    fixtures = os.path.join(os.path.dirname(__file__), "fixtures")
    seen = {}

    def script(fig):
        ax3d = fig.axes[0]
        play_ax, play_label = _find_button(fig, "Play", "Pause")
        seen["label_before"] = play_label.get_text()
        seen["t_before"] = _hud_seconds(ax3d)

        x, y = play_ax.transAxes.transform((0.5, 0.5))
        MouseEvent("button_press_event", fig.canvas, x, y, button=1)._process()
        MouseEvent("button_release_event", fig.canvas, x, y, button=1)._process()
        seen["label_after"] = play_label.get_text()

        fig.canvas.start_event_loop(1.5)      # let the playback timer run
        seen["t_after"] = _hud_seconds(ax3d)
        plt.close(fig)

    def arm(_event):
        if seen.get("armed"):
            return
        seen["armed"] = True
        fig = plt.gcf()
        timer = fig.canvas.new_timer(interval=300)
        timer.single_shot = True
        timer.add_callback(lambda: script(fig))
        timer.start()
        seen["_timer"] = timer               # keep a reference alive

    real_show = plt.show

    def patched_show(*args, **kwargs):
        plt.gcf().canvas.mpl_connect("draw_event", arm)
        real_show(*args, **kwargs)

    plt.show = patched_show
    try:
        visualize(os.path.join(fixtures, "hr_sample.csv"),
                  os.path.join(fixtures, "lr_sample.csv"),
                  obj=None, window=[-1.0, 3.0], fps=15, record=None)
    except Exception as exc:                              # pragma: no cover
        if type(exc).__name__ == "TclError":
            pytest.skip(f"Tk failed mid-test in this environment: {exc}")
        raise
    finally:
        plt.show = real_show

    if "t_after" not in seen:                             # pragma: no cover
        pytest.skip("GUI event loop never delivered the scripted interaction")

    assert seen["label_before"] == "Play"
    assert seen["label_after"] == "Pause", "clicking Play didn't toggle the label"
    advanced = seen["t_after"] - seen["t_before"]
    assert advanced > 0.5, (
        f"flight clock advanced only {advanced:.2f}s during 1.5s of real event loop - "
        f"Play toggled the label but playback never actually started"
    )
