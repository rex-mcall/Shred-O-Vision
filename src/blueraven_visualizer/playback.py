"""Shared playback machinery for the matplotlib renderers.

Both the single-flight dashboard (render_matplotlib) and the two-flight
comparison (render_compare) animate the same way: a caller-supplied
``draw(t)`` updates a set of artists, and this module turns that into either
an exported MP4/GIF or an interactive player (slider, Play/Pause, Step,
Restart, keyboard shortcuts).

There is deliberately one copy of this. Every hard bug fixed in it - a blank
3D panel on the first frame, Play doing nothing, the camera snapping back
after a mouse rotate, the slider lagging the mesh - came from how playback
interacts with matplotlib's redraw model, not from anything flight-specific,
and a second copy would have to rediscover each one.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

VIDEO_EXTS = (".mp4", ".gif")


def validate_record_path(record):
    """Fail fast on an output path we can't encode.

    Without a recognized extension the encoder can't tell what to write and
    fails deep inside a third-party library with a bare ``KeyError: None``.
    """
    ext = os.path.splitext(record)[1].lower()
    if ext not in VIDEO_EXTS:
        raise ValueError(
            f"Can't tell what format to save as: {record!r} needs to end in .mp4 or .gif."
        )
    return ext


def nearest_index(t, x):
    """Index of the sample in sorted array ``t`` closest to ``x`` (clamped).

    Binary search rather than argmin over the whole array: playback calls
    this every frame, and a long flight's 500 Hz log is ~180k samples.
    """
    n = len(t)
    if n == 0:
        raise ValueError("empty time array")
    i = int(np.searchsorted(t, x))
    if i <= 0:
        return 0
    if i >= n:
        return n - 1
    return i - 1 if (x - t[i - 1]) <= (t[i] - x) else i


def _draw_artist(fig, artist):
    # Figure-level artists (e.g. a shared title) have no parent axes.
    (artist.axes if artist.axes is not None else fig).draw_artist(artist)


def export_video(fig, frame_times, draw, animated, record, fps):
    """Render ``frame_times`` through ``draw`` into an MP4 or GIF.

    Frames are rendered by BLITTING, not by a full canvas draw each time.
    Measured on the single-flight dashboard: a full redraw costs ~280 ms/frame
    before any mesh is considered - matplotlib rebuilding axis chrome, ticks
    and font metrics that never change between frames - while only the
    ``animated`` artists actually move. Reusing a cached background and
    redrawing just those cuts the fixed cost to a few ms.
    """
    ext = validate_record_path(record)
    for a in animated:
        a.set_animated(True)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)

    def render_frame(tf):
        draw(tf)
        fig.canvas.restore_region(background)
        for a in animated:
            _draw_artist(fig, a)
        fig.canvas.blit(fig.bbox)
        return np.asarray(fig.canvas.buffer_rgba())

    n_total = len(frame_times)
    print(f"Rendering {n_total} frames to {record} ...")
    show_progress = n_total >= 100
    progress_step = max(1, n_total // 10)

    def note_progress(i):
        if show_progress and i and (i % progress_step == 0 or i == n_total - 1):
            print(f"  {i + 1}/{n_total} frames ({(i + 1) / n_total:.0%})", flush=True)

    if ext == ".gif":
        from PIL import Image
        frames = []
        for i, tf in enumerate(frame_times):
            frames.append(Image.fromarray(render_frame(tf).copy())
                          .convert("P", palette=Image.ADAPTIVE))
            note_progress(i)
        frames[0].save(record, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)
    else:
        # imageio-ffmpeg ships its own ffmpeg binary (a core dependency), so
        # MP4 export needs nothing installed system-wide.
        import imageio_ffmpeg
        # H.264 wants dimensions divisible by 16. Left to itself the encoder
        # RESCALES a 1400x800 frame to 1408x800 - slightly blurring every
        # frame - and prints a warning about it. Pad with white instead.
        h, w = fig.canvas.get_width_height()[::-1]
        pad_h, pad_w = (-h) % 16, (-w) % 16
        writer = imageio_ffmpeg.write_frames(
            record, (w + pad_w, h + pad_h), fps=fps, pix_fmt_in="rgba", quality=7,
            ffmpeg_log_level="error",
        )
        writer.send(None)
        try:
            for i, tf in enumerate(frame_times):
                frame = render_frame(tf)
                if pad_h or pad_w:
                    frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)),
                                   constant_values=255)
                writer.send(np.ascontiguousarray(frame).tobytes())
                note_progress(i)
        finally:
            writer.close()

    plt.close(fig)
    print(f"Saved {record}  ({n_total} frames, {n_total / fps:.1f}s)")
    return record


def run_interactive(fig, frame_times, draw, animated, *, t_end, fps, speed,
                    blit=True, slider_label="t (s)"):
    """Attach playback controls to ``fig`` and show it.

    ``draw(t)`` must update every artist in ``animated``. The caller leaves
    room along the bottom of the figure (~0.12 of its height) for the
    controls, which are added after all of the caller's own axes.
    """
    from matplotlib.widgets import Slider, Button

    n_frames = len(frame_times)
    t_start = float(frame_times[0])

    sax = fig.add_axes([0.08, 0.02, 0.84, 0.022])
    rax = fig.add_axes([0.08, 0.065, 0.14, 0.035])
    lax = fig.add_axes([0.23, 0.065, 0.09, 0.035])
    pax = fig.add_axes([0.33, 0.065, 0.12, 0.035])
    nax = fig.add_axes([0.46, 0.065, 0.09, 0.035])
    sld = Slider(sax, slider_label, t_start, t_end, valinit=t_start)
    # Slider.set_val() unconditionally triggers its own canvas.draw_idle()
    # (gated on `drawon`, not `eventson` - eventson=False in sync_slider()
    # only suppresses the on_changed callback): a second, competing full
    # redraw on every frame that undoes the point of blitting. The slider's
    # own artists are blitted with everything else instead.
    sld.drawon = False
    btn_restart = Button(rax, "|<< Restart")
    btn_prev = Button(lax, "Step <")
    btn_play = Button(pax, "Play")
    btn_next = Button(nax, "Step >")

    state = {"playing": False, "timer": None, "i": 0}

    # Everything that changes per frame: the caller's artists, AND the
    # slider's bar/handle/value label (whose own deferred redraws get
    # coalesced by the GUI loop and lag during playback), AND the Play
    # button's label, whose "Play"/"Pause" text would otherwise stay stale
    # under blitting because the cached background has the old text baked in.
    anim_artists = list(animated) + [sld.poly, sld._handle, sld.valtext, btn_play.label]

    use_blit = blit
    blit_bg = {"data": None}

    def paint_animated():
        for a in anim_artists:
            _draw_artist(fig, a)
        fig.canvas.blit(fig.bbox)

    def on_draw(_event):
        """Re-establish blitting after ANY full canvas draw.

        Animated artists are skipped by normal draws, so every full redraw
        leaves the canvas holding exactly the static background to cache -
        and leaves our artists off-screen until they're painted back. One
        hook covers every source of a full redraw: the GUI's own first paint
        when the window opens (painting once before show() isn't enough - a
        real backend redraws on realize and wipes it, leaving an empty 3D
        panel and a blank Play button), window resizes, and the 3D axes'
        mouse-drag rotate/pan/zoom, whose own draw_idle() would otherwise
        leave the cached background stale enough to snap the camera back.
        """
        if not use_blit:
            return
        blit_bg["data"] = fig.canvas.copy_from_bbox(fig.bbox)
        paint_animated()

    def fast_redraw():
        if use_blit and blit_bg["data"] is not None:
            fig.canvas.restore_region(blit_bg["data"])
            paint_animated()
        else:
            fig.canvas.draw_idle()

    if use_blit:
        for a in anim_artists:
            a.set_animated(True)
        fig.canvas.mpl_connect("draw_event", on_draw)
        fig.canvas.draw()

    def sync_slider(i):
        sld.eventson = False
        sld.set_val(frame_times[i])
        sld.eventson = True

    def goto(i):
        i = max(0, min(i, n_frames - 1))
        state["i"] = i
        draw(frame_times[i])
        sync_slider(i)
        fast_redraw()

    def on_slider(val):
        state["i"] = nearest_index(frame_times, val)
        draw(val)
        fast_redraw()
    sld.on_changed(on_slider)

    def step():
        # Paced to elapsed wall-clock time (scaled by `speed`), not a fixed
        # frame increment: if a frame took longer than its 1/fps slot, jump
        # to where playback should be *now* instead of drifting into slow
        # motion.
        elapsed = time.perf_counter() - state["wall_t0"]
        target_t = state["data_t0"] + elapsed * speed
        if target_t >= t_end:
            stop()
            goto(n_frames - 1)
            return
        i = nearest_index(frame_times, target_t)
        if i == state["i"]:
            return
        state["i"] = i
        draw(frame_times[i])
        sync_slider(i)
        fast_redraw()

    def stop():
        timer = state.get("timer")
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
            state["timer"] = None
        state["playing"] = False
        btn_play.label.set_text("Play")
        fast_redraw()

    def play():
        if state["playing"]:
            return
        if state["i"] >= n_frames - 1:
            state["i"] = 0
        state["playing"] = True
        btn_play.label.set_text("Pause")
        state["wall_t0"] = time.perf_counter()
        state["data_t0"] = frame_times[state["i"]]
        # A plain backend timer, NOT FuncAnimation: FuncAnimation defers
        # starting until the next draw_event, and every redraw here is a blit,
        # which never emits one - so it was created and never started, and
        # Play appeared to do nothing while the slider still worked.
        interval_ms = int(round(1000.0 / max(fps, 1)))
        timer = fig.canvas.new_timer(interval=interval_ms)
        timer.add_callback(step)
        timer.start()
        state["timer"] = timer
        fast_redraw()

    def toggle_play(_=None):
        stop() if state["playing"] else play()
    btn_play.on_clicked(toggle_play)

    def restart(_=None):
        stop()
        goto(0)
    btn_restart.on_clicked(restart)

    def step_prev(_=None):
        stop()
        goto(state["i"] - 1)
    btn_prev.on_clicked(step_prev)

    def step_next(_=None):
        stop()
        goto(state["i"] + 1)
    btn_next.on_clicked(step_next)

    def on_key(event):
        if event.key == " ":
            toggle_play()
        elif event.key == "right":
            step_next()
        elif event.key == "left":
            step_prev()
        elif event.key in ("r", "home"):
            restart()
        elif event.key == "end":
            stop()
            goto(n_frames - 1)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Controls: Play/Pause or SPACE | drag slider to scrub | "
          "Step</Step> or LEFT/RIGHT arrows | Restart or R/Home | End = last frame")
    plt.show()
    return fig
