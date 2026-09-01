"""matplotlib rendering backend.

No heavy 3D-engine dependency: draws the OBJ (or a built-in fins-and-colors
rocket glyph, untextured either way) spinning in a 3D panel next to a
synchronized telemetry dashboard, with a moving time cursor. Works headless
(Agg backend) for MP4/GIF export, or interactively with a scrub slider.
Interactive playback is paced to the wall clock (not a fixed frame budget),
so it tracks real time even when a frame takes longer to render than its
nominal slot.

The LR (low-rate baro/derived) CSV is optional: without it you still get the
3D orientation view and HR-derived accel/gyro panels, just no altitude,
velocity, pyro-voltage, tilt/roll, or LR-flag event markers.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .dialogs import ASK, resolve_files
from .io import load_blueraven
from .quaternion import quat_rotmat, align_rotation, nose_vec
from .mesh import (
    load_obj, rocket_primitive, decimate_mesh, glyph_face_colors, shade_triangles,
    angular_roll_marker, solid_color_with_marker,
)
from .events import first_true_time, nearest, detect_shred
from .report import build_report

BASE_COLOR = (0.78, 0.80, 0.90)
SHRED_COLOR = (0.85, 0.06, 0.10)


def visualize(hr_csv=ASK, lr_csv=ASK, obj=ASK, *, window=None, pad=1.5,
              model_nose="+z", upright_start=True, shred_highlight=True,
              fps=30, speed=1.0, record=None, decim_plot=5,
              show_3d=True, max_faces="auto", dpi=100, blit=True):

    # ---- resolve files (pops dialogs for any left as ASK) ----
    hr_csv, lr_csv, obj = resolve_files(hr_csv, lr_csv, obj)
    has_lr = lr_csv is not None

    if max_faces == "auto":
        # Interactive playback needs to redraw many times a second, so a
        # detailed real-world OBJ (tens of thousands of faces) has to be
        # decimated to stay smooth. A one-time video export doesn't have
        # that constraint - it just needs to finish rendering in a
        # reasonable total time, not hit a live frame budget - so give it
        # the full, undecimated mesh instead: much better fin/panel detail
        # in the saved clip, at the cost of a slower (but still one-time)
        # export.
        max_faces = None if record else 10000

    # ---- load ----
    _, hc = load_blueraven(hr_csv)

    t_hr = hc("Flight_Time_(s)")
    Q = np.c_[hc("Quat_1"), hc("Quat_2"), hc("Quat_3"), hc("Quat_4")]
    Q = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)
    acc = np.c_[hc("Accel_X"), hc("Accel_Y"), hc("Accel_Z")]
    accel_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(np.c_[hc("Gyro_X"), hc("Gyro_Y"), hc("Gyro_Z")], axis=1)

    tShred, gPk, iSh = detect_shred(t_hr, accel_mag)
    events = {"shred": tShred}

    lc = None
    if has_lr:
        _, lc = load_blueraven(lr_csv)
        t_lr = lc("Flight_Time_(s)")
        volts = {k: lc(k + "_Volts") for k in ["Apo", "Main", "3rd", "4th"]}
        batt = lc("Batt_Volts")
        alt = lc("Baro_Altitude_AGL_(feet)")
        vup = lc("Velocity_Up")
        tilt = lc("Tilt_Angle_(deg)")
        roll = lc("Roll_Angle_(deg)")
        events.update({
            "liftoff": first_true_time(t_lr, lc("Liftoff") > 0.5),
            "burnout": first_true_time(t_lr, lc("Burnout_Coast") > 0.5),
            "apogee":  first_true_time(t_lr, lc("Apogee") > 0.5),
            "apo fire": first_true_time(t_lr, lc("Apo_fired") > 0.5),
            "main fire": first_true_time(t_lr, lc("Main_fired") > 0.5),
        })

    print(build_report(t_hr, accel_mag, gyro_mag, t_lr=(t_lr if has_lr else None), lc=lc))

    # ---- playback window ----
    t_win_src = t_lr if has_lr else t_hr
    if window == "shred":
        win = (tShred - pad, tShred + pad)
    elif window:
        win = tuple(window)
    else:
        win = (float(t_win_src[0]), float(t_win_src[-1]))
    win = (max(win[0], t_win_src[0]), min(win[1], t_win_src[-1]))

    # ---- model ----
    parts = None
    if obj:
        V, F = load_obj(obj)
        n0 = len(F)
        V, F = decimate_mesh(V, F, max_faces)
        if len(F) < n0:
            print(f"Mesh decimated {n0} -> {len(F)} faces for smooth playback "
                  f"(raise max_faces, or set max_faces=None for MP4 export).")
    else:
        V, F, parts = rocket_primitive()   # small enough it never needs decimation
    # Center on the bounding-box midpoint, not the vertex mean: real OBJ
    # exports are often unevenly tessellated (e.g. a finely-meshed tail
    # section next to a coarse nosecone), which skews the mean well off the
    # object's actual visual center and inflates Rmax below - wasting frame
    # space around a model that then looks tiny/needle-thin.
    V = V - (V.min(0) + V.max(0)) / 2
    V = (align_rotation(nose_vec(model_nose), [1, 0, 0]) @ V.T).T   # nose -> +X
    Rmax = float(np.linalg.norm(V, axis=1).max())

    # A real imported OBJ has no per-part labels to build a roll marker from
    # the way the built-in glyph's parts do - so give it the same kind of
    # spin-visibility stripe based on angular position around the model's
    # own long axis instead, which works on any mesh regardless of what its
    # geometry actually represents.
    roll_marker = angular_roll_marker(V, F) if parts is None else None

    v0 = quat_rotmat(Q[nearest(t_hr, win[0])]) @ np.array([1.0, 0, 0])
    Rworld = align_rotation(v0, [0, 0, 1]) if upright_start else np.eye(3)

    # ---- figure layout ----
    if show_3d:
        fig = plt.figure(figsize=(14, 8), facecolor="white", dpi=dpi)
        gs = GridSpec(4, 2, width_ratios=[1.05, 1.0], hspace=0.5, wspace=0.24,
                      left=0.04, right=0.93, top=0.93, bottom=0.16)
        tcol = 1
        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    else:
        fig = plt.figure(figsize=(11, 8), facecolor="white", dpi=dpi)
        gs = GridSpec(4, 1, hspace=0.5, left=0.08, right=0.90, top=0.93, bottom=0.16)
        tcol = 0
        ax3d = None

    if has_lr:
        axV = fig.add_subplot(gs[0, tcol])
        axA = fig.add_subplot(gs[1, tcol], sharex=axV)
        axH = fig.add_subplot(gs[2, tcol], sharex=axV)
        axT = fig.add_subplot(gs[3, tcol], sharex=axV)
        tele = [axV, axA, axH, axT]
    else:
        axA = fig.add_subplot(gs[0:2, tcol])
        axG = fig.add_subplot(gs[2:4, tcol], sharex=axA)
        tele = [axA, axG]

    # 3D panel (no edge lines -> much faster software rasterization)
    face_colors_normal = face_colors_shred = None
    if parts is not None:
        face_colors_normal = glyph_face_colors(parts)
        face_colors_shred = glyph_face_colors(parts, highlight=SHRED_COLOR)
    elif roll_marker is not None:
        face_colors_normal = solid_color_with_marker(BASE_COLOR, roll_marker)
        face_colors_shred = solid_color_with_marker(SHRED_COLOR, roll_marker)

    mesh = hud = None
    if show_3d:
        mesh = Poly3DCollection(V[F], facecolor=(face_colors_normal if face_colors_normal is not None else BASE_COLOR),
                                edgecolor="none")
        ax3d.add_collection3d(mesh)
        # The axis box has to be a symmetric cube sized to fit the model at
        # ANY rotation (it tumbles), so a slender model - long body, small
        # diameter - will always show some empty margin. Keep that margin
        # tight (not the ~50% used previously) so the model actually reads
        # as a rocket instead of a thin sliver lost in empty space.
        for setlim in (ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim):
            setlim(-Rmax * 1.12, Rmax * 1.12)
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z (up)")
        ax3d.view_init(elev=16, azim=-60)
        up = Rworld @ v0 * Rmax * 1.05
        ax3d.plot([0, up[0]], [0, up[1]], [0, up[2]], "--", color="0.6", lw=1)
        hud = ax3d.text2D(0.02, 0.97, "", transform=ax3d.transAxes, va="top",
                          fontsize=10, family="monospace")

    # telemetry panels (static traces; cursor moves)
    if has_lr:
        axV.plot(t_lr, volts["Apo"], label="Apo", lw=1.1)
        axV.plot(t_lr, volts["Main"], label="Main", lw=1.1)
        axV.plot(t_lr, volts["3rd"], label="3rd", lw=1.0)
        axV.plot(t_lr, volts["4th"], label="4th", lw=1.0)
        axVb = axV.twinx()
        axVb.plot(t_lr, batt, color="0.45", lw=1.0, ls=":")
        axVb.set_ylabel("Batt V", color="0.45")
        axV.set_ylabel("Pyro V"); axV.set_title("Ejection-charge continuity & battery")
        axV.legend(ncol=4, fontsize=7, loc="upper right")

        axA.plot(t_hr[::decim_plot], accel_mag[::decim_plot], color=(0.80, 0.10, 0.20), lw=0.7)
        axA.set_ylabel("|accel| (g)"); axA.set_title("Acceleration magnitude (500 Hz)")
        axA.annotate(f"{gPk:.0f} g", (tShred, gPk), textcoords="offset points",
                     xytext=(5, -2), color=SHRED_COLOR, fontweight="bold")

        axH.plot(t_lr, alt, color=(0.10, 0.45, 0.80), lw=1.3, label="alt AGL")
        axHv = axH.twinx()
        axHv.plot(t_lr, vup, color=(0.85, 0.33, 0.10), lw=1.0, label="vel up")
        axH.set_ylabel("Alt AGL (ft)"); axHv.set_ylabel("Vel up (ft/s)", color=(0.85, 0.33, 0.10))
        axH.set_title("Altitude & vertical velocity")

        axT.plot(t_lr, tilt, color=(0.10, 0.60, 0.45), lw=1.3, label="tilt")
        axT.plot(t_lr, roll, color=(0.90, 0.55, 0.10), lw=0.9, label="roll")
        axT.set_ylabel("deg"); axT.set_title("Tilt & roll"); axT.set_xlabel("Flight time (s)")
        axT.legend(ncol=2, fontsize=7, loc="upper right")
    else:
        axA.plot(t_hr[::decim_plot], accel_mag[::decim_plot], color=(0.80, 0.10, 0.20), lw=0.7)
        axA.set_ylabel("|accel| (g)"); axA.set_title("Acceleration magnitude (500 Hz)")
        axA.annotate(f"{gPk:.0f} g", (tShred, gPk), textcoords="offset points",
                     xytext=(5, -2), color=SHRED_COLOR, fontweight="bold")

        axG.plot(t_hr[::decim_plot], gyro_mag[::decim_plot], color=(0.50, 0.15, 0.65), lw=0.7)
        axG.set_ylabel("|gyro| (deg/s)"); axG.set_title("Angular rate magnitude (500 Hz)")
        axG.set_xlabel("Flight time (s)")

    # event lines + moving cursor on every telemetry panel
    cursors = []
    for ax in tele:
        for name, te in events.items():
            if np.isnan(te):
                continue
            ax.axvline(te, color=(SHRED_COLOR if name == "shred" else "0.6"),
                       lw=(1.4 if name == "shred" else 0.7),
                       ls=("-" if name == "shred" else ":"), alpha=0.8)
        cursors.append(ax.axvline(win[0], color="k", lw=1.0))
        ax.set_xlim(*win)
    for ax in tele[:-1]:
        ax.tick_params(labelbottom=False)

    # ---- frame update ----
    frame_times = np.arange(win[0], win[1], speed / fps)

    def draw(tf):
        artists = []
        if show_3d:
            ih = nearest(t_hr, tf)
            Vk = (Rworld @ quat_rotmat(Q[ih]) @ V.T).T
            tri = Vk[F]
            mesh.set_verts(tri)
            post = (not np.isnan(tShred)) and tf >= tShred
            if face_colors_normal is not None:
                base = face_colors_shred if (post and shred_highlight) else face_colors_normal
            else:
                base = SHRED_COLOR if (post and shred_highlight) else BASE_COLOR
            mesh.set_facecolor(shade_triangles(base, tri))
            # Blitting (see use_blit below) draws animated artists via
            # ax.draw_artist(), which - unlike a full fig.canvas.draw() -
            # does NOT trigger Axes3D's normal per-child do_3d_projection()
            # pass. Without calling it here explicitly, set_verts() above
            # would have no visible effect under blitting: the mesh would
            # render frozen at its last fully-drawn orientation forever.
            # Guarded on ax3d.M: that's the 3D projection matrix Axes3D
            # itself computes during its own first full draw - this draw()
            # function runs once (for the initial pose) before any full
            # draw has happened yet, when it's still None.
            if ax3d.M is not None:
                mesh.do_3d_projection()
            tilt_now = np.degrees(np.arccos(
                np.clip(np.dot(quat_rotmat(Q[ih]) @ [1, 0, 0], v0), -1, 1)))
            hud.set_text(f"T+{tf:5.2f} s\ntilt {tilt_now:3.0f} deg\n"
                         f"spin {gyro_mag[ih]:4.0f} deg/s"
                         + ("\n--- SHRED ---" if post else ""))
            artists += [mesh, hud]
        for c in cursors:
            c.set_xdata([tf, tf])
        return artists + cursors

    draw(win[0])

    # ---- record or show ----
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    if record:
        anim = FuncAnimation(fig, lambda i: draw(frame_times[i]),
                             frames=len(frame_times), interval=1000 / fps, blit=False)
        ext = os.path.splitext(record)[1].lower()
        if ext not in (".mp4", ".gif"):
            raise ValueError(
                f"Can't tell what format to save as: {record!r} needs to end in "
                f".mp4 or .gif."
            )
        if ext == ".gif":
            anim.save(record, writer="pillow", fps=fps)
        else:
            if not FFMpegWriter.isAvailable():
                # No system ffmpeg on PATH - fall back to the ffmpeg binary
                # bundled by imageio-ffmpeg (a core dependency), so MP4
                # export works out of the box without a system install.
                try:
                    import imageio_ffmpeg
                    import matplotlib
                    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
                except ImportError:
                    pass
            if not FFMpegWriter.isAvailable():
                raise RuntimeError(
                    f"Can't write {record}: no usable ffmpeg found (checked PATH and the "
                    f"bundled imageio-ffmpeg copy). Record to a .gif instead - no extra "
                    f"install needed for that."
                )
            anim.save(record, writer="ffmpeg", fps=fps, dpi=110)
        plt.close(fig)
        dur = len(frame_times) / fps
        print(f"Saved {record}  ({len(frame_times)} frames, {dur:.1f}s)")
        return record

    # interactive: slider + play/pause/step/restart, with keyboard shortcuts
    from matplotlib.widgets import Slider, Button
    sax = fig.add_axes([0.08, 0.02, 0.84, 0.022])
    rax = fig.add_axes([0.08, 0.065, 0.14, 0.035])
    lax = fig.add_axes([0.23, 0.065, 0.09, 0.035])
    pax = fig.add_axes([0.33, 0.065, 0.12, 0.035])
    nax = fig.add_axes([0.46, 0.065, 0.09, 0.035])
    sld = Slider(sax, "t (s)", win[0], win[1], valinit=win[0])
    # Slider.set_val() unconditionally triggers its own canvas.draw_idle()
    # (gated on `drawon`, not on `eventson` - setting eventson=False during
    # sync_slider() below only suppresses the on_changed callback, not this)
    # - a second, competing full redraw on every single frame update,
    # independent of and undoing the point of fast_redraw()'s blitting.
    # This widget's own artists are in anim_artists and get blitted there.
    sld.drawon = False
    btn_restart = Button(rax, "|<< Restart")
    btn_prev = Button(lax, "Step <")
    btn_play = Button(pax, "Play")
    btn_next = Button(nax, "Step >")

    n_frames = len(frame_times)
    state = {"playing": False, "anim": None, "i": 0}
    # Everything that needs to visually update every frame: the 3D mesh +
    # HUD text, the telemetry cursor lines, AND the slider's own bar/handle/
    # value-label. That last part matters: Slider.set_val() schedules its
    # own redraw via a DEFERRED canvas.draw_idle() call, entirely decoupled
    # from whatever draws the rest of the figure - during fast playback
    # those deferred redraws get coalesced by the GUI event loop and lag
    # behind everything else, which is exactly "the slider doesn't animate
    # with the rest of the program". Blitting the slider's own artists
    # ourselves, in the same immediate pass as the mesh/cursors, keeps it
    # in lockstep instead of relying on that separate deferred redraw.
    # btn_play.label is here too: its "Play"/"Pause" text changes on every
    # toggle, and with nothing else prompting a redraw of it, a blitted pass
    # that skipped it would leave the stale label visible indefinitely (the
    # cached background has the old text baked in).
    anim_artists = (([mesh, hud] if show_3d else []) + cursors
                    + [sld.poly, sld._handle, sld.valtext, btn_play.label])

    # A full (non-blit) redraw of this figure - 3D pane + 4 telemetry axes,
    # each with their own ticks/labels - is drastically slower than the
    # actual content change per frame would suggest (matplotlib recomputes
    # axis chrome, tick positions, and font metrics from scratch on every
    # full draw): ~1000ms/frame measured, regardless of mesh complexity,
    # vs <1ms/frame once blitting only redraws what actually changed. See
    # draw()'s explicit do_3d_projection() call above for why blitting is
    # safe here even with the 3D mesh. FuncAnimation has its own built-in
    # blit support, but it only covers its own timer-driven ticks - manual
    # interactions (drag the slider, click Step/Restart) go through a
    # separate code path, so blitting is managed by hand here and used
    # consistently for every kind of update, not just Play.
    use_blit = blit
    blit_bg = {"data": None}

    def capture_blit_background():
        if not use_blit:
            return
        for a in anim_artists:
            a.set_visible(False)
        fig.canvas.draw()
        blit_bg["data"] = fig.canvas.copy_from_bbox(fig.bbox)
        for a in anim_artists:
            a.set_visible(True)

    def fast_redraw():
        if use_blit and blit_bg["data"] is not None:
            fig.canvas.restore_region(blit_bg["data"])
            for a in anim_artists:
                a.axes.draw_artist(a)
            fig.canvas.blit(fig.bbox)
        else:
            fig.canvas.draw_idle()

    def camera_state():
        if not show_3d:
            return None
        return (ax3d.azim, ax3d.elev, ax3d.get_xlim3d(), ax3d.get_ylim3d(), ax3d.get_zlim3d())

    if use_blit:
        for a in anim_artists:
            a.set_animated(True)
        # The 3D axes' own mouse-drag rotate/pan/zoom (Axes3D._on_move) ends
        # with its own independent, full canvas.draw_idle() - it's not part
        # of, and knows nothing about, our blit setup. That full redraw
        # itself looks correct in the moment, but our cached background
        # snapshot is now stale (captured at the old camera angle): the very
        # next slider/play/step update would call fast_redraw(), which
        # restores that stale background and silently snaps the view back
        # to wherever it was before the user rotated it. Recapturing after
        # every mouse-button release fixes that (it covers the end of a
        # rotate/pan/zoom drag) - but a release also fires for every button/
        # slider click, which never touch the camera, so gate the (costly)
        # recapture on the camera actually having moved rather than paying
        # for a full redraw on every click.
        last_camera = {"state": camera_state()}

        def recapture_if_camera_moved(_evt):
            now = camera_state()
            if now != last_camera["state"]:
                last_camera["state"] = now
                capture_blit_background()

        fig.canvas.mpl_connect("resize_event", lambda evt: capture_blit_background())
        fig.canvas.mpl_connect("button_release_event", recapture_if_camera_moved)
        capture_blit_background()

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
        state["i"] = nearest(frame_times, val)
        draw(val)
        fast_redraw()
    sld.on_changed(on_slider)

    def step(_):
        # Paced to elapsed wall-clock time (scaled by `speed`), not a fixed
        # frame-index increment: if a frame took longer to render than its
        # nominal 1/fps slot, this jumps straight to where playback should
        # be *now* instead of drifting into slow motion.
        elapsed = time.perf_counter() - state["wall_t0"]
        target_t = state["data_t0"] + elapsed * speed
        if target_t >= win[1]:
            stop()
            goto(n_frames - 1)
            return
        i = nearest(frame_times, target_t)
        if i == state["i"]:
            return   # nothing new to draw yet
        state["i"] = i
        draw(frame_times[i])
        sync_slider(i)
        fast_redraw()

    def stop():
        if state["anim"] is not None:
            try:
                state["anim"].event_source.stop()
            except Exception:
                pass
            state["anim"] = None
        state["playing"] = False
        btn_play.label.set_text("Play")
        fast_redraw()

    def play():
        from matplotlib.animation import FuncAnimation
        if state["playing"]:
            return
        if state["i"] >= n_frames - 1:
            state["i"] = 0
        state["playing"] = True
        btn_play.label.set_text("Pause")
        state["wall_t0"] = time.perf_counter()
        state["data_t0"] = frame_times[state["i"]]
        # Poll at the intended frame rate, not faster: step() already only
        # does real work when the wall clock has actually advanced past the
        # next frame, so polling faster than fps buys nothing but wasted
        # ticks. blit=False here - the FuncAnimation timer only drives
        # *when* step() runs; fast_redraw() (called from inside step())
        # handles the actual blitting.
        interval_ms = 1000.0 / max(fps, 1)
        state["anim"] = FuncAnimation(fig, step, interval=interval_ms,
                                      blit=False, cache_frame_data=False)
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
