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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .dialogs import ASK, resolve_files
from .io import load_blueraven
from .quaternion import quat_rotmat, align_rotation, nose_vec
from .mesh import (
    load_obj, rocket_primitive, decimate_mesh, glyph_face_colors, shade_triangles,
    angular_roll_marker, solid_color_with_marker, detect_nose_axis,
    warn_if_partial_model,
)
from .events import first_true_time, nearest, detect_peak_accel, detect_roll_axis
from .report import build_report
from .playback import export_video, run_interactive, validate_record_path

BASE_COLOR = (0.78, 0.80, 0.90)


def _resolve_roll_axis(roll_axis, t_hr, accel):
    """'auto' -> detected from the accelerometer; say so when it isn't the
    standard mounting, since that changes how the whole flight is drawn."""
    if roll_axis != "auto":
        return roll_axis
    detected = detect_roll_axis(t_hr, accel)
    if detected != "+x":
        print(f"Flight computer is mounted with body {detected} toward the nose "
              f"(detected from the accelerometer; the standard mounting is +x). "
              f"Override with --roll-axis if the rocket looks wrong.")
    return detected
HIGHLIGHT_COLOR = (0.85, 0.06, 0.10)


def visualize(hr_csv=ASK, lr_csv=ASK, obj=ASK, *, window=None, pad=1.5,
              model_nose="auto", roll_axis="auto", upright_start=True, highlight_peak=False,
              fps=30, speed=1.0, record=None, decim_plot=5,
              show_3d=True, max_faces="auto", dpi=100, blit=True):

    # ---- resolve files (pops dialogs for any left as ASK) ----
    hr_csv, lr_csv, obj = resolve_files(hr_csv, lr_csv, obj)
    has_lr = lr_csv is not None
    if record:
        validate_record_path(record)

    if max_faces == "auto":
        # Interactive playback redraws many times a second, so a detailed
        # real-world OBJ has to be decimated to stay responsive. A one-time
        # export isn't racing a live frame budget, so it gets a much higher
        # allowance - but not an unlimited one: now that the whole flight is
        # the default window, a long flight is ~1000 frames, and at fully
        # undecimated detail (~0.35 s/frame on the bundled example) that's
        # minutes of silent grinding. Bounded high enough that a typical
        # rocket OBJ passes through essentially untouched.
        max_faces = 50000 if record else 10000

    # ---- load ----
    _, hc = load_blueraven(hr_csv)

    t_hr = hc("Flight_Time_(s)")
    Q = np.c_[hc("Quat_1"), hc("Quat_2"), hc("Quat_3"), hc("Quat_4")]
    Q = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)
    acc = np.c_[hc("Accel_X"), hc("Accel_Y"), hc("Accel_Z")]
    accel_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(np.c_[hc("Gyro_X"), hc("Gyro_Y"), hc("Gyro_Z")], axis=1)

    t_peak_g, gPk, iPk = detect_peak_accel(t_hr, accel_mag)
    roll_axis = _resolve_roll_axis(roll_axis, t_hr, acc)
    nose_body = nose_vec(roll_axis)
    events = {"peak g": t_peak_g}

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
    t_start, t_end = float(t_win_src[0]), float(t_win_src[-1])
    t_apogee = events.get("apogee", np.nan)

    if window in ("peak", "shred"):
        win = (t_peak_g - pad, t_peak_g + pad)
    elif window == "full":
        win = (t_start, t_end)
    elif window:
        win = tuple(window)
    elif not np.isnan(t_apogee):
        # Default: the part of the flight that's actually worth watching -
        # pad through apogee. Everything after it is a long descent under
        # canopy (88 s of it on the bundled flight, most of the recording),
        # during which the orientation view has little to show. Pass
        # --window full for the whole thing.
        win = (t_start, t_apogee)
    else:
        win = (t_start, t_end)
    win = (max(win[0], t_start), min(win[1], t_end))

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
    if model_nose == "auto":
        model_nose = detect_nose_axis(V)
        if obj:
            print(f"Model's long axis detected as {model_nose} "
                  f"(override with --model-nose if the rocket looks mis-oriented).")
            warn_if_partial_model(V, model_nose)
    V = (align_rotation(nose_vec(model_nose), [1, 0, 0]) @ V.T).T   # nose -> +X
    Rmax = float(np.linalg.norm(V, axis=1).max())

    # A real imported OBJ has no per-part labels to build a roll marker from
    # the way the built-in glyph's parts do - so give it the same kind of
    # spin-visibility stripe based on angular position around the model's
    # own long axis instead, which works on any mesh regardless of what its
    # geometry actually represents.
    roll_marker = angular_roll_marker(V, F) if parts is None else None
    # Everything above is built nose-along-+X (the roll marker is measured
    # about +X). Now carry the model into the flight computer's frame so its
    # nose lies along whichever board axis actually points out the nose.
    V = (align_rotation([1, 0, 0], nose_body) @ V.T).T

    v0 = quat_rotmat(Q[nearest(t_hr, win[0])]) @ nose_body
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
    face_colors_normal = face_colors_highlight = None
    if parts is not None:
        face_colors_normal = glyph_face_colors(parts)
        face_colors_highlight = glyph_face_colors(parts, highlight=HIGHLIGHT_COLOR)
    elif roll_marker is not None:
        face_colors_normal = solid_color_with_marker(BASE_COLOR, roll_marker)
        face_colors_highlight = solid_color_with_marker(HIGHLIGHT_COLOR, roll_marker)

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
        axA.annotate(f"{gPk:.0f} g", (t_peak_g, gPk), textcoords="offset points",
                     xytext=(5, -2), color=HIGHLIGHT_COLOR, fontweight="bold")

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
        axA.annotate(f"{gPk:.0f} g", (t_peak_g, gPk), textcoords="offset points",
                     xytext=(5, -2), color=HIGHLIGHT_COLOR, fontweight="bold")

        axG.plot(t_hr[::decim_plot], gyro_mag[::decim_plot], color=(0.50, 0.15, 0.65), lw=0.7)
        axG.set_ylabel("|gyro| (deg/s)"); axG.set_title("Angular rate magnitude (500 Hz)")
        axG.set_xlabel("Flight time (s)")

    # event lines + moving cursor on every telemetry panel
    cursors = []
    for ax in tele:
        for te in events.values():
            if np.isnan(te):
                continue
            # One quiet marker style for every detected event. Peak-g used to
            # get a bold red line across all four panels, which reads as an
            # alarm on a nominal flight where that instant is just max thrust,
            # and crowded the traces it was drawn over. The peak is still
            # called out by value on the acceleration trace itself.
            ax.axvline(te, color="0.6", lw=0.7, ls=":", alpha=0.8)
        cursors.append(ax.axvline(win[0], color="k", lw=1.0))
        ax.set_xlim(*win)
    for ax in tele[:-1]:
        ax.tick_params(labelbottom=False)

    # ---- frame update ----
    frame_times = np.arange(win[0], win[1], speed / fps)
    if len(frame_times) == 0:
        raise ValueError(f"Playback window {win[0]:.2f}..{win[1]:.2f} s contains no frames "
                         f"- check --window against the flight's recorded time range.")

    def draw(tf):
        artists = []
        if show_3d:
            ih = nearest(t_hr, tf)
            Vk = (Rworld @ quat_rotmat(Q[ih]) @ V.T).T
            tri = Vk[F]
            mesh.set_verts(tri)
            post = highlight_peak and (not np.isnan(t_peak_g)) and tf >= t_peak_g
            if face_colors_normal is not None:
                base = face_colors_highlight if post else face_colors_normal
            else:
                base = HIGHLIGHT_COLOR if post else BASE_COLOR
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
                np.clip(np.dot(quat_rotmat(Q[ih]) @ nose_body, v0), -1, 1)))
            hud.set_text(f"T+{tf:5.2f} s\ntilt {tilt_now:3.0f} deg\n"
                         f"spin {gyro_mag[ih]:4.0f} deg/s"
                         + ("\n--- PAST PEAK G ---" if post else ""))
            artists += [mesh, hud]
        for c in cursors:
            c.set_xdata([tf, tf])
        return artists + cursors

    draw(win[0])

    # ---- record or show ----
    content_artists = ([mesh, hud] if show_3d else []) + cursors
    if record:
        return export_video(fig, frame_times, draw, content_artists, record, fps)
    return run_interactive(fig, frame_times, draw, content_artists,
                           t_end=win[1], fps=fps, speed=speed, blit=blit)
