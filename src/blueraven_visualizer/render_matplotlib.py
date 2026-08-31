"""matplotlib rendering backend.

Zero extra dependencies beyond numpy/pandas/matplotlib. Draws the OBJ (or a
built-in glyph, untextured either way) spinning in a 3D panel next to a
synchronized telemetry dashboard, with a moving time cursor. Works headless
(Agg backend) for MP4/GIF export, or interactively with a scrub slider.

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
from .mesh import load_obj, rocket_primitive, decimate_mesh
from .events import first_true_time, nearest, detect_shred

BASE_COLOR = (0.78, 0.80, 0.90)
SHRED_COLOR = (0.85, 0.06, 0.10)


def visualize(hr_csv=ASK, lr_csv=ASK, obj=ASK, *, window=None, pad=1.5,
              model_nose="+z", upright_start=True, shred_highlight=True,
              fps=30, speed=1.0, record=None, decim_plot=5,
              show_3d=True, max_faces=1500, dpi=100, blit=True):

    # ---- resolve files (pops dialogs for any left as ASK) ----
    hr_csv, lr_csv, obj = resolve_files(hr_csv, lr_csv, obj)
    has_lr = lr_csv is not None

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
    print(f"Shred (peak accel): T+{tShred:.2f}s, {gPk:.0f} g")

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
    if obj:
        V, F = load_obj(obj)
    else:
        V, F = rocket_primitive()
    n0 = len(F)
    V, F = decimate_mesh(V, F, max_faces)
    if len(F) < n0:
        print(f"Mesh decimated {n0} -> {len(F)} faces for smooth playback "
              f"(raise max_faces, or set max_faces=None for MP4 export).")
    V = V - V.mean(0)
    V = (align_rotation(nose_vec(model_nose), [1, 0, 0]) @ V.T).T   # nose -> +X
    Rmax = float(np.linalg.norm(V, axis=1).max())

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
    mesh = hud = None
    if show_3d:
        mesh = Poly3DCollection(V[F], facecolor=BASE_COLOR, edgecolor="none")
        ax3d.add_collection3d(mesh)
        for setlim in (ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim):
            setlim(-Rmax * 1.5, Rmax * 1.5)
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z (up)")
        ax3d.view_init(elev=16, azim=-60)
        up = Rworld @ v0 * Rmax * 1.4
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
            mesh.set_verts(Vk[F])
            post = (not np.isnan(tShred)) and tf >= tShred
            mesh.set_facecolor(SHRED_COLOR if (post and shred_highlight) else BASE_COLOR)
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
        if ext == ".gif":
            anim.save(record, writer="pillow", fps=fps)
        else:
            if not FFMpegWriter.isAvailable():
                raise RuntimeError(
                    f"Can't write {record}: this needs the 'ffmpeg' binary on your PATH "
                    f"(matplotlib shells out to it for MP4 export), and it wasn't found. "
                    f"Either install ffmpeg, or record to a .gif instead - no extra "
                    f"install needed for that."
                )
            anim.save(record, writer="ffmpeg", fps=fps, dpi=110)
        plt.close(fig)
        dur = len(frame_times) / fps
        print(f"Saved {record}  ({len(frame_times)} frames, {dur:.1f}s)")
        return record

    # interactive: slider + play/pause
    from matplotlib.widgets import Slider, Button
    sax = fig.add_axes([0.08, 0.06, 0.72, 0.022])
    bax = fig.add_axes([0.83, 0.05, 0.10, 0.04])
    sld = Slider(sax, "t (s)", win[0], win[1], valinit=win[0])
    btn = Button(bax, "Play")
    state = {"playing": False, "anim": None, "i": 0}
    anim_artists = ([mesh, hud] if show_3d else []) + cursors
    # blitting on a 3D axis can fail to reproject the mesh; only blit in 2D-only mode
    use_blit = blit and not show_3d

    def on_slider(val):
        draw(val)
        fig.canvas.draw_idle()
    sld.on_changed(on_slider)

    def step(_):
        i = state["i"]
        if i >= len(frame_times):
            stop()
            return anim_artists
        state["i"] = i + 1
        return draw(frame_times[i])

    def stop():
        if state["anim"] is not None:
            try:
                state["anim"].event_source.stop()
            except Exception:
                pass
            state["anim"] = None
        if use_blit:
            for a in anim_artists:
                a.set_animated(False)
        state["playing"] = False
        btn.label.set_text("Play")
        j = min(state["i"], len(frame_times) - 1)
        draw(frame_times[j])
        sld.eventson = False
        sld.set_val(frame_times[j])
        sld.eventson = True
        fig.canvas.draw_idle()

    def toggle(_):
        from matplotlib.animation import FuncAnimation
        if state["playing"]:
            stop()
            return
        state["playing"] = True
        btn.label.set_text("Pause")
        state["i"] = nearest(frame_times, sld.val)
        if use_blit:
            for a in anim_artists:
                a.set_animated(True)
        state["anim"] = FuncAnimation(fig, step, interval=1000 / fps,
                                      blit=use_blit, cache_frame_data=False)
        fig.canvas.draw_idle()
    btn.on_clicked(toggle)

    print("Controls: drag slider to scrub, Play/Pause to animate.")
    plt.show()
    return fig
