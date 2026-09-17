"""Two flights side by side, synced at liftoff.

Each flight gets its own spinning 3D rocket with a HUD showing that flight's
own log timestamp, current barometric altitude, and current Mach number. A
shared readout above both shows the synced clock - seconds since liftoff -
and one set of playback controls drives both rockets.

Syncing uses each LR file's own ``Liftoff`` flag rather than assuming its
clock starts at liftoff. Blue Raven exports do currently zero their clock
there, so today the per-flight log time matches the synced T+; relying on
the flag keeps the comparison correct for a log that doesn't.

Mach is shown only up to each flight's own ``Apogee`` flag, then blanked.
The Blue Raven's inertial velocity drifts once the airframe is tumbling or
under canopy - every flight checked reports Mach 2-4 at its final sample,
while landing - so a live Mach readout through the descent would present
that drift as flight performance. Blanking at the flight computer's own
apogee flag uses its judgement rather than a guess of ours. It will not catch
drift that begins mid-ascent after a breakup.
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .io import load_blueraven
from .quaternion import quat_rotmat, align_rotation, nose_vec
from .mesh import (
    load_obj, rocket_primitive, decimate_mesh, glyph_face_colors, shade_triangles,
    angular_roll_marker, solid_color_with_marker, detect_nose_axis,
    warn_if_partial_model,
)
from .events import first_true_time
from .atmosphere import mach_number
from .report import build_report
from .playback import export_video, run_interactive, validate_record_path, nearest_index

BASE_COLOR = (0.78, 0.80, 0.90)

# obj_b default: draw flight B with the same model as flight A. Distinct from
# None, which means "use the built-in glyph".
SAME_AS_A = object()

_BLRV_NAME = re.compile(
    r"BlRv_(?P<serial>[^ _]+)[ _](?:HR|LR)_(?P<date>\d{2}-\d{2}-\d{4})_"
    r"(?P<h>\d{2})_(?P<m>\d{2})_(?P<s>\d{2})",
    re.IGNORECASE,
)


def flight_label(path):
    """A short human label from a Blue Raven export filename, e.g.
    ``BlRv_wvuer1 HR_06-17-2026_09_25_14.csv`` -> ``wvuer1  06-17-2026 09:25``."""
    name = os.path.basename(path)
    m = _BLRV_NAME.search(name)
    if not m:
        return os.path.splitext(name)[0]
    return f"{m['serial']}  {m['date']} {m['h']}:{m['m']}"


class Flight:
    """One flight's data, with times available both as logged and synced."""

    def __init__(self, hr_csv, lr_csv, label=None):
        if lr_csv is None:
            raise ValueError(
                "Comparing flights needs each flight's LOW-RATE (LR) file too - the "
                "altitude and Mach readouts come from it."
            )
        self.label = label or flight_label(hr_csv)
        self.notes = []

        _, hc = load_blueraven(hr_csv)
        t_hr = hc("Flight_Time_(s)")
        Q = np.c_[hc("Quat_1"), hc("Quat_2"), hc("Quat_3"), hc("Quat_4")]
        ok = np.isfinite(t_hr) & np.isfinite(Q).all(axis=1)
        self.t_hr = t_hr[ok]
        Q = Q[ok]
        self.Q = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)
        acc = np.c_[hc("Accel_X"), hc("Accel_Y"), hc("Accel_Z")][ok]
        gyr = np.c_[hc("Gyro_X"), hc("Gyro_Y"), hc("Gyro_Z")][ok]
        self.accel_mag = np.linalg.norm(acc, axis=1)
        self.gyro_mag = np.linalg.norm(gyr, axis=1)

        _, lc = load_blueraven(lr_csv)
        self._lc = lc
        t_lr = lc("Flight_Time_(s)")
        ok_lr = np.isfinite(t_lr)
        self.t_lr = t_lr[ok_lr]
        self.alt = lc("Baro_Altitude_AGL_(feet)")[ok_lr]

        self.liftoff = first_true_time(self.t_lr, (lc("Liftoff") > 0.5)[ok_lr])
        if np.isnan(self.liftoff):
            self.liftoff = 0.0
            self.notes.append("no Liftoff flag in the LR file - assuming log time 0 is liftoff")
        self.apogee = first_true_time(self.t_lr, (lc("Apogee") > 0.5)[ok_lr])

        self.mach = np.full(len(self.t_lr), np.nan)
        cols = [lc(c, required=False) for c in
                ("Velocity_Up", "Velocity_DR", "Velocity_CR", "Temperature_(F)")]
        if all(c is not None for c in cols):
            self.mach = mach_number(*(c[ok_lr] for c in cols))
            if np.isnan(self.apogee):
                self.notes.append("no Apogee flag - Mach is shown for the whole flight, "
                                  "including the descent, where it drifts")
            else:
                self.mach[self.t_lr > self.apogee] = np.nan
        else:
            self.notes.append("velocity or temperature columns missing - Mach unavailable")

        if len(self.t_hr) == 0 or len(self.t_lr) == 0:
            raise ValueError(f"{self.label}: no usable samples in the HR or LR file.")

        # Span where BOTH files have data, in synced (since-liftoff) time.
        self.start = max(self.t_hr[0], self.t_lr[0]) - self.liftoff
        self.end = min(self.t_hr[-1], self.t_lr[-1]) - self.liftoff
        self.apogee_sync = self.apogee - self.liftoff if not np.isnan(self.apogee) else np.nan
        # Tolerance for "is this instant inside the data": half an LR sample.
        self._tol = 0.5 * float(np.median(np.diff(self.t_lr))) if len(self.t_lr) > 1 else 0.0

    def report(self):
        return build_report(self.t_hr, self.accel_mag, self.gyro_mag,
                            t_lr=self.t_lr, lc=self._lc)

    def sample(self, t_sync):
        """Everything the HUD shows at synced time ``t_sync``."""
        t_log = t_sync + self.liftoff
        before = t_sync < self.start - self._tol
        after = t_sync > self.end + self._tol
        il = nearest_index(self.t_lr, t_log)
        return {
            "t_log": t_log,
            "status": "before" if before else ("after" if after else "live"),
            "ih": nearest_index(self.t_hr, t_log),
            "alt": self.alt[il],
            "mach": self.mach[il],
            "past_apogee": (not np.isnan(self.apogee)) and t_log > self.apogee,
        }


def _row(label, value, note=""):
    return f"{label:<9}{value:>10}" + (f"  {note}" if note else "")


def hud_text(s):
    """HUD body for one flight at one instant (see Flight.sample)."""
    if s["status"] == "before":
        rows = [_row("log time", "--", "(no data yet)"), _row("ALTITUDE", "--"), _row("MACH", "--")]
    elif s["status"] == "after":
        rows = [_row("log time", f"{s['t_log']:.2f} s", "(end of data)"),
                _row("ALTITUDE", "--"), _row("MACH", "--")]
    else:
        alt = f"{s['alt']:,.0f} ft" if np.isfinite(s["alt"]) else "--"
        if np.isfinite(s["mach"]):
            mach = _row("MACH", f"{s['mach']:.2f}")
        else:
            mach = _row("MACH", "--", "(past apogee)" if s["past_apogee"] else "")
        rows = [_row("log time", f"{s['t_log']:.2f} s"), _row("ALTITUDE", alt), mach]
    return "\n".join(rows)


def sync_window(fa, fb, window):
    """Playback window in synced time (seconds since liftoff)."""
    start = min(fa.start, fb.start)
    end_all = max(fa.end, fb.end)
    if window in ("peak", "shred"):
        raise ValueError(
            "--window peak isn't available when comparing flights - each one peaks at a "
            "different moment. Use the default (launch through apogee), 'full', or an "
            "explicit 't0,t1' in seconds since liftoff."
        )
    if window == "full":
        win = (start, end_all)
    elif window:
        win = (float(window[0]), float(window[1]))
    else:
        # Launch through the LATER apogee, so both ascents play out in full.
        def ascent_end(f):
            return f.apogee_sync if not np.isnan(f.apogee_sync) else f.end
        win = (start, max(ascent_end(fa), ascent_end(fb)))
    return (max(win[0], start), min(win[1], end_all))


def _build_model(obj, max_faces, model_nose, label):
    parts = None
    if obj:
        V, F = load_obj(obj)
        n0 = len(F)
        V, F = decimate_mesh(V, F, max_faces)
        if len(F) < n0:
            print(f"[{label}] Mesh decimated {n0} -> {len(F)} faces.")
    else:
        V, F, parts = rocket_primitive()
    V = V - (V.min(0) + V.max(0)) / 2
    if model_nose == "auto":
        model_nose = detect_nose_axis(V)
        if obj:
            print(f"[{label}] Model's long axis detected as {model_nose} "
                  f"(override with --model-nose if the rocket looks mis-oriented).")
            warn_if_partial_model(V, model_nose)
    V = (align_rotation(nose_vec(model_nose), [1, 0, 0]) @ V.T).T
    if parts is not None:
        colors = glyph_face_colors(parts)
    else:
        colors = solid_color_with_marker(BASE_COLOR, angular_roll_marker(V, F))
    return {"V": V, "F": F, "colors": colors, "Rmax": float(np.linalg.norm(V, axis=1).max())}


def compare_flights(hr_a, lr_a, hr_b, lr_b, *, obj_a=None, obj_b=SAME_AS_A,
                    label_a=None, label_b=None, window=None, fps=30, speed=1.0,
                    record=None, model_nose="auto", upright_start=True,
                    max_faces="auto", dpi=100, blit=True):
    """Play two flights side by side, synced at liftoff.

    ``obj_b`` defaults to flight A's model; pass None for the built-in glyph.
    ``window`` is in seconds since liftoff: None (launch through the later
    apogee), ``"full"``, or ``(t0, t1)``.
    """
    if record:
        validate_record_path(record)
    if max_faces == "auto":
        # Two meshes share each frame's budget.
        max_faces = 25000 if record else 5000

    fa = Flight(hr_a, lr_a, label_a)
    fb = Flight(hr_b, lr_b, label_b)
    if fa.label == fb.label:
        fa.label, fb.label = f"{fa.label} (A)", f"{fb.label} (B)"
    for tag, f in (("A", fa), ("B", fb)):
        print(f"\n=== Flight {tag}: {f.label} ===")
        print(f.report())
        for note in f.notes:
            print(f"Note: {note}")

    win = sync_window(fa, fb, window)
    frame_times = np.arange(win[0], win[1], speed / fps)
    if len(frame_times) == 0:
        raise ValueError(f"Playback window {win[0]:.2f}..{win[1]:.2f} s since liftoff contains "
                         f"no frames - check --window against both flights' recordings.")

    model_a = _build_model(obj_a, max_faces, model_nose, fa.label)
    model_b = model_a if obj_b is SAME_AS_A else _build_model(obj_b, max_faces, model_nose, fb.label)

    fig = plt.figure(figsize=(14, 8), facecolor="white", dpi=dpi)
    gs = GridSpec(1, 2, wspace=0.04, left=0.02, right=0.98, top=0.88, bottom=0.15)
    title = fig.text(0.5, 0.965, "", ha="center", va="top", fontsize=15,
                     family="monospace", weight="bold")

    panels = []
    for k, (f, model) in enumerate(((fa, model_a), (fb, model_b))):
        ax = fig.add_subplot(gs[0, k], projection="3d")
        if k == 1:
            # Linked cameras: rotating one view rotates the other, so the two
            # rockets are always seen from the same angle.
            ax.shareview(panels[0]["ax"])
        V, F, Rmax = model["V"], model["F"], model["Rmax"]
        mesh = Poly3DCollection(V[F], facecolor=model["colors"], edgecolor="none")
        ax.add_collection3d(mesh)
        for setlim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setlim(-Rmax * 1.12, Rmax * 1.12)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (up)")
        ax.view_init(elev=16, azim=-60)
        ax.set_title(f"Flight {'AB'[k]}: {f.label}", fontsize=12)

        i0 = nearest_index(f.t_hr, win[0] + f.liftoff)
        v0 = quat_rotmat(f.Q[i0]) @ np.array([1.0, 0, 0])
        Rworld = align_rotation(v0, [0, 0, 1]) if upright_start else np.eye(3)
        up = Rworld @ v0 * Rmax * 1.05
        ax.plot([0, up[0]], [0, up[1]], [0, up[2]], "--", color="0.6", lw=1)

        hud = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, va="top",
                        fontsize=12, family="monospace")
        panels.append({"ax": ax, "flight": f, "model": model, "mesh": mesh,
                       "hud": hud, "Rworld": Rworld})

    def draw(ts):
        for p in panels:
            f, V, F = p["flight"], p["model"]["V"], p["model"]["F"]
            s = f.sample(ts)
            Vk = (p["Rworld"] @ quat_rotmat(f.Q[s["ih"]]) @ V.T).T
            tri = Vk[F]
            p["mesh"].set_verts(tri)
            p["mesh"].set_facecolor(shade_triangles(p["model"]["colors"], tri))
            # Blitted draws skip Axes3D's own per-child projection pass; without
            # this the mesh would stay frozen at its last fully-drawn pose. M is
            # None until the axes' first full draw.
            if p["ax"].M is not None:
                p["mesh"].do_3d_projection()
            p["hud"].set_text(hud_text(s))
        title.set_text(f"T+ {ts:6.2f} s since liftoff")

    draw(win[0])
    animated = [a for p in panels for a in (p["mesh"], p["hud"])] + [title]
    if record:
        return export_video(fig, frame_times, draw, animated, record, fps)
    return run_interactive(fig, frame_times, draw, animated, t_end=win[1], fps=fps,
                           speed=speed, blit=blit, slider_label="T+ (s)")
