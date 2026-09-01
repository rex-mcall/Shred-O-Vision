"""PyVista/VTK rendering backend: textured, hardware-accelerated 3D orientation
playback for Blue Raven quaternion data.

Optional dependency (`pip install blueraven-visualizer[pyvista]`). Scope is
deliberately narrower than the matplotlib backend: this gives the
best-looking 3D orientation view - with real per-part textures, via VTK's own
OBJ+MTL importer, which is what the matplotlib backend's hand-rolled
`mesh.load_obj()` can't do (no vt/usemtl/mtllib support) - in its own window,
with wall-clock-paced play/pause (SPACE), step (arrow keys), and scrub
(slider). It does not embed the multi-panel telemetry dashboard; use
--renderer matplotlib (the default) for that.
"""

import math
import os
import re
import tempfile
import time
import numpy as np

from .dialogs import ASK, resolve_files
from .io import load_blueraven
from .quaternion import quat_rotmat, align_rotation, nose_vec, pose_rotation
from .mesh import (rocket_primitive, glyph_face_colors, detect_nose_axis,
                   warn_if_partial_model)
from .events import nearest, detect_peak_accel
from .report import build_report

try:
    import pyvista as pv
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError as exc:  # pragma: no cover - exercised only when extra is missing
    raise ImportError(
        "The pyvista renderer needs the optional 'pyvista' extra: "
        "pip install blueraven-visualizer[pyvista]"
    ) from exc

BASE_COLOR = (0.78, 0.80, 0.90)
HIGHLIGHT_COLOR = (0.85, 0.06, 0.10)


def _rotation_transform(R, center):
    """vtkTransform that rotates by 3x3 matrix R about world-space `center`."""
    m = vtk.vtkMatrix4x4()
    m.Identity()
    for i in range(3):
        for j in range(3):
            m.SetElement(i, j, R[i, j])
    t = vtk.vtkTransform()
    t.Translate(*center)
    t.Concatenate(m)
    t.Translate(*(-np.asarray(center)))
    return t


_CLAMP_OPTION = re.compile(r"-clamp\s+(on|off)\s*")


def _sanitize_obj_and_mtl(obj_path, mtl_path):
    """Write temp copies of obj+mtl that work around two confirmed
    vtkOBJImporter limitations, both hit in practice with real-world
    OpenRocket/CAD OBJ exports (not just a hypothetical edge case - this is
    what happens with an unmodified, real designer's export):

      - it mis-parses a map_Kd/map_Ka/etc.'s `-clamp on|off` option,
        corrupting the texture filename that follows it, so textures fail
        to load ("Could not open file ...-clamp off .../some_texture.png")
      - it fails to match usemtl/newmtl names against each other when
        they're long and/or contain punctuation like the parts list a CAD
        tool tends to name materials after (verified: renaming every
        material to a short mat_N id fixes this every time, with no
        visible difference since VTK only uses the name to link usemtl to
        newmtl, never displays it)

    Never touches the original files. Returns (obj_path, mtl_path) for the
    sanitized temp copies.
    """
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as fh:
        mtl_text = fh.read()

    name_map = {}
    counter = [0]

    def short_name(original):
        if original not in name_map:
            counter[0] += 1
            name_map[original] = f"mat_{counter[0]}"
        return name_map[original]

    def rename_mtl(m):
        original = m.group(1).strip()
        return f"# {original}\nnewmtl {short_name(original)}"

    mtl_text = re.sub(r"^newmtl[ \t]+(.+)$", rename_mtl, mtl_text, flags=re.MULTILINE)
    mtl_text = _CLAMP_OPTION.sub("", mtl_text)

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as fh:
        obj_text = fh.read()

    obj_text = re.sub(r"^usemtl[ \t]+(.+)$",
                      lambda m: f"usemtl {short_name(m.group(1).strip())}",
                      obj_text, flags=re.MULTILINE)
    obj_text = re.sub(r"^mtllib[ \t]+.+$", "mtllib model.mtl", obj_text, flags=re.MULTILINE)

    tmp_dir = tempfile.mkdtemp(prefix="blueraven_obj_")
    tmp_obj = os.path.join(tmp_dir, "model.obj")
    tmp_mtl = os.path.join(tmp_dir, "model.mtl")
    with open(tmp_obj, "w", encoding="utf-8") as fh:
        fh.write(obj_text)
    with open(tmp_mtl, "w", encoding="utf-8") as fh:
        fh.write(mtl_text)
    return tmp_obj, tmp_mtl


def _actor_points(actor):
    """Vertices of a VTK actor as an (N, 3) array, or None.

    vtkOBJImporter hands back raw VTK actors, not pyvista-wrapped ones, so
    the convenient `.mapper.dataset.points` accessor doesn't exist here -
    go through the VTK API instead.
    """
    mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
    dataset = mapper.GetInput() if mapper is not None else None
    points = dataset.GetPoints() if dataset is not None else None
    if points is None or points.GetNumberOfPoints() == 0:
        return None
    return vtk_to_numpy(points.GetData())


def _load_textured_actors(plotter, obj_path):
    """Import obj+mtl+textures via VTK's own OBJ importer."""
    obj_dir = os.path.dirname(os.path.abspath(obj_path))
    mtl_path = None
    with open(obj_path, "r", errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("mtllib"):
                mtl_name = ln.split(None, 1)[1].strip()
                mtl_path = os.path.join(obj_dir, mtl_name)
                break

    import_obj, import_mtl = obj_path, mtl_path
    if mtl_path and os.path.exists(mtl_path):
        import_obj, import_mtl = _sanitize_obj_and_mtl(obj_path, mtl_path)

    importer = vtk.vtkOBJImporter()
    importer.SetFileName(import_obj)
    if import_mtl and os.path.exists(import_mtl):
        importer.SetFileNameMTL(import_mtl)
    # Textures live alongside the ORIGINAL obj, not the sanitized temp copy -
    # only the material-name/`-clamp` text was rewritten, texture filenames
    # (as relative paths from here) are untouched.
    importer.SetTexturePath(obj_dir)
    importer.SetRenderWindow(plotter.ren_win)
    importer.Update()
    return list(plotter.renderer.actors.values())


def visualize(hr_csv=ASK, obj=ASK, *, window=None, pad=1.5,
              model_nose="auto", upright_start=True, highlight_peak=False,
              fps=30, speed=1.0, record=None, off_screen=None):

    hr_csv, _, obj = resolve_files(hr_csv, None, obj)

    _, hc = load_blueraven(hr_csv)
    t_hr = hc("Flight_Time_(s)")
    Q = np.c_[hc("Quat_1"), hc("Quat_2"), hc("Quat_3"), hc("Quat_4")]
    Q = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)
    acc = np.c_[hc("Accel_X"), hc("Accel_Y"), hc("Accel_Z")]
    accel_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(np.c_[hc("Gyro_X"), hc("Gyro_Y"), hc("Gyro_Z")], axis=1)
    t_peak_g, gPk, _ = detect_peak_accel(t_hr, accel_mag)
    print(build_report(t_hr, accel_mag, gyro_mag))

    if window in ("peak", "shred"):
        win = (t_peak_g - pad, t_peak_g + pad)
    elif window:
        win = tuple(window)
    else:
        win = (float(t_hr[0]), float(t_hr[-1]))
    win = (max(win[0], t_hr[0]), min(win[1], t_hr[-1]))

    if off_screen is None:
        off_screen = record is not None
    plotter = pv.Plotter(off_screen=off_screen, window_size=(1000, 800))
    plotter.set_background("white")

    textured = bool(obj)
    actors = []
    if textured:
        actors = _load_textured_actors(plotter, obj)
        textured = bool(actors)

    poly = base_points = face_colors_normal = face_colors_highlight = None
    if not textured:
        V, F, parts = rocket_primitive()
        V = V - V.mean(0)
        if model_nose == "auto":
            model_nose = detect_nose_axis(V)
        V = (align_rotation(nose_vec(model_nose), [1, 0, 0]) @ V.T).T
        faces = np.hstack([np.full((len(F), 1), 3), F]).astype(np.int64)
        poly = pv.PolyData(V, faces)
        face_colors_normal = (glyph_face_colors(parts) * 255).astype(np.uint8)
        face_colors_highlight = (glyph_face_colors(parts, highlight=HIGHLIGHT_COLOR) * 255).astype(np.uint8)
        poly.cell_data["colors"] = face_colors_normal
        # smooth_shading=True would make add_mesh bind the actor to an
        # internally-generated normals copy instead of this PolyData, so the
        # per-frame poly.points/cell_data mutations in apply_pose() below
        # would silently stop reaching the screen. Flat shading suits the
        # glyph's faceted panels anyway.
        actor = plotter.add_mesh(poly, scalars="colors", rgb=True, smooth_shading=False)
        actors = [actor]
        base_points = V.copy()
        center = np.zeros(3)
    else:
        bounds = np.array(plotter.bounds).reshape(3, 2)
        center = bounds.mean(axis=1)
        # Unlike the glyph above (and the matplotlib backend's load_obj
        # path), vtkOBJImporter loads the file's raw geometry as-is, in
        # whatever local axis the model's own nose actually points along -
        # for mmavenged.obj that's +Z, not +X. Without this, the per-frame
        # rotation below (which assumes body +X is the nose, matching the
        # Blue Raven quaternion convention) rotates the model around the
        # wrong axis entirely: it visibly tumbles ~90 deg off from where the
        # reported tilt/orientation actually is.
        if model_nose == "auto":
            # Detect from the imported geometry itself rather than assuming a
            # convention - exporters disagree about which axis is "up".
            pts = [p for p in (_actor_points(a) for a in actors) if p is not None]
            all_pts = np.vstack(pts) if pts else None
            model_nose = detect_nose_axis(all_pts) if all_pts is not None else "+z"
            print(f"Model's long axis detected as {model_nose} "
                  f"(override with --model-nose if the rocket looks mis-oriented).")
            if all_pts is not None:
                warn_if_partial_model(all_pts, model_nose)
        model_align = align_rotation(nose_vec(model_nose), [1, 0, 0])

    v0 = quat_rotmat(Q[nearest(t_hr, win[0])]) @ np.array([1.0, 0, 0])
    Rworld = align_rotation(v0, [0, 0, 1]) if upright_start else np.eye(3)

    # ---- camera: fixed, locked, side-on-with-elevation (matches the
    # matplotlib backend's view_init(elev=16, azim=-60) convention, so a
    # low-tilt rocket reads as "pointing up" on screen instead of the
    # generic "iso" corner view, which draws even a perfectly vertical
    # object as a diagonal line) ----
    bounds = np.array(plotter.bounds).reshape(3, 2)
    extent = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))
    elev, azim = math.radians(16), math.radians(-60)
    direction = np.array([math.cos(elev) * math.cos(azim),
                          math.cos(elev) * math.sin(azim),
                          math.sin(elev)])
    focal_point = tuple(center)
    cam_pos = tuple(np.asarray(center) + direction * extent * 1.3)
    plotter.camera_position = [cam_pos, focal_point, (0, 0, 1)]
    plotter.camera.zoom(0.85)   # headroom margin - better to show empty space than crop the model

    # ---- ground plane + fixed launch-vertical reference line, so "up" and
    # scale stay legible regardless of how the rocket itself is tumbling ----
    ground_z = bounds[2, 0] - extent * 0.05
    ground = pv.Plane(center=(center[0], center[1], ground_z), direction=(0, 0, 1),
                      i_size=extent * 1.4, j_size=extent * 1.4)
    plotter.add_mesh(ground, color=(0.88, 0.88, 0.88), show_edges=True,
                     edge_color=(0.75, 0.75, 0.75), lighting=False)
    ref_dir = Rworld @ v0
    ref_top = np.asarray(center) + ref_dir * extent * 0.65
    plotter.add_mesh(pv.Line(tuple(center), tuple(ref_top)), color=(0.6, 0.6, 0.6), line_width=2)

    # Lock the camera: mouse drag/scroll on a tumbling rocket makes it very
    # hard to tell rocket motion from camera motion, so the only view change
    # comes from the rocket's own logged orientation.
    plotter.iren.interactor.SetInteractorStyle(vtk.vtkInteractorStyleUser())

    plotter.add_text("", position="upper_left", font_size=12, name="hud")

    def apply_pose(tf):
        ih = nearest(t_hr, tf)
        post = highlight_peak and (not np.isnan(t_peak_g)) and tf >= t_peak_g
        if textured:
            R = pose_rotation(Rworld, Q[ih], model_align)
            transform = _rotation_transform(R, center)
            for a in actors:
                a.SetUserTransform(transform)
        else:
            # base_points was already pre-aligned (nose -> +X) at construction
            R = pose_rotation(Rworld, Q[ih])
            poly.points = (R @ base_points.T).T
            poly.cell_data["colors"] = (face_colors_highlight if post
                                        else face_colors_normal)
        tilt_now = np.degrees(np.arccos(
            np.clip(np.dot(quat_rotmat(Q[ih]) @ [1, 0, 0], v0), -1, 1)))
        plotter.add_text(
            f"T+{tf:5.2f} s\ntilt {tilt_now:3.0f} deg\nspin {gyro_mag[ih]:4.0f} deg/s"
            + ("\nPAST PEAK G" if post else ""),
            position="upper_left", font_size=12, name="hud",
            color="red" if post else "black",
        )

    apply_pose(win[0])

    if record:
        frame_times = np.arange(win[0], win[1], speed / fps)
        ext = os.path.splitext(record)[1].lower()
        if ext not in (".mp4", ".gif"):
            # Without a recognized extension, imageio can't tell what format
            # to write and fails deep inside write_frame() with a bare
            # `KeyError: None` - not helpful. This also protects against
            # ext == "" specifically, which open_movie() would otherwise
            # silently accept and only fail on the first actual frame write.
            raise ValueError(
                f"Can't tell what format to save as: {record!r} needs to end in "
                f".mp4 or .gif."
            )
        if ext == ".gif":
            plotter.open_gif(record, fps=fps)
        else:
            plotter.open_movie(record, framerate=fps)
        # A whole-flight export is easily ~1000 frames; without progress it
        # looks like the program has hung.
        n_total = len(frame_times)
        print(f"Rendering {n_total} frames to {record} ...")
        show_progress = n_total >= 100
        step = max(1, n_total // 10)
        for i, tf in enumerate(frame_times):
            apply_pose(tf)
            plotter.write_frame()
            if show_progress and i and (i % step == 0 or i == n_total - 1):
                print(f"  {i + 1}/{n_total} frames ({(i + 1) / n_total:.0%})", flush=True)
        plotter.close()
        print(f"Saved {record}  ({len(frame_times)} frames, {len(frame_times) / fps:.1f}s)")
        return record

    # ---- interactive: slider + spacebar play/pause + step/restart keys ----
    # Paced to elapsed wall-clock time (scaled by `speed`), same approach as
    # the matplotlib backend: playback tracks real time even if a given
    # frame's render takes longer than its nominal slot.
    state = {"playing": False, "t": win[0], "wall_t0": 0.0, "data_t0": win[0]}

    def set_pose(tf):
        tf = max(win[0], min(win[1], tf))
        apply_pose(tf)
        state["t"] = tf
        return tf

    def toggle_play():
        if state["playing"]:
            state["playing"] = False
        else:
            state["playing"] = True
            state["wall_t0"] = time.perf_counter()
            state["data_t0"] = win[0] if state["t"] >= win[1] else state["t"]

    def tick():
        if not state["playing"]:
            return
        elapsed = time.perf_counter() - state["wall_t0"]
        target_t = state["data_t0"] + elapsed * speed
        if target_t >= win[1]:
            state["playing"] = False
            set_pose(win[1])
            return
        set_pose(target_t)

    def step_frame(delta_t):
        state["playing"] = False
        set_pose(state["t"] + delta_t)

    def restart():
        state["playing"] = False
        set_pose(win[0])

    plotter.add_slider_widget(set_pose, [win[0], win[1]], value=win[0], title="t (s)")
    plotter.add_key_event("space", toggle_play)
    plotter.add_key_event("Right", lambda: step_frame(speed / fps))
    plotter.add_key_event("Left", lambda: step_frame(-speed / fps))
    plotter.add_key_event("r", restart)
    plotter.add_key_event("Home", restart)
    plotter.add_key_event("End", lambda: set_pose(win[1]))

    print("Controls: SPACE play/pause | drag slider to scrub | "
          "LEFT/RIGHT arrows step | R/Home restart | End = last frame")

    # NOTE: pyvista/VTK's add_timer_event() is unreliable for driving
    # animation - it's a long-open, unfixed upstream bug (the callback often
    # never fires at all: https://github.com/pyvista/pyvista/discussions/7654,
    # https://github.com/pyvista/pyvista/issues/6985), reproducible even with
    # pyvista's own official animation example. Driving the loop from Python
    # with interactive_update=True + Plotter.update() instead - not the
    # VTK-internal timer - is what actually works reliably.
    plotter.show(interactive_update=True, auto_close=False)
    while not plotter._closed:
        tick()
        plotter.update(stime=15, force_redraw=True)
    return plotter
