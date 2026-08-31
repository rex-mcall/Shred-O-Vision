"""PyVista/VTK rendering backend: textured, hardware-accelerated 3D orientation
playback for Blue Raven quaternion data.

Optional dependency (`pip install blueraven-visualizer[pyvista]`). Scope is
deliberately narrower than the matplotlib backend: this gives the
best-looking 3D orientation view - with real per-part textures, via VTK's own
OBJ+MTL importer, which is what the matplotlib backend's hand-rolled
`mesh.load_obj()` can't do (no vt/usemtl/mtllib support) - in its own window.
It does not embed the multi-panel telemetry dashboard; use --renderer
matplotlib (the default) for that.
"""

import os
import numpy as np

from .dialogs import ASK, resolve_files
from .io import load_blueraven
from .quaternion import quat_rotmat, align_rotation, nose_vec
from .mesh import rocket_primitive
from .events import nearest, detect_shred
from .report import build_report

try:
    import pyvista as pv
    import vtk
except ImportError as exc:  # pragma: no cover - exercised only when extra is missing
    raise ImportError(
        "The pyvista renderer needs the optional 'pyvista' extra: "
        "pip install blueraven-visualizer[pyvista]"
    ) from exc

BASE_COLOR = (0.78, 0.80, 0.90)
SHRED_COLOR = (0.85, 0.06, 0.10)


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

    importer = vtk.vtkOBJImporter()
    importer.SetFileName(obj_path)
    if mtl_path and os.path.exists(mtl_path):
        importer.SetFileNameMTL(mtl_path)
    importer.SetTexturePath(obj_dir)
    importer.SetRenderWindow(plotter.ren_win)
    importer.Update()
    return list(plotter.renderer.actors.values())


def visualize(hr_csv=ASK, obj=ASK, *, window=None, pad=1.5,
              model_nose="+z", upright_start=True, shred_highlight=True,
              fps=30, speed=1.0, record=None, off_screen=None):

    hr_csv, _, obj = resolve_files(hr_csv, None, obj)

    _, hc = load_blueraven(hr_csv)
    t_hr = hc("Flight_Time_(s)")
    Q = np.c_[hc("Quat_1"), hc("Quat_2"), hc("Quat_3"), hc("Quat_4")]
    Q = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)
    acc = np.c_[hc("Accel_X"), hc("Accel_Y"), hc("Accel_Z")]
    accel_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(np.c_[hc("Gyro_X"), hc("Gyro_Y"), hc("Gyro_Z")], axis=1)
    tShred, gPk, _ = detect_shred(t_hr, accel_mag)
    print(build_report(t_hr, accel_mag, gyro_mag))

    if window == "shred":
        win = (tShred - pad, tShred + pad)
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

    poly = base_points = None
    if not textured:
        V, F = rocket_primitive()
        V = V - V.mean(0)
        V = (align_rotation(nose_vec(model_nose), [1, 0, 0]) @ V.T).T
        faces = np.hstack([np.full((len(F), 1), 3), F]).astype(np.int64)
        poly = pv.PolyData(V, faces)
        actor = plotter.add_mesh(poly, color=BASE_COLOR, smooth_shading=True)
        actors = [actor]
        base_points = V.copy()
        center = np.zeros(3)
    else:
        bounds = np.array(plotter.bounds).reshape(3, 2)
        center = bounds.mean(axis=1)

    v0 = quat_rotmat(Q[nearest(t_hr, win[0])]) @ np.array([1.0, 0, 0])
    Rworld = align_rotation(v0, [0, 0, 1]) if upright_start else np.eye(3)

    plotter.camera_position = "iso"
    plotter.camera.zoom(1.2)
    plotter.add_text("", position="upper_left", font_size=12, name="hud")

    def apply_pose(tf):
        ih = nearest(t_hr, tf)
        R = Rworld @ quat_rotmat(Q[ih])
        post = (not np.isnan(tShred)) and tf >= tShred
        if textured:
            transform = _rotation_transform(R, center)
            for a in actors:
                a.SetUserTransform(transform)
        else:
            poly.points = (R @ base_points.T).T
            actors[0].GetProperty().SetColor(*(SHRED_COLOR if (post and shred_highlight) else BASE_COLOR))
        tilt_now = np.degrees(np.arccos(
            np.clip(np.dot(quat_rotmat(Q[ih]) @ [1, 0, 0], v0), -1, 1)))
        plotter.add_text(
            f"T+{tf:5.2f} s\ntilt {tilt_now:3.0f} deg\nspin {gyro_mag[ih]:4.0f} deg/s"
            + ("\nSHRED" if post else ""),
            position="upper_left", font_size=12, name="hud",
            color="red" if post else "black",
        )

    apply_pose(win[0])

    if record:
        frame_times = np.arange(win[0], win[1], speed / fps)
        ext = os.path.splitext(record)[1].lower()
        if ext == ".gif":
            plotter.open_gif(record, fps=fps)
        else:
            plotter.open_movie(record, framerate=fps)
        for tf in frame_times:
            apply_pose(tf)
            plotter.write_frame()
        plotter.close()
        print(f"Saved {record}  ({len(frame_times)} frames, {len(frame_times) / fps:.1f}s)")
        return record

    plotter.add_slider_widget(apply_pose, [win[0], win[1]], value=win[0], title="t (s)")
    plotter.show()
    return plotter
