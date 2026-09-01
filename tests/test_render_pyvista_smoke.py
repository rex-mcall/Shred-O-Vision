import os

import pytest

pv = pytest.importorskip("pyvista")

from blueraven_visualizer.render_pyvista import visualize, _sanitize_obj_and_mtl  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
HR_SAMPLE = os.path.join(FIXTURES, "hr_sample.csv")

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "mothman_avenged")
EXAMPLE_HR = os.path.join(EXAMPLE_DIR, "BlRv_wvuer1_HR_06-17-2026_09_25_14.csv")
EXAMPLE_OBJ = os.path.join(EXAMPLE_DIR, "mmavenged.obj")


def test_visualize_builtin_glyph_gif(tmp_path):
    out = tmp_path / "out.gif"
    result = visualize(HR_SAMPLE, obj=None, window=[-1.0, 1.0], fps=5, record=str(out))
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_record_without_a_recognized_extension_raises_a_clear_error(tmp_path):
    """Regression guard: a filename with no/unrecognized extension (e.g. a
    user typing a name in the wizard without ".mp4") used to reach
    imageio's write_frame() and fail with a bare, unhelpful `KeyError: None`
    deep inside a third-party library."""
    out = tmp_path / "no_extension"
    with pytest.raises(ValueError, match=r"\.mp4 or \.gif"):
        visualize(HR_SAMPLE, obj=None, window=[-1.0, 1.0], fps=5, record=str(out))


def test_visualize_textured_example_obj(tmp_path):
    out = tmp_path / "textured.gif"
    result = visualize(EXAMPLE_HR, obj=EXAMPLE_OBJ, window="shred", pad=0.4,
                        fps=5, record=str(out))
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_interactive_setup_locks_camera_and_adds_ground_reference(monkeypatch):
    """Regression guard: mouse-driven camera rotation was disabled, and a
    ground plane + launch-vertical reference line were added, because a
    freely-orbitable camera with no spatial reference made it hard to tell
    rocket motion from camera motion. Drives visualize() through its real
    interactive-mode setup (everything up to plotter.show()) without
    actually blocking on a live window."""
    captured = {}
    orig_show = pv.Plotter.show

    def fake_show(self, *a, **k):
        captured["style"] = type(self.iren.interactor.GetInteractorStyle()).__name__
        captured["n_actors"] = len(self.renderer.actors)
        self._closed = True   # let visualize()'s manual update loop exit immediately
        return None

    monkeypatch.setattr(pv.Plotter, "show", fake_show)
    visualize(HR_SAMPLE, obj=None, window=[-1.0, 1.0], record=None)
    monkeypatch.setattr(pv.Plotter, "show", orig_show)

    assert captured["style"] == "vtkInteractorStyleUser"
    # rocket + ground plane + reference line + HUD text, at minimum
    assert captured["n_actors"] >= 4


def test_glyph_actor_stays_bound_to_its_live_polydata():
    """Regression guard: add_mesh(..., smooth_shading=True) makes pyvista
    bind the actor's mapper to an internally-generated normals copy instead
    of the PolyData we hold - so per-frame poly.points/cell_data mutations
    in apply_pose() would silently stop reaching the screen (found via a
    garbled/frozen render). The glyph must use flat shading to avoid this."""
    import numpy as np
    from blueraven_visualizer.mesh import rocket_primitive, glyph_face_colors

    V, F, parts = rocket_primitive()
    faces = np.hstack([np.full((len(F), 1), 3), F]).astype(np.int64)
    poly = pv.PolyData(V, faces)
    poly.cell_data["colors"] = (glyph_face_colors(parts) * 255).astype(np.uint8)

    plotter = pv.Plotter(off_screen=True)
    actor = plotter.add_mesh(poly, scalars="colors", rgb=True, smooth_shading=False)
    assert actor.mapper.dataset is poly
    plotter.close()


# --- OBJ/MTL sanitization: real-world OpenRocket/CAD-style exports (like
# the bundled example's *original*, unmodified source file) hit two
# confirmed vtkOBJImporter bugs; _sanitize_obj_and_mtl works around both. ---

def _write_obj_mtl(tmp_path, mtl_body, obj_material_lines):
    mtl_path = tmp_path / "model.mtl"
    mtl_path.write_text(mtl_body, encoding="utf-8")
    obj_path = tmp_path / "model.obj"
    obj_path.write_text(
        "mtllib model.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        + "\n".join(obj_material_lines) + "\n"
        "f 1 2 3\n",
        encoding="utf-8",
    )
    return str(obj_path), str(mtl_path)


def test_sanitize_strips_the_clamp_option_that_corrupts_texture_paths(tmp_path):
    obj_path, mtl_path = _write_obj_mtl(
        tmp_path,
        mtl_body=(
            "newmtl Part One\n"
            "map_Kd -o 0.0 0.0 0.0 -s 1.0 1.0 1.0 -clamp off textures/part.png\n"
        ),
        obj_material_lines=["usemtl Part One"],
    )
    _, sanitized_mtl = _sanitize_obj_and_mtl(obj_path, mtl_path)
    text = open(sanitized_mtl, encoding="utf-8").read()
    assert "-clamp" not in text
    assert "textures/part.png" in text   # the actual filename is untouched


def test_sanitize_shortens_long_material_names_consistently_across_files(tmp_path):
    long_name = "mat_10_Iris Ultra Compact Parachute [Cd 2.2 (22.0 oz) 128.2 in^3]"
    obj_path, mtl_path = _write_obj_mtl(
        tmp_path,
        mtl_body=f"newmtl {long_name}\nKd 1.0 1.0 1.0\n",
        obj_material_lines=[f"usemtl {long_name}"],
    )
    sanitized_obj, sanitized_mtl = _sanitize_obj_and_mtl(obj_path, mtl_path)
    obj_text = open(sanitized_obj, encoding="utf-8").read()
    mtl_text = open(sanitized_mtl, encoding="utf-8").read()

    import re
    obj_name = re.search(r"^usemtl (\S+)$", obj_text, re.MULTILINE).group(1)
    mtl_name = re.search(r"^newmtl (\S+)$", mtl_text, re.MULTILINE).group(1)
    assert obj_name == mtl_name   # same replacement name in both files
    assert len(obj_name) < 20     # actually shortened, not just re-quoted
    assert long_name in mtl_text  # original name preserved as a comment


def test_sanitize_gives_different_materials_different_short_names(tmp_path):
    obj_path, mtl_path = _write_obj_mtl(
        tmp_path,
        mtl_body="newmtl Booster\nKd 1 1 1\nnewmtl Booster Motor Casing\nKd 1 0 0\n",
        obj_material_lines=["usemtl Booster", "usemtl Booster Motor Casing"],
    )
    _, sanitized_mtl = _sanitize_obj_and_mtl(obj_path, mtl_path)
    import re
    names = re.findall(r"^newmtl (\S+)$", open(sanitized_mtl, encoding="utf-8").read(),
                       re.MULTILINE)
    assert len(names) == len(set(names)) == 2


def test_sanitize_never_touches_the_original_files(tmp_path):
    obj_path, mtl_path = _write_obj_mtl(
        tmp_path,
        mtl_body="newmtl A Very Long Original Name\nmap_Kd -clamp off tex.png\n",
        obj_material_lines=["usemtl A Very Long Original Name"],
    )
    orig_obj = open(obj_path, encoding="utf-8").read()
    orig_mtl = open(mtl_path, encoding="utf-8").read()
    sanitized_obj, sanitized_mtl = _sanitize_obj_and_mtl(obj_path, mtl_path)

    assert sanitized_obj != obj_path and sanitized_mtl != mtl_path
    assert open(obj_path, encoding="utf-8").read() == orig_obj
    assert open(mtl_path, encoding="utf-8").read() == orig_mtl
