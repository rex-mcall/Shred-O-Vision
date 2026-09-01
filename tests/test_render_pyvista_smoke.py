import os

import pytest

pv = pytest.importorskip("pyvista")

from blueraven_visualizer.render_pyvista import visualize  # noqa: E402

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
