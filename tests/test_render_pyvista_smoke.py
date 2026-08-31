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
