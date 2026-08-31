import os

from blueraven_visualizer.render_matplotlib import visualize

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
HR_SAMPLE = os.path.join(FIXTURES, "hr_sample.csv")
LR_SAMPLE = os.path.join(FIXTURES, "lr_sample.csv")

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "mothman_avenged")
EXAMPLE_HR = os.path.join(EXAMPLE_DIR, "BlRv_wvuer1_HR_06-17-2026_09_25_14.csv")
EXAMPLE_LR = os.path.join(EXAMPLE_DIR, "BlRv_wvuer1_LR_06-17-2026_09_25_14.csv")
EXAMPLE_OBJ = os.path.join(EXAMPLE_DIR, "mmavenged.obj")


def test_visualize_full_dashboard_gif(tmp_path):
    out = tmp_path / "out.gif"
    result = visualize(HR_SAMPLE, LR_SAMPLE, obj=None, window=[-1.0, 1.0],
                        fps=8, record=str(out), show_3d=True)
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_visualize_lr_optional(tmp_path):
    out = tmp_path / "orientation_only.gif"
    result = visualize(HR_SAMPLE, None, obj=None, window=[-1.0, 1.0],
                        fps=8, record=str(out), show_3d=True)
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_visualize_no_3d(tmp_path):
    out = tmp_path / "telemetry_only.gif"
    result = visualize(HR_SAMPLE, LR_SAMPLE, obj=None, window=[-1.0, 1.0],
                        fps=8, record=str(out), show_3d=False)
    assert out.exists()


def test_visualize_bundled_example_with_real_textured_obj(tmp_path):
    """End-to-end against the exact files shipped in examples/ - catches
    problems in the bundled data itself (e.g. a mangled OBJ/MTL path fix),
    not just in the code."""
    out = tmp_path / "example.gif"
    result = visualize(EXAMPLE_HR, EXAMPLE_LR, obj=EXAMPLE_OBJ, window="shred",
                        pad=0.5, fps=6, record=str(out), max_faces=800)
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0
