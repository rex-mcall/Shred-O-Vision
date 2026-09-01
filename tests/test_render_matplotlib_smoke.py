import os

import pytest

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


def test_record_without_a_recognized_extension_raises_a_clear_error(tmp_path):
    """Same regression guard as the pyvista backend: a filename with no/
    unrecognized extension should fail fast with a clear message instead of
    an obscure error partway through matplotlib's animation writer."""
    out = tmp_path / "no_extension"
    with pytest.raises(ValueError, match=r"\.mp4 or \.gif"):
        visualize(HR_SAMPLE, LR_SAMPLE, obj=None, window=[-1.0, 1.0],
                  fps=8, record=str(out))


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


def _decimated_face_count(output):
    """Parse 'Mesh decimated N -> M faces ...' out of captured stdout."""
    for line in output.splitlines():
        if line.startswith("Mesh decimated"):
            return int(line.split("->")[1].split()[0])
    return None


def test_max_faces_auto_gives_exports_far_more_detail_than_interactive(capsys, tmp_path):
    """Regression guard: interactive playback needs aggressive decimation to
    redraw fast, but a one-time export isn't racing a live frame budget - it
    should keep far more geometry so fins/panel seams survive. Bounded, not
    unlimited: whole-flight is the default window now, so an unbounded
    export means ~1000 frames of very slow rendering."""
    out = tmp_path / "detail.gif"
    visualize(EXAMPLE_HR, EXAMPLE_LR, obj=EXAMPLE_OBJ, window="peak", pad=0.2,
              fps=4, record=str(out))
    export_faces = _decimated_face_count(capsys.readouterr().out)

    visualize(EXAMPLE_HR, EXAMPLE_LR, obj=EXAMPLE_OBJ, window=[6.0, 6.5],
              fps=8, record=None)
    interactive_faces = _decimated_face_count(capsys.readouterr().out)

    assert interactive_faces is not None, "interactive playback should still decimate"
    assert export_faces is None or export_faces > interactive_faces * 3, (
        f"export kept {export_faces} faces vs interactive {interactive_faces} - "
        f"exports should keep substantially more detail"
    )
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
