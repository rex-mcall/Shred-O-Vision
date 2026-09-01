import os

import numpy as np
import pytest

from blueraven_visualizer.mesh import (
    load_obj, rocket_primitive, decimate_mesh, glyph_face_colors, GLYPH_COLORS,
    shade_triangles,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_load_obj_quad_and_negative_index_triangle():
    V, F = load_obj(os.path.join(FIXTURES, "tiny.obj"))
    assert V.shape == (5, 3)
    # the quad triangulates to 2 tris, the negative-index face to 1 -> 3 total
    assert F.shape == (3, 3)
    assert F.min() >= 0 and F.max() < len(V)
    # the negative-index face (-3 -2 5) should resolve to vertices 2,3,4 (0-based)
    assert list(F[-1]) == [2, 3, 4]


def test_rocket_primitive_is_a_closed_triangle_mesh():
    V, F, parts = rocket_primitive(n=12)
    assert V.shape[1] == 3
    assert F.shape[1] == 3
    assert F.min() >= 0 and F.max() < len(V)
    assert len(parts) == len(F)


def test_rocket_primitive_has_fins_and_a_roll_marker():
    V, F, parts = rocket_primitive(n=24, fin_count=4)
    assert set(parts) == {"body", "nose", "stripe", "fin", "fin_marked"}
    # exactly one marked fin (2 triangles), the rest are plain fins
    assert list(parts).count("fin_marked") == 2
    assert list(parts).count("fin") == 2 * (4 - 1)


def test_rocket_primitive_fin_count_is_configurable():
    _, F, parts = rocket_primitive(fin_count=6)
    n_fin_tris = sum(1 for p in parts if p in ("fin", "fin_marked"))
    assert n_fin_tris == 2 * 6


def test_decimate_mesh_reduces_face_count():
    V, F, _ = rocket_primitive(n=64)
    n0 = len(F)
    Vd, Fd = decimate_mesh(V, F, target_faces=n0 // 4)
    assert len(Fd) < n0
    assert Fd.min() >= 0 and Fd.max() < len(Vd)


def test_decimate_mesh_noop_when_under_target():
    V, F, _ = rocket_primitive(n=8)
    Vd, Fd = decimate_mesh(V, F, target_faces=10_000)
    assert np.array_equal(V, Vd)
    assert np.array_equal(F, Fd)


def test_glyph_face_colors_matches_default_palette():
    _, F, parts = rocket_primitive(n=12)
    colors = glyph_face_colors(parts)
    assert colors.shape == (len(F), 3)
    for i, p in enumerate(parts):
        assert tuple(colors[i]) == GLYPH_COLORS[p]


def test_glyph_face_colors_highlight_preserves_roll_marker():
    _, F, parts = rocket_primitive(n=12)
    shred_red = (0.85, 0.06, 0.10)
    colors = glyph_face_colors(parts, highlight=shred_red)
    marker = GLYPH_COLORS["fin_marked"]
    for i, p in enumerate(parts):
        if p in ("fin_marked", "stripe"):
            assert tuple(colors[i]) == pytest.approx(marker)
        else:
            assert tuple(colors[i]) == pytest.approx(shred_red)


def test_shade_triangles_varies_brightness_by_orientation():
    """Regression guard: matplotlib's Poly3DCollection defaults to
    shade=False (flat, unlit color) - a real mesh rendered that way reads as
    a featureless silhouette with no visible depth or edges, however
    detailed the geometry, which is what made a real textured rocket model
    look like a plain rod instead of a rocket with fins."""
    # two triangles facing opposite directions (+Z normal vs -Z normal)
    tri = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
    ], dtype=float)
    colors = shade_triangles((0.5, 0.5, 0.5), tri, light_dir=(0, 0, 1))
    assert not np.allclose(colors[0], colors[1])


def test_shade_triangles_respects_ambient_floor():
    # a triangle facing directly away from the light shouldn't go fully black
    tri = np.array([[[0, 0, 0], [0, 1, 0], [1, 0, 0]]], dtype=float)   # normal ~ -Z
    colors = shade_triangles((1.0, 1.0, 1.0), tri, light_dir=(0, 0, 1), ambient=0.3)
    assert colors[0].min() >= 0.3 - 1e-8


def test_shade_triangles_preserves_per_face_base_colors():
    tri = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ], dtype=float)
    base = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    colors = shade_triangles(base, tri, light_dir=(0, 0, 1))
    # same geometry/light -> same brightness -> colors stay proportional to
    # (and distinguishable by) their own base hue, not averaged together
    assert colors[0][0] > colors[0][2]   # first face stays red-dominant
    assert colors[1][2] > colors[1][0]   # second face stays blue-dominant


def test_shade_triangles_output_clipped_to_valid_color_range():
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
    colors = shade_triangles((1.0, 1.0, 1.0), tri, light_dir=(0, 0, 1), ambient=1.5)
    assert colors.min() >= 0
    assert colors.max() <= 1
