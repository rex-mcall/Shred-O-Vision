import os

import numpy as np
import pytest

from blueraven_visualizer.mesh import (
    load_obj, rocket_primitive, decimate_mesh, glyph_face_colors, GLYPH_COLORS,
    shade_triangles, angular_roll_marker, solid_color_with_marker, MARKER_COLOR,
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


def _face_at(x, deg, r=1.0):
    """A tiny triangle whose centroid sits at angle `deg` around the local
    +X axis, at position x along it - for angular_roll_marker tests."""
    a = np.radians(deg)
    y, z = r * np.cos(a), r * np.sin(a)
    return [[x, y, z], [x + 0.01, y, z], [x, y + 0.001, z]]


def _mesh_from_faces(face_specs):
    V, F = [], []
    for spec in face_specs:
        tri = _face_at(*spec)
        base = len(V)
        V.extend(tri)
        F.append([base, base + 1, base + 2])
    return np.array(V), np.array(F)


def test_angular_roll_marker_selects_only_faces_near_zero_degrees():
    V, F = _mesh_from_faces([(0, 0), (0, 90), (0, 180), (0, -90)])
    mask = angular_roll_marker(V, F, width_deg=16)
    assert mask.tolist() == [True, False, False, False]


def test_angular_roll_marker_spans_the_full_length_regardless_of_x():
    """The marker is a stripe running the model's whole length, not a
    localized patch - angle is the only criterion, independent of position
    along the axis."""
    V, F = _mesh_from_faces([(-10, 0), (0, 0), (10, 0), (40, 0)])
    mask = angular_roll_marker(V, F, width_deg=16)
    assert mask.all()


def test_angular_roll_marker_respects_width():
    V, F = _mesh_from_faces([(0, 5), (0, 20)])
    assert angular_roll_marker(V, F, width_deg=16)[0]      # inside +-8 deg
    assert not angular_roll_marker(V, F, width_deg=16)[1]  # outside


def test_solid_color_with_marker_colors_only_marked_faces():
    mask = np.array([True, False, True])
    colors = solid_color_with_marker((0.5, 0.5, 0.5), mask)
    assert np.allclose(colors[0], MARKER_COLOR)
    assert np.allclose(colors[1], (0.5, 0.5, 0.5))
    assert np.allclose(colors[2], MARKER_COLOR)


# --- loader robustness: real exporters vary in ways a naive
# fixed-offset parser silently mangles (found by battery-testing formats) ---

def _write(tmp_path, text, name="m.obj"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8", newline="")
    return str(p)


def test_load_obj_accepts_tab_separated_face_lines(tmp_path):
    """Regression guard: `f\t1 2 3` is valid OBJ, but a parser checking for
    a literal space at index 1 drops every such face. When an exporter uses
    tabs for only part of a file, whole components (a body tube, say) go
    missing while the rest renders fine."""
    path = _write(tmp_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf\t1 2 3\n")
    V, F = load_obj(path)
    assert len(V) == 3 and len(F) == 1


def test_load_obj_handles_mixed_tab_and_space_faces(tmp_path):
    path = _write(tmp_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 2 0 0\nf\t1 2 3\nf 1 2 4\n")
    _, F = load_obj(path)
    assert len(F) == 2


def test_load_obj_strips_utf8_bom_without_eating_the_first_vertex(tmp_path):
    """A BOM made the first `v` line unparseable, silently shifting every
    subsequent face index by one and skewing the whole mesh."""
    path = _write(tmp_path, "﻿v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    V, F = load_obj(path)
    assert len(V) == 3
    assert F.max() < len(V)


def test_load_obj_raises_a_clear_error_when_nothing_loads(tmp_path):
    path = _write(tmp_path, "# just a comment\no thing\n")
    with pytest.raises(ValueError, match="No usable geometry"):
        load_obj(path)


def test_load_obj_drops_out_of_range_face_indices(tmp_path):
    path = _write(tmp_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nf 1 2 99\n")
    V, F = load_obj(path)
    assert len(F) == 1
    assert F.max() < len(V)


def test_decimate_mesh_lands_close_to_the_requested_face_count():
    """Regression guard: the old one-shot analytic guess at grid resolution
    assumed faces thin out as res^3, which is badly wrong for real thin-
    shelled models - asking for 10000 faces on a dense mesh returned ~700,
    throwing away most of the detail being paid for."""
    V, F, _ = rocket_primitive(n=160, fin_count=4)
    for target in (400, 1200):
        _, Fd = decimate_mesh(V, F, target)
        assert len(Fd) <= target
        assert len(Fd) >= 0.5 * target, (
            f"asked for {target} faces, got {len(Fd)} - decimation is overshooting"
        )
