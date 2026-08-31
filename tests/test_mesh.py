import os

import numpy as np

from blueraven_visualizer.mesh import load_obj, rocket_primitive, decimate_mesh

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
    V, F = rocket_primitive(n=12)
    assert V.shape[1] == 3
    assert F.shape[1] == 3
    assert F.min() >= 0 and F.max() < len(V)


def test_decimate_mesh_reduces_face_count():
    V, F = rocket_primitive(n=64)
    n0 = len(F)
    Vd, Fd = decimate_mesh(V, F, target_faces=n0 // 4)
    assert len(Fd) < n0
    assert Fd.min() >= 0 and Fd.max() < len(Vd)


def test_decimate_mesh_noop_when_under_target():
    V, F = rocket_primitive(n=8)
    Vd, Fd = decimate_mesh(V, F, target_faces=10_000)
    assert np.array_equal(V, Vd)
    assert np.array_equal(F, Fd)
