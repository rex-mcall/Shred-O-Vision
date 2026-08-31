import numpy as np
import pytest

from blueraven_visualizer.quaternion import quat_rotmat, align_rotation, nose_vec


def test_quat_rotmat_identity():
    R = quat_rotmat([1, 0, 0, 0])
    assert np.allclose(R, np.eye(3))


def test_quat_rotmat_90deg_about_z():
    half = np.pi / 4
    q = [np.cos(half), 0, 0, np.sin(half)]
    R = quat_rotmat(q)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-8)


@pytest.mark.parametrize("q", [
    [1, 0, 0, 0],
    [0.7071, 0.7071, 0, 0],
    [0.5, 0.5, 0.5, 0.5],
    [0.1, 0.2, 0.3, 0.9],
])
def test_quat_rotmat_orthonormal(q):
    R = quat_rotmat(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)


def test_align_rotation_identity_when_already_aligned():
    R = align_rotation([1, 0, 0], [1, 0, 0])
    assert np.allclose(R, np.eye(3))


def test_align_rotation_maps_a_onto_b():
    R = align_rotation([1, 0, 0], [0, 1, 0])
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-8)


def test_align_rotation_antiparallel_edge_case():
    R = align_rotation([1, 0, 0], [-1, 0, 0])
    assert np.allclose(R @ [1, 0, 0], [-1, 0, 0], atol=1e-8)
    # must be a proper rotation, not a reflection
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)


@pytest.mark.parametrize("spec,expected", [
    ("+z", [0, 0, 1]),
    ("z", [0, 0, 1]),
    ("-x", [-1, 0, 0]),
    ("y", [0, 1, 0]),
])
def test_nose_vec(spec, expected):
    assert np.allclose(nose_vec(spec), expected)
