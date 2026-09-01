import numpy as np
import pytest

from blueraven_visualizer.quaternion import quat_rotmat, align_rotation, nose_vec, pose_rotation


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


def test_pose_rotation_without_model_align_matches_plain_composition():
    Rworld = align_rotation([1, 0, 0], [0, 0, 1])
    q = [1, 0, 0, 0]
    assert np.allclose(pose_rotation(Rworld, q), Rworld @ quat_rotmat(q))


def test_pose_rotation_applies_model_align_before_body_rotation():
    """Regression guard for the pyvista textured-OBJ bug: a raw/unaligned
    mesh whose own native nose axis isn't already body +X (e.g. an OBJ
    modeled nose-along-+Z, like the bundled mmavenged.obj) must have that
    axis mapped to +X *before* the logged body rotation is applied - not
    after, and not skipped. Without model_align, the model visibly rotates
    around the wrong axis (it did, in production, before this was found)."""
    q = [1, 0, 0, 0]   # identity attitude -> v0 = quat_rotmat(q) @ [1,0,0] = [1,0,0]
    Rworld = align_rotation([1, 0, 0], [0, 0, 1])   # maps that v0 to world +Z

    model_align = align_rotation(nose_vec("+z"), [1, 0, 0])   # native nose -> +X
    R = pose_rotation(Rworld, q, model_align)

    # The model's own raw nose vector (+Z, un-pre-aligned) must end up at
    # world +Z, exactly like the body-convention +X axis does for any mesh
    # that's already pre-aligned (e.g. the built-in glyph).
    assert np.allclose(R @ nose_vec("+z"), [0, 0, 1], atol=1e-8)

    # Applying Rworld @ quat_rotmat(q) alone (the pre-fix behavior) does NOT
    # satisfy this - proving model_align is load-bearing, not a no-op here.
    buggy_R = Rworld @ quat_rotmat(q)
    assert not np.allclose(buggy_R @ nose_vec("+z"), [0, 0, 1], atol=1e-8)
