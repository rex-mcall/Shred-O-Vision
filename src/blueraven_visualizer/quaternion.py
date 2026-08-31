"""Quaternion / rotation math shared by both rendering backends."""

import numpy as np


def quat_rotmat(q):
    """Active rotation matrix from quaternion [w x y z] (validated against the
    Blue Raven Tilt_Angle log; rocket long axis = body +X)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def align_rotation(a, b):
    """Rotation mapping unit vector a onto unit vector b."""
    a = np.asarray(a, float); a = a / np.linalg.norm(a)
    b = np.asarray(b, float); b = b / np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        ax = np.cross(a, [1, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0, 1, 0])
        ax = ax / np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def nose_vec(s):
    """Parse a nose-direction spec like '+x', '-z' into a unit vector."""
    s = str(s).lower().strip()
    sgn = -1 if s.startswith("-") else 1
    s = s.lstrip("+-")
    return sgn * np.array({"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}.get(s, [0, 0, 1]),
                          dtype=float)
