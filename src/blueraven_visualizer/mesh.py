"""Mesh loading and simplification shared by both rendering backends.

`load_obj` here is a minimal, dependency-free geometry-only reader (vertices +
triangulated faces) used by the matplotlib backend and as a no-material
fallback. It does not read `.mtl` materials or `vt` texture coordinates -
the PyVista backend gets real per-face textures for free by handing the
`.obj`/`.mtl` pair to VTK's own `vtkOBJImporter` instead of parsing them here.
"""

import numpy as np


def load_obj(path):
    """Minimal OBJ loader: v / f, triangulates n-gons, resolves negative indices."""
    verts, tris = [], []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln[0] == "#":
                continue
            if ln[0] == "v" and ln[1:2] in (" ", "\t"):
                p = ln.split()[1:4]
                if len(p) >= 3:
                    verts.append([float(p[0]), float(p[1]), float(p[2])])
            elif ln[0] == "f" and ln[1:2] == " ":
                idx = []
                for tok in ln.split()[1:]:
                    j = int(tok.split("/")[0])
                    idx.append(j)
                nv = len(verts)
                idx = [(nv + i if i < 0 else i - 1) for i in idx]   # to 0-based
                for k in range(1, len(idx) - 1):
                    tris.append([idx[0], idx[k], idx[k + 1]])
    return np.array(verts, float), np.array(tris, int)


def rocket_primitive(n=24, body=2.0, nose=1.0, r=0.35):
    """Fallback rocket glyph (nose along +Z) when no OBJ is supplied."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring_b = np.c_[r * np.cos(th), r * np.sin(th), np.full(n, -body / 2)]
    ring_t = np.c_[r * np.cos(th), r * np.sin(th), np.full(n, body / 2)]
    tip = np.array([[0, 0, body / 2 + nose]])
    base = np.array([[0, 0, -body / 2]])
    V = np.vstack([ring_b, ring_t, tip, base])
    iT, iB = 2 * n, 2 * n + 1
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [[i, j, n + j], [i, n + j, n + i]]      # body
        tris += [[n + i, n + j, iT]]                    # nose cone
        tris += [[i, iB, j]]                            # base cap
    return V, np.array(tris, int)


def decimate_mesh(V, F, target_faces):
    """Vertex-clustering decimation: snaps vertices to a grid and rebuilds faces.
    Fast, dependency-free, and good enough for an attitude silhouette. matplotlib's
    software 3D cost scales with face count, so this is the main perf lever."""
    if target_faces is None or len(F) <= target_faces:
        return V, F
    lo, hi = V.min(0), V.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    ratio = (target_faces / len(F)) ** (1 / 3)
    res = max(4, int(round((len(V) ** (1 / 3)) * ratio)))
    cell = np.clip(np.floor((V - lo) / span * res).astype(int), 0, res - 1)
    key = (cell[:, 0] * res + cell[:, 1]) * res + cell[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    newV = np.zeros((inv.max() + 1, 3))
    np.add.at(newV, inv, V)
    newV /= np.bincount(inv)[:, None]
    nf = inv[F]
    good = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf = np.unique(np.sort(nf[good], axis=1), axis=0)
    return newV, nf
