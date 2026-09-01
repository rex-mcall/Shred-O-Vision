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


# Default paint scheme for the fallback glyph. "fin_marked"/"stripe" are the
# roll-visibility marker: one fin plus a matching body stripe, so spin is
# visible even from a distance and even (see glyph_face_colors) through the
# post-shred highlight.
GLYPH_COLORS = {
    "body": (0.85, 0.85, 0.88),
    "nose": (0.30, 0.30, 0.34),
    "fin": (0.30, 0.30, 0.34),
    "fin_marked": (1.00, 0.80, 0.00),
    "stripe": (1.00, 0.80, 0.00),
}


def rocket_primitive(n=24, body=2.0, nose=1.0, r=0.35, fin_count=4,
                      fin_span=0.55, fin_root_chord=0.55, fin_tip_chord=0.22):
    """Fallback rocket glyph (nose along +Z) when no OBJ is supplied: a
    body/nose cone plus trapezoidal fins at the base, with one fin (and a
    matching body stripe) marked so roll/spin is visible during playback.

    Returns (V, F, parts) where parts[i] labels face i as one of "body",
    "nose", "stripe", "fin", or "fin_marked" - see glyph_face_colors()."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring_b = np.c_[r * np.cos(th), r * np.sin(th), np.full(n, -body / 2)]
    ring_t = np.c_[r * np.cos(th), r * np.sin(th), np.full(n, body / 2)]
    tip = np.array([[0, 0, body / 2 + nose]])
    base = np.array([[0, 0, -body / 2]])
    iT, iB = 2 * n, 2 * n + 1
    stripe_seg = 0   # body ring segment used as the roll-visibility stripe

    tris, parts = [], []
    for i in range(n):
        j = (i + 1) % n
        body_part = "stripe" if i == stripe_seg else "body"
        tris += [[i, j, n + j], [i, n + j, n + i]]      # body
        parts += [body_part, body_part]
        tris += [[n + i, n + j, iT]]                    # nose cone
        parts += ["nose"]
        tris += [[i, iB, j]]                            # base cap
        parts += ["body"]

    # Trapezoidal fins around the aft body: simple flat radial plates, one
    # per fin_count, swept-leading-edge / straight-trailing-edge silhouette.
    # The marked fin is aligned with the stripe segment above.
    z_base = -body / 2
    fin_thetas = np.linspace(0, 2 * np.pi, fin_count, endpoint=False) + th[stripe_seg]
    fin_verts = []
    for k, ang in enumerate(fin_thetas):
        c, s = np.cos(ang), np.sin(ang)
        root_le = [r * c, r * s, z_base + fin_root_chord]
        root_te = [r * c, r * s, z_base]
        tip_te = [(r + fin_span) * c, (r + fin_span) * s, z_base]
        tip_le = [(r + fin_span) * c, (r + fin_span) * s, z_base + fin_tip_chord]
        base_idx = 2 * n + 2 + len(fin_verts)
        fin_verts += [root_le, root_te, tip_te, tip_le]
        part = "fin_marked" if k == 0 else "fin"
        tris += [[base_idx, base_idx + 1, base_idx + 2],
                 [base_idx, base_idx + 2, base_idx + 3]]
        parts += [part, part]

    V = np.vstack([ring_b, ring_t, tip, base, np.array(fin_verts)])
    return V, np.array(tris, int), np.array(parts, dtype=object)


def glyph_face_colors(parts, highlight=None):
    """RGB array (len(parts), 3) for the glyph's paint scheme. If `highlight`
    is given (an RGB color), every face except the roll marker (the marked
    fin + stripe) uses that color instead - this is how the post-shred
    highlight can still leave the roll marker visible against it."""
    marker = GLYPH_COLORS["fin_marked"]
    out = []
    for p in parts:
        if p in ("fin_marked", "stripe"):
            out.append(marker)
        elif highlight is not None:
            out.append(highlight)
        else:
            out.append(GLYPH_COLORS[p])
    return np.array(out, dtype=float)


def shade_triangles(base_colors, tri_verts, light_dir=(0.35, -0.35, 0.87), ambient=0.35):
    """Simple per-face diffuse shading from each triangle's own normal.

    matplotlib's Poly3DCollection has a `shade=True` option, but it computes
    normals once at construction and never updates them - so as the mesh
    rotates during animation the "lit" side stays fixed to the object's
    *initial* pose instead of following the current one, and without any
    shading at all a mesh rendered in one flat color reads as a silhouette
    with no visible depth or edges, however detailed the underlying geometry
    actually is. This recomputes real per-face brightness from whatever
    (possibly already-rotated) triangle vertices are passed in, every call.

    base_colors: single RGB, or one RGB per face (len(tri_verts), 3).
    tri_verts: (n_faces, 3, 3) triangle vertex positions.
    Returns an (n_faces, 3) RGB array, clipped to [0, 1].
    """
    tri_verts = np.asarray(tri_verts, float)
    e1 = tri_verts[:, 1] - tri_verts[:, 0]
    e2 = tri_verts[:, 2] - tri_verts[:, 0]
    normals = np.cross(e1, e2)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norm == 0, 1, norm)

    light = np.asarray(light_dir, float)
    light = light / np.linalg.norm(light)
    brightness = np.clip(normals @ light, 0, 1) * (1 - ambient) + ambient

    base = np.asarray(base_colors, float)
    if base.ndim == 1:
        base = np.tile(base, (len(tri_verts), 1))
    return np.clip(base * brightness[:, None], 0, 1)


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
