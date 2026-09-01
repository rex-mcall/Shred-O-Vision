"""Mesh loading and simplification shared by both rendering backends.

`load_obj` here is a minimal, dependency-free geometry-only reader (vertices +
triangulated faces) used by the matplotlib backend and as a no-material
fallback. It does not read `.mtl` materials or `vt` texture coordinates -
the PyVista backend gets real per-face textures for free by handing the
`.obj`/`.mtl` pair to VTK's own `vtkOBJImporter` instead of parsing them here.
"""

import os

import numpy as np


def load_obj(path):
    """Minimal OBJ loader: v / f, triangulates n-gons, resolves negative indices.

    Tokenizes on ANY whitespace rather than checking for a literal space at a
    fixed offset: real exporters do write `f\\t1 2 3`, and a space-only check
    silently drops every such face. When only part of a file uses tabs (one
    exporter, one component), the result is a model that loads with whole
    sections - a body tube, say - simply missing. Likewise `utf-8-sig`, so a
    leading BOM doesn't eat the first vertex and shift every face index by one.
    """
    verts, tris = [], []
    skipped = 0
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for ln in fh:
            parts = ln.split()
            if not parts or parts[0].startswith("#"):
                continue
            tag = parts[0]
            if tag == "v":
                if len(parts) >= 4:
                    try:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        skipped += 1
            elif tag == "f":
                idx = []
                for tok in parts[1:]:
                    try:
                        idx.append(int(tok.split("/")[0]))
                    except ValueError:
                        idx = []
                        break
                if len(idx) < 3:
                    skipped += 1
                    continue
                nv = len(verts)
                idx = [(nv + i if i < 0 else i - 1) for i in idx]   # to 0-based
                for k in range(1, len(idx) - 1):
                    tris.append([idx[0], idx[k], idx[k + 1]])
    if skipped:
        print(f"Note: skipped {skipped} malformed line(s) in {os.path.basename(path)}.")

    V = np.array(verts, float)
    F = np.array(tris, int)
    if len(V) == 0 or len(F) == 0:
        raise ValueError(
            f"No usable geometry found in {os.path.basename(path)} "
            f"({len(V)} vertices, {len(F)} faces). If this file opens fine in other "
            f"software, please report it - it likely uses an OBJ feature this "
            f"reader doesn't handle yet."
        )
    # Drop faces referencing vertices that don't exist rather than letting them
    # index out of bounds later (some exporters emit stray/global indices).
    ok = (F >= 0).all(axis=1) & (F < len(V)).all(axis=1)
    if not ok.all():
        print(f"Note: dropped {(~ok).sum()} face(s) with out-of-range vertex indices.")
        F = F[ok]
    return V, F


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


def detect_nose_axis(V):
    """Guess which local axis a model's nose points along, as a '+z'-style
    string for nose_vec().

    Every renderer here assumes the model's long axis is its roll axis, and
    misjudging it is visually catastrophic rather than subtle: the mesh gets
    rotated about the wrong axis, so a rocket that the telemetry says is
    flying straight up is drawn lying sideways, or - looking straight down
    the tube - as nothing but fins radiating around an almost invisible
    body. Different CAD tools and exporters disagree about which axis is
    "up", so assuming +z silently breaks any model that doesn't share that
    convention.

    Axis = the longest bounding-box dimension (a rocket is far longer than
    it is wide). Direction = whichever end is thinner, since the nose
    tapers and the fin/motor end flares.
    """
    V = np.asarray(V, float)
    span = V.max(0) - V.min(0)
    axis = int(np.argmax(span))
    if span[axis] <= 0:
        return "+z"

    lateral = [i for i in range(3) if i != axis]
    a = V[:, axis]
    lo, hi = a.min(), a.max()
    end = 0.15 * (hi - lo)

    def mean_radius(mask):
        if not mask.any():
            return 0.0
        pts = V[mask][:, lateral]
        return float(np.linalg.norm(pts - pts.mean(0), axis=1).mean())

    r_lo = mean_radius(a <= lo + end)
    r_hi = mean_radius(a >= hi - end)
    sign = "+" if r_hi <= r_lo else "-"
    return f"{sign}{'xyz'[axis]}"


MARKER_COLOR = GLYPH_COLORS["fin_marked"]


def angular_roll_marker(V, F, width_deg=16):
    """Boolean mask (len(F),) marking faces within a narrow angular wedge
    around the model's local long axis (+X, after the usual nose->+X
    alignment). A real imported OBJ has no per-part labels to build a
    glyph_face_colors()-style marker from (no "this face is the marked
    fin"), so this gives any mesh - however it's actually built - the same
    kind of roll-visibility marker: a stripe that runs the model's full
    length at a fixed angle in its own body frame, so it rotates rigidly
    with the mesh and stays visible as a spin reference regardless of what
    the geometry itself represents."""
    centroids = V[F].mean(axis=1)
    angle = np.degrees(np.arctan2(centroids[:, 2], centroids[:, 1]))
    return np.abs(angle) <= (width_deg / 2)


def solid_color_with_marker(base_color, marker_mask, marker_color=MARKER_COLOR):
    """RGB array (len(marker_mask), 3): base_color everywhere, marker_color
    on the faces angular_roll_marker() picked out."""
    colors = np.tile(np.asarray(base_color, dtype=float), (len(marker_mask), 1))
    colors[marker_mask] = marker_color
    return colors


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


def _cluster_at(V, F, res):
    """One vertex-clustering pass at grid resolution `res`. Snaps vertices to
    an res^3 grid over the bounding box, averages each cell, and rebuilds the
    faces (dropping any that collapsed to a degenerate sliver)."""
    lo, hi = V.min(0), V.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
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


def decimate_mesh(V, F, target_faces):
    """Vertex-clustering decimation: snaps vertices to a grid and rebuilds faces.
    Fast, dependency-free, and good enough for an attitude silhouette. matplotlib's
    software 3D cost scales with face count, so this is the main perf lever.

    Searches for the grid resolution that lands closest to `target_faces`
    without exceeding it. The previous one-shot analytic guess at `res`
    assumed faces thin out as res^3, which badly underestimates how many
    survive on real (thin-shelled, unevenly tessellated) models: asking for
    10000 faces on a 154k-face export actually returned 699 - a 14x
    overshoot that threw away most of the detail being paid for.
    """
    if target_faces is None or len(F) <= target_faces:
        return V, F

    # Grow res until we exceed the target, then binary-search the gap. Each
    # pass is only a few ms even on a ~150k-face mesh, and this runs once.
    lo_res, hi_res = 4, 4
    best = _cluster_at(V, F, lo_res)
    while hi_res < 1024:
        nxt = hi_res * 2
        cand = _cluster_at(V, F, nxt)
        if len(cand[1]) > target_faces:
            break
        lo_res, best, hi_res = nxt, cand, nxt
    else:
        return best

    hi_res = min(hi_res * 2, 1024)
    while lo_res + 1 < hi_res:
        mid = (lo_res + hi_res) // 2
        cand = _cluster_at(V, F, mid)
        if len(cand[1]) <= target_faces:
            lo_res, best = mid, cand
        else:
            hi_res = mid
    return best
