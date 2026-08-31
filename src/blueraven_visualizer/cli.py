"""Command-line entry point. Dispatches to the matplotlib (default) or
pyvista rendering backend."""

import argparse

from .dialogs import ASK


def _parse_window(w):
    if w and w != "shred":
        return [float(x) for x in w.split(",")]
    return w


def _parse_optional(value, none_word="none"):
    """'none' (case-insensitive) means None; ASK passes through unchanged."""
    if value is ASK:
        return ASK
    if value is not None and str(value).strip().lower() == none_word:
        return None
    return value


def build_parser():
    ap = argparse.ArgumentParser(
        prog="blueraven-visualizer",
        description="Blue Raven orientation + telemetry dashboard. "
                    "Omit file arguments to get pop-up file pickers.")
    ap.add_argument("hr_csv", nargs="?", default=ASK, help="HR CSV (omit -> dialog)")
    ap.add_argument("lr_csv", nargs="?", default=ASK,
                     help="LR CSV; 'none' = orientation-only, no telemetry dashboard "
                          "(omit -> dialog, Cancel there also skips it)")
    ap.add_argument("--obj", default=ASK, help="OBJ model; 'none' = built-in glyph (omit -> dialog)")
    ap.add_argument("--renderer", choices=["matplotlib", "pyvista"], default="matplotlib",
                     help="matplotlib (default, no extra deps, full telemetry dashboard) or "
                          "pyvista (textured/hardware-accelerated 3D view only, needs the "
                          "'pyvista' extra)")
    ap.add_argument("--window", default=None, help="'shred' or 't0,t1'")
    ap.add_argument("--pad", type=float, default=1.5)
    ap.add_argument("--model-nose", default="+z")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--record", default=None, help="output .mp4/.gif (else interactive)")
    ap.add_argument("--no-3d", action="store_true",
                     help="[matplotlib only] telemetry only (scrubs instantly)")
    ap.add_argument("--max-faces", type=int, default=1500,
                    help="[matplotlib only] decimate the model to this many faces (-1 = no limit)")
    ap.add_argument("--dpi", type=int, default=100, help="[matplotlib only] figure DPI (lower = faster)")
    ap.add_argument("--no-blit", action="store_true",
                     help="[matplotlib only] disable blitting in 2D-only mode")
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)

    lr_csv = _parse_optional(a.lr_csv)
    obj = _parse_optional(a.obj)
    window = _parse_window(a.window)

    if a.renderer == "pyvista":
        from . import render_pyvista as backend
        backend.visualize(a.hr_csv, obj, window=window, pad=a.pad,
                           model_nose=a.model_nose, fps=a.fps, speed=a.speed,
                           record=a.record)
    else:
        from . import render_matplotlib as backend
        backend.visualize(a.hr_csv, lr_csv, obj, window=window, pad=a.pad,
                           model_nose=a.model_nose, fps=a.fps, speed=a.speed,
                           record=a.record, show_3d=not a.no_3d,
                           max_faces=(None if a.max_faces < 0 else a.max_faces),
                           dpi=a.dpi, blit=not a.no_blit)


if __name__ == "__main__":
    main()
