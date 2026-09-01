"""Command-line entry point. Dispatches to the matplotlib (default) or
pyvista rendering backend. Run with no arguments (or --menu) to get a
plain-language guided wizard instead of flags - see menu.py."""

import argparse
import sys

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
                    "Omit all arguments for a guided menu (or pass --menu); "
                    "omit just a file argument to get a pop-up file picker for it.")
    ap.add_argument("hr_csv", nargs="?", default=ASK, help="HR CSV (omit -> dialog)")
    ap.add_argument("lr_csv", nargs="?", default=ASK,
                     help="LR CSV; 'none' = orientation-only, no telemetry dashboard "
                          "(omit -> dialog, Cancel there also skips it)")
    ap.add_argument("--menu", action="store_true",
                     help="run the guided, plain-language wizard instead of using flags")
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
    ap.add_argument("--max-faces", type=int, default=10000,
                    help="[matplotlib only] decimate the model to this many faces (-1 = no limit)")
    ap.add_argument("--dpi", type=int, default=100, help="[matplotlib only] figure DPI (lower = faster)")
    ap.add_argument("--no-blit", action="store_true",
                     help="[matplotlib only] disable blitting in 2D-only mode")
    return ap


def _run(hr_csv, lr_csv, obj, renderer, window, record, **matplotlib_only):
    if renderer == "pyvista":
        from . import render_pyvista as backend
        backend.visualize(hr_csv, obj, window=window, record=record,
                           **{k: v for k, v in matplotlib_only.items()
                              if k in ("pad", "model_nose", "fps", "speed")})
    else:
        from . import render_matplotlib as backend
        backend.visualize(hr_csv, lr_csv, obj, window=window, record=record, **matplotlib_only)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if not argv or "--menu" in argv:
        from .menu import run_menu
        choices = run_menu()
        _run(choices["hr_csv"], choices["lr_csv"], choices["obj"], choices["renderer"],
             choices["window"], choices["record"])
        return

    ap = build_parser()
    a = ap.parse_args(argv)

    lr_csv = _parse_optional(a.lr_csv)
    obj = _parse_optional(a.obj)
    window = _parse_window(a.window)

    _run(a.hr_csv, lr_csv, obj, a.renderer, window, a.record,
         pad=a.pad, model_nose=a.model_nose, fps=a.fps, speed=a.speed,
         show_3d=not a.no_3d, max_faces=(None if a.max_faces < 0 else a.max_faces),
         dpi=a.dpi, blit=not a.no_blit)


if __name__ == "__main__":
    main()
