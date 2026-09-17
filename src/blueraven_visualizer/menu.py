"""Guided, plain-language wizard for people who don't want to learn CLI
flags. Runs automatically when the program is started with no arguments
(e.g. double-clicking a shortcut, or VS Code's Run button), or explicitly
via `--menu`. Kept deliberately fast: it always opens the HR, LR, and OBJ
file pickers directly (Cancel skips the optional ones) rather than asking
a yes/no question first, then just a few numbered questions for everything
else.
"""

import importlib.util
import os

from .dialogs import pick_file

_RULE = "=" * 60


def _choice(question, options, default=1):
    """options: list of (label, value). Returns the chosen value. Blank
    input accepts the default."""
    print(f"\n{question}")
    for i, (label, _value) in enumerate(options, 1):
        marker = "  <- default" if i == default else ""
        print(f"  {i}. {label}{marker}")
    while True:
        raw = input(f"Enter a number [1-{len(options)}], or press Enter for the default: ").strip()
        if not raw:
            return options[default - 1][1]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        print(f"Please type a number between 1 and {len(options)}.")


def _text(question, default):
    raw = input(f"{question} [{default}]: ").strip()
    return raw or default


def run_menu():
    print(_RULE)
    print("  Blue Raven Visualizer - guided setup")
    print(_RULE)
    print("A file picker will open for each file below - Cancel skips an")
    print("optional one. Then answer a couple of quick questions.\n")

    print("Opening file picker: HIGH-RATE (HR) Blue Raven CSV...")
    hr_csv = pick_file("Select HIGH-RATE (HR) Blue Raven CSV",
                        [("CSV files", "*.csv"), ("All files", "*.*")])
    if not hr_csv:
        print("\nNo file was selected, so there's nothing to visualize. Exiting.")
        raise SystemExit(0)
    print(f"  Using: {hr_csv}")

    print("\nOpening file picker: LOW-RATE (LR) Blue Raven CSV "
          "(adds altitude/velocity/pyro charts - Cancel to skip)...")
    lr_csv = pick_file("Select LOW-RATE (LR) Blue Raven CSV",
                        [("CSV files", "*.csv"), ("All files", "*.*")], optional=True)
    print(f"  Using: {lr_csv}" if lr_csv else "  Skipped - orientation view only.")

    print("\nOpening file picker: 3D rocket model, .obj "
          "(Cancel to use a generic rocket shape)...")
    obj = pick_file("Select OBJ model", [("OBJ models", "*.obj"), ("All files", "*.*")],
                    optional=True)
    print(f"  Using: {obj}" if obj else "  Skipped - using the generic rocket shape.")

    if _choice("Compare this flight side by side with a second one?",
               [("No - just this flight", False),
                ("Yes - pick a second flight to compare (synced at liftoff)", True)]):
        return _compare_menu(hr_csv, lr_csv, obj)

    pyvista_available = importlib.util.find_spec("pyvista") is not None
    renderer_options = [("Standard - shows the flight-data charts (recommended)", "matplotlib")]
    if pyvista_available:
        renderer_options.append(
            ("Realistic 3D - real textures/colors on your model, no charts", "pyvista"))
    else:
        print("  (A realistic-textures mode is available too, but it needs an extra one-time")
        print("   install - run 'pip install blueraven-visualizer[pyvista]' to unlock it.)")
    renderer = _choice("Which display mode do you want?", renderer_options)

    window = _choice(
        "What part of the flight do you want to see?",
        [("Launch through apogee (recommended - skips the long descent)", None),
         ("The entire recording, including the descent", "full"),
         ("Zoom in on the peak-acceleration moment "
          "(max thrust on a normal flight; the break on a failure)", "peak")],
    )

    highlight_peak = _choice(
        "Mark the peak-acceleration moment by turning the rocket red for the "
        "rest of the flight? (Handy when investigating a breakup - on a normal "
        "flight that instant is just max thrust, so it's off by default.)",
        [("No", False), ("Yes", True)],
    )

    record = _ask_record()

    renderer_label = "Realistic 3D (textured)" if renderer == "pyvista" else "Standard (with charts)"

    print(f"\n{_RULE}")
    print("Starting with these choices:")
    print(f"  HR file:   {hr_csv}")
    print(f"  LR file:   {lr_csv or '(none)'}")
    print(f"  3D model:  {obj or '(generic rocket shape)'}")
    print(f"  Display:   {renderer_label}")
    window_label = {None: "launch through apogee", "full": "entire recording",
                    "peak": "peak-acceleration moment"}.get(window, str(window))
    print(f"  Window:    {window_label}")
    print(f"  Mark peak: {'yes' if highlight_peak else 'no'}")
    print(f"  Action:    {'save video to ' + record if record else 'watch interactively'}")
    print(f"{_RULE}\n")

    return {
        "hr_csv": hr_csv, "lr_csv": lr_csv, "obj": obj, "renderer": renderer,
        "window": window, "record": record, "highlight_peak": highlight_peak,
        "compare": None,
    }


def _ask_record():
    """Watch now, or pick a format and filename to export. Returns the output
    path, or None to watch interactively."""
    action = _choice(
        "What do you want to do with it?",
        [("Watch it now (play/pause/scrub on screen)", "watch"),
         ("Save a video file I can share", "export")],
    )
    if action != "export":
        return None
    ext = _choice("Which video format?", [("MP4 (recommended - plays everywhere)", "mp4"),
                                          ("GIF (bigger file, no player needed)", "gif")])
    record = _text("Output filename", f"blueraven_clip.{ext}")
    # Someone typing a custom name naturally won't always think to include the
    # extension - but without one, the exporter can't tell what format to
    # write. Force it to match what was actually chosen above.
    if not record.lower().endswith(f".{ext}"):
        record = os.path.splitext(record)[0] + f".{ext}"
    return record


def _compare_menu(hr_a, lr_a, obj_a):
    """Collect the second flight for a side-by-side comparison."""
    from .render_compare import SAME_AS_A

    csv_ft = [("CSV files", "*.csv"), ("All files", "*.*")]
    if not lr_a:
        print("\nComparing needs each flight's LOW-RATE (LR) file - the altitude and Mach")
        print("readouts come from it.")
        print("Opening file picker: LOW-RATE (LR) CSV for the FIRST flight...")
        lr_a = pick_file("Select LOW-RATE (LR) CSV for the FIRST flight", csv_ft)
        if not lr_a:
            print("\nNo LR file was selected, so there's nothing to compare. Exiting.")
            raise SystemExit(0)
        print(f"  Using: {lr_a}")

    print("\nOpening file picker: HIGH-RATE (HR) CSV for the SECOND flight...")
    hr_b = pick_file("Select HIGH-RATE (HR) CSV for the SECOND flight", csv_ft)
    if not hr_b:
        print("\nNo second flight was selected, so there's nothing to compare. Exiting.")
        raise SystemExit(0)
    print(f"  Using: {hr_b}")

    print("\nOpening file picker: LOW-RATE (LR) CSV for the SECOND flight...")
    lr_b = pick_file("Select LOW-RATE (LR) CSV for the SECOND flight", csv_ft)
    if not lr_b:
        print("\nNo LR file was selected for the second flight - altitude and Mach come")
        print("from it, so there's nothing to compare. Exiting.")
        raise SystemExit(0)
    print(f"  Using: {lr_b}")

    print("\nOpening file picker: 3D model for the SECOND flight, .obj "
          "(Cancel to use the same model as the first flight)...")
    obj_b = pick_file("Select OBJ model for the SECOND flight (Cancel = same as first flight)",
                      [("OBJ models", "*.obj"), ("All files", "*.*")], optional=True)
    print(f"  Using: {obj_b}" if obj_b else "  Using the same model as the first flight.")

    window = _choice(
        "What part of the flights do you want to see?",
        [("Launch through apogee (recommended - plays until the later of the two apogees)", None),
         ("The entire recordings, including the descents", "full")],
    )
    record = _ask_record()

    print(f"\n{_RULE}")
    print("Comparing two flights, synced at liftoff:")
    print(f"  Flight A:  {hr_a}")
    print(f"             {lr_a}")
    print(f"  Flight B:  {hr_b}")
    print(f"             {lr_b}")
    print(f"  Models:    A = {obj_a or '(generic rocket shape)'}")
    print(f"             B = {obj_b or 'same as A'}")
    print(f"  Window:    {'entire recordings' if window == 'full' else 'launch through apogee'}")
    print(f"  Action:    {'save video to ' + record if record else 'watch interactively'}")
    print(f"{_RULE}\n")

    return {
        "hr_csv": hr_a, "lr_csv": lr_a, "obj": obj_a, "renderer": "matplotlib",
        "window": window, "record": record, "highlight_peak": False,
        "compare": {"hr_csv": hr_b, "lr_csv": lr_b, "obj": obj_b or SAME_AS_A},
    }
