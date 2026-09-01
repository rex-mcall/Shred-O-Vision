"""Guided, plain-language wizard for people who don't want to learn CLI
flags. Runs automatically when the program is started with no arguments
(e.g. double-clicking a shortcut, or VS Code's Run button), or explicitly
via `--menu`. Uses the same native file-picker dialogs as everywhere else -
just wrapped in numbered questions with sensible defaults instead of flags.
"""

import importlib.util

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
    print("Answer a few questions to get started - press Enter to accept")
    print("the default for anything you're not sure about.")

    print("\nStep 1 of 5: pick your flight-computer files.")
    print("A window will open - navigate to your Blue Raven CSV export.")
    input("Press Enter to choose the HIGH-RATE (HR) CSV file...")
    hr_csv = pick_file("Select HIGH-RATE (HR) Blue Raven CSV",
                        [("CSV files", "*.csv"), ("All files", "*.*")])
    if not hr_csv:
        print("\nNo file was selected, so there's nothing to visualize. Exiting.")
        raise SystemExit(0)
    print(f"  Using: {hr_csv}")

    want_lr = _choice(
        "Do you also have a matching LOW-RATE (LR) CSV file? It adds altitude, "
        "velocity, and pyro-continuity charts alongside the 3D view.",
        [("Yes, I have one", True), ("No, orientation view only", False)],
    )
    lr_csv = None
    if want_lr:
        input("Press Enter to choose the LOW-RATE (LR) CSV file...")
        lr_csv = pick_file("Select LOW-RATE (LR) Blue Raven CSV",
                            [("CSV files", "*.csv"), ("All files", "*.*")], optional=True)
        print(f"  Using: {lr_csv}" if lr_csv else "  Skipped - no LR file selected.")

    print("\nStep 2 of 5: 3D rocket model (optional).")
    want_obj = _choice(
        "Do you have a 3D model of your rocket (a .obj file)?",
        [("No, use a generic rocket shape", False), ("Yes, I have one", True)],
        default=1,
    )
    obj = None
    if want_obj:
        input("Press Enter to choose the .obj model file...")
        obj = pick_file("Select OBJ model", [("OBJ models", "*.obj"), ("All files", "*.*")],
                        optional=True)
        print(f"  Using: {obj}" if obj else "  Skipped - using the generic rocket shape.")

    print("\nStep 3 of 5: display mode.")
    pyvista_available = importlib.util.find_spec("pyvista") is not None
    renderer_options = [("Standard - shows the flight-data charts (recommended)", "matplotlib")]
    if pyvista_available:
        renderer_options.append(
            ("Realistic 3D - real textures/colors on your model, no charts", "pyvista"))
    else:
        print("  (A realistic-textures mode is available too, but it needs an extra one-time")
        print("   install - run 'pip install blueraven-visualizer[pyvista]' to unlock it.)")
    renderer = _choice("Which display mode do you want?", renderer_options)

    print("\nStep 4 of 5: what part of the flight?")
    window = _choice(
        "What part of the flight do you want to see?",
        [("Just the shred / failure moment (recommended)", "shred"),
         ("The entire flight", None)],
    )

    print("\nStep 5 of 5: watch it, or save a video?")
    action = _choice(
        "What do you want to do with it?",
        [("Watch it now (play/pause/scrub on screen)", "watch"),
         ("Save a video file I can share", "export")],
    )
    record = None
    if action == "export":
        ext = "mp4"
        fmt = _choice("Which video format?", [("MP4 (recommended - plays everywhere)", "mp4"),
                                              ("GIF (bigger file, no player needed)", "gif")])
        ext = fmt
        record = _text("Output filename", f"blueraven_clip.{ext}")

    renderer_label = "Realistic 3D (textured)" if renderer == "pyvista" else "Standard (with charts)"

    print(f"\n{_RULE}")
    print("Starting with these choices:")
    print(f"  HR file:   {hr_csv}")
    print(f"  LR file:   {lr_csv or '(none)'}")
    print(f"  3D model:  {obj or '(generic rocket shape)'}")
    print(f"  Display:   {renderer_label}")
    print(f"  Window:    {window or 'entire flight'}")
    print(f"  Action:    {'save video to ' + record if record else 'watch interactively'}")
    print(f"{_RULE}\n")

    return {
        "hr_csv": hr_csv, "lr_csv": lr_csv, "obj": obj, "renderer": renderer,
        "window": window, "record": record,
    }
