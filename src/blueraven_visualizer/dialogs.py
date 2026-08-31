"""File-picker helpers shared by both rendering backends.

ASK is a sentinel: a parameter left as ASK triggers a file-picker dialog. This
is distinct from None, which for `obj`/`lr_csv` explicitly means "skip it".
"""

ASK = object()


def pick_file(title, filetypes, optional=False):
    """Pop a native open-file dialog. Falls back to a console prompt if no GUI
    (e.g. headless). Returns a path string, or None if cancelled/optional."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.update()
        root.destroy()
        return path or None
    except Exception:
        prompt = f"{title}\n  path{' (blank to skip)' if optional else ''}: "
        ans = input(prompt).strip().strip('"').strip("'")
        return ans or None


def resolve_files(hr_csv, lr_csv, obj):
    """Turn ASK sentinels into user-chosen paths via dialogs.

    hr_csv is required (SystemExit if left unresolved). lr_csv and obj are
    both optional - Cancel on either dialog (or passing None outright) means
    "run without it".
    """
    csv_ft = [("CSV files", "*.csv"), ("All files", "*.*")]
    if hr_csv is ASK:
        hr_csv = pick_file("Select HIGH-RATE (HR) Blue Raven CSV", csv_ft)
    if not hr_csv:
        raise SystemExit("No HR CSV selected - aborting.")
    if lr_csv is ASK:
        lr_csv = pick_file("Select LOW-RATE (LR) Blue Raven CSV  (Cancel = orientation only, "
                            "no telemetry dashboard)", csv_ft, optional=True)
    if obj is ASK:
        obj = pick_file("Select OBJ model  (Cancel = built-in rocket glyph)",
                        [("OBJ models", "*.obj"), ("All files", "*.*")], optional=True)
    return hr_csv, lr_csv, obj
