"""Convenience entry point for VS Code's Run button (or `python run.py`
from anywhere). Just click Run.

blueraven_visualizer.cli uses relative imports, so it can't be run directly
as a script (VS Code's Run button does `python <file>`, not `python -m
package.module`) - this thin wrapper uses a plain absolute import instead,
which works either way as long as the package is installed (`pip install -e
.`) in whatever interpreter is running it.

With no arguments (i.e. just clicking Run) this pops native file-picker
dialogs for the HR CSV, LR CSV, and OBJ model - same as running the CLI
with no arguments. To pass real arguments instead, either edit sys.argv
below, or run `blueraven-visualizer ...` / `python -m blueraven_visualizer.cli ...`
from a terminal.
"""

from blueraven_visualizer.cli import main

if __name__ == "__main__":
    main()
