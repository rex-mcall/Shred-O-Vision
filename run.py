"""Convenience entry point for VS Code's Run button (or `python run.py`
from anywhere). Just click Run.

blueraven_visualizer.cli uses relative imports, so it can't be run directly
as a script (VS Code's Run button does `python <file>`, not `python -m
package.module`) - this thin wrapper uses a plain absolute import instead,
which works either way as long as the package is installed (`pip install -e
.`) in whatever interpreter is running it.

With no arguments (i.e. just clicking Run) this launches the guided,
plain-language wizard (see menu.py) - it asks a few questions in the
terminal and pops native file-picker dialogs at the right moments. To use
flags instead, either edit sys.argv below, or run
`blueraven-visualizer ...` / `python -m blueraven_visualizer.cli ...`
from a terminal.
"""

from blueraven_visualizer.cli import main

if __name__ == "__main__":
    main()
