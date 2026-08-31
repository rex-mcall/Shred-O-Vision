import matplotlib

# Must happen before anything imports matplotlib.pyplot (render_matplotlib does,
# at module level) so tests never try to open a GUI window / need a display.
matplotlib.use("Agg")
