"""Blue Raven flight-orientation visualizer.

Replays Featherweight Blue Raven quaternion logs as a spinning 3D rocket
model synced to a telemetry dashboard. See render_matplotlib.visualize (the
default, zero-extra-dependency backend) and render_pyvista.visualize (the
optional textured/hardware-accelerated backend) for details, and
render_compare.compare_flights for two flights side by side.
"""

from .render_matplotlib import visualize
from .render_compare import compare_flights

__version__ = "0.1.0"
__all__ = ["visualize", "compare_flights", "__version__"]
