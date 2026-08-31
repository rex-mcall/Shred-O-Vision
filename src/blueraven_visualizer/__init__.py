"""Blue Raven flight-orientation visualizer.

Replays Featherweight Blue Raven quaternion logs as a spinning 3D rocket
model synced to a telemetry dashboard. See render_matplotlib.visualize (the
default, zero-extra-dependency backend) and render_pyvista.visualize (the
optional textured/hardware-accelerated backend) for details.
"""

from .render_matplotlib import visualize

__version__ = "0.1.0"
__all__ = ["visualize", "__version__"]
