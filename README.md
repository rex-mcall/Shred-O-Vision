# blueraven-visualizer

Turns a [Featherweight Blue Raven](https://www.featherweightaltimeters.com/) flight-computer log into a 3D rocket-orientation animation, synced to a telemetry dashboard (pyro-charge voltages, altitude/velocity, tilt/roll), with automatic detection of the peak-acceleration ("shred"/structural-failure) instant.

Built for post-flight forensics: pinpoint exactly when and how a rocket lost stability, and watch it happen from the rocket's own point of view.

![CI](https://github.com/rexmcall/blueraven-visualizer/actions/workflows/ci.yml/badge.svg)
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)

## What it does

- Reads a matched pair of Blue Raven CSV exports: the 500 Hz **HR** file (quaternions, accel, gyro) and the 50 Hz **LR** file (baro altitude/velocity, pyro voltages, tilt/roll, flight-state flags). The LR file is optional — without it you still get the 3D orientation view plus HR-derived acceleration/gyro panels.
- Spins a 3D model (your own `.obj`, or a built-in rocket glyph) using the logged quaternions, with a moving time cursor synced across every panel.
- Auto-detects liftoff, burnout, apogee, drogue/main-charge fire, and the shred instant (peak IMU acceleration), and marks them on every plot.
- Exports a real-time MP4/GIF, or opens an interactive scrub/play viewer.
- Two rendering backends — pick whichever fits what you need:

  | | `--renderer matplotlib` (default) | `--renderer pyvista` |
  |---|---|---|
  | Dependencies | numpy/pandas/matplotlib only | + [PyVista](https://pyvista.org)/VTK (`pip install blueraven-visualizer[pyvista]`) |
  | Telemetry dashboard | Yes, full 4-panel layout | No — 3D orientation view only |
  | OBJ textures/materials | No (flat-shaded silhouette) | Yes, real per-part textures via VTK's OBJ importer |
  | Rendering | Software (matplotlib `mplot3d`) | Hardware-accelerated |

## Install

```bash
git clone https://github.com/rexmcall/blueraven-visualizer.git
cd blueraven-visualizer
pip install -e .
```

For the textured/hardware-accelerated backend, also install the optional extra:

```bash
pip install -e ".[pyvista]"
```

## Usage

```bash
# pop-up file pickers for each file
blueraven-visualizer

# explicit files, interactive scrub/play viewer
blueraven-visualizer HR.csv LR.csv --obj my_rocket.obj

# orientation only, no LR telemetry file
blueraven-visualizer HR.csv none

# render an MP4 of just the shred window
blueraven-visualizer HR.csv LR.csv --obj my_rocket.obj --window shred --pad 2.0 --record shred.mp4

# textured, hardware-accelerated 3D view (needs the pyvista extra)
blueraven-visualizer HR.csv LR.csv --obj my_rocket.obj --renderer pyvista
```

Run `blueraven-visualizer --help` for the full flag list (playback speed/FPS, model-nose axis, mesh decimation for the matplotlib backend, etc).

> `--record` to a `.gif` always works out of the box. `--record` to `.mp4` with the matplotlib backend needs the `ffmpeg` binary on your `PATH` (matplotlib shells out to it) - you'll get a clear error telling you so if it's missing. The pyvista backend's MP4 export doesn't need system ffmpeg (it bundles its own via `imageio-ffmpeg`).

From Python:

```python
from blueraven_visualizer import visualize
visualize("HR.csv", "LR.csv", obj="my_rocket.obj", window="shred", record="shred.mp4")
```

## Try it on the bundled example

`examples/mothman_avenged/` ships a real flight (WVUER's "Mothman Avenged," IREC 2026) with its matched, fully-textured OBJ model, plus a [reference output video](examples/mothman_avenged/reference_output.mp4) so you can see what to expect before running anything:

```bash
cd examples/mothman_avenged
blueraven-visualizer "BlRv_wvuer1_HR_06-17-2026_09_25_14.csv" "BlRv_wvuer1_LR_06-17-2026_09_25_14.csv" --obj mmavenged.obj --window shred --pad 2
```

## CSV format

Expects the raw, unmodified CSV exports from the Featherweight Blue Raven app: the HR file needs `Flight_Time_(s)`, `Quat_1..4`, `Accel_X/Y/Z`, `Gyro_X/Y/Z`; the LR file (if used) needs `Flight_Time_(s)`, the pyro-channel `*_Volts` columns, `Baro_Altitude_AGL_(feet)`, `Velocity_Up`, `Tilt_Angle_(deg)`, `Roll_Angle_(deg)`, and the flight-state flag columns (`Liftoff`, `Burnout_Coast`, `Apogee`, `Apo_fired`, `Main_fired`). If a column's missing you'll get an error naming the file and column, rather than a raw crash.

## Contributing / issues

Bug reports and PRs welcome — this was built for one team's flight data, so if your Blue Raven export doesn't parse cleanly, please open an issue with (a redacted sample of) the CSV header.

## License

[MIT](LICENSE)
