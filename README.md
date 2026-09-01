# blueraven-visualizer

Turns a [Featherweight Blue Raven](https://www.featherweightaltimeters.com/) flight-computer log into a 3D rocket-orientation animation, synced to a telemetry dashboard (pyro-charge voltages, altitude/velocity, tilt/roll), with automatic detection of the peak-acceleration ("shred"/structural-failure) instant.

Built for post-flight forensics: pinpoint exactly when and how a rocket lost stability, and watch it happen from the rocket's own point of view.

![CI](https://github.com/rexmcall/blueraven-visualizer/actions/workflows/ci.yml/badge.svg)
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)

## What it does

- Reads a matched pair of Blue Raven CSV exports: the 500 Hz **HR** file (quaternions, accel, gyro) and the 50 Hz **LR** file (baro altitude/velocity, pyro voltages, tilt/roll, flight-state flags). The LR file is optional — without it you still get the 3D orientation view plus HR-derived acceleration/gyro panels.
- Spins a 3D model (your own `.obj`, or a built-in rocket glyph with fins, a colored nose, and one fin + a matching body stripe painted a marker color so roll/spin is visible during playback) using the logged quaternions, with a moving time cursor synced across every panel.
- Auto-detects liftoff, burnout, apogee, drogue/main-charge fire, the shred instant (peak IMU acceleration), and tumble onset (peak angular rate), and marks them on every plot.
- Prints a console flight report summarizing every detected event plus max velocity/Mach and peak altitude - data only, no interpretation.
- Interactive playback (Play/Pause, Step◀/▶, Restart, drag-to-scrub, keyboard shortcuts) is paced to the wall clock, so it tracks real time (at `--speed 1`, the default) even if a frame takes longer to render than its nominal slot - it catches up rather than falling into slow motion. Exports a real-time-accurate MP4/GIF for sharing.
- Two rendering backends — pick whichever fits what you need:

  | | `--renderer matplotlib` (default) | `--renderer pyvista` |
  |---|---|---|
  | Dependencies | numpy/pandas/matplotlib only | + [PyVista](https://pyvista.org)/VTK (`pip install blueraven-visualizer[pyvista]`) |
  | Telemetry dashboard | Yes, full 4-panel layout | No — 3D orientation view only |
  | OBJ textures/materials | No (flat-shaded silhouette) | Yes, real per-part textures via VTK's OBJ importer |
  | Rendering | Software (matplotlib `mplot3d`) | Hardware-accelerated |

## No command-line experience? Start here

Run `blueraven-visualizer` with nothing after it (or double-click `run.py`, or hit Run in VS Code) and you get a guided, plain-language wizard instead of flags to remember - it asks a few yes/no and numbered questions in the terminal and pops a file-picker window at the right moments:

```
============================================================
  Blue Raven Visualizer - guided setup
============================================================
Answer a few questions to get started - press Enter to accept
the default for anything you're not sure about.

Step 1 of 5: pick your flight-computer files.
A window will open - navigate to your Blue Raven CSV export.
Press Enter to choose the HIGH-RATE (HR) CSV file...
```

Everything below this point (the `--flag` examples) is for when you want more control.

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
# guided wizard (see "No command-line experience?" above)
blueraven-visualizer
# same wizard, even if other flags are present
blueraven-visualizer --menu

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

> `--record` to `.mp4` or `.gif` works out of the box for both backends - no system ffmpeg install needed. matplotlib prefers a system `ffmpeg` on your `PATH` if you have one, and otherwise falls back to the copy bundled by `imageio-ffmpeg` (a core dependency); pyvista always uses its bundled copy.

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

Every run also prints a flight report to the console:

```
----------------------------------------------------------
  BLUE RAVEN FLIGHT REPORT
----------------------------------------------------------
  Liftoff               T+   0.00 s
  Burnout (flag)        T+   6.30 s
  Max velocity          T+   6.28 s    1602 ft/s  (Mach 1.50)
  Peak altitude AGL     T+   6.56 s    9214 ft
----------------------------------------------------------
  >> SHRED (peak g)     T+   6.47 s    309 g
     Tumble onset       T+   6.69 s    3779 deg/s  (10.5 rev/s)
----------------------------------------------------------
  Baro apogee           T+  13.88 s
  Drogue/Apo fired      T+  15.42 s
  Main fired             --
----------------------------------------------------------
```

> **A note on OBJ models:** because the rocket tumbles freely, the 3D view has to fit your whole model at any rotation without distorting it - so a very slender model (long body, small diameter), or one exported as an "exploded" CAD diagram with gaps between parts, will look thin no matter how the camera is tuned. An assembled, non-exploded model looks best.

## CSV format

Expects the raw, unmodified CSV exports from the Featherweight Blue Raven app: the HR file needs `Flight_Time_(s)`, `Quat_1..4`, `Accel_X/Y/Z`, `Gyro_X/Y/Z`; the LR file (if used) needs `Flight_Time_(s)`, the pyro-channel `*_Volts` columns, `Baro_Altitude_AGL_(feet)`, `Velocity_Up`, `Tilt_Angle_(deg)`, `Roll_Angle_(deg)`, and the flight-state flag columns (`Liftoff`, `Burnout_Coast`, `Apogee`, `Apo_fired`, `Main_fired`). `Velocity_DR/CR` and `Temperature_(F)` are used for the report's Mach number if present, but optional. If a column's missing you'll get an error naming the file and column, rather than a raw crash.

## Contributing / issues

Bug reports and PRs welcome — this was built for one team's flight data, so if your Blue Raven export doesn't parse cleanly, please open an issue with (a redacted sample of) the CSV header.

## License

[MIT](LICENSE)
