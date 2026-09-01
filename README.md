# Shred-O-Vision

Turns a [Featherweight Blue Raven](https://www.featherweightaltimeters.com/) flight-computer log into a 3D rocket-orientation animation, synced to a telemetry dashboard (pyro-charge voltages, altitude/velocity, tilt/roll).

Replay any flight from the rocket's own point of view - a clean nominal flight or a breakup. It reports the key instants (liftoff, burnout, apogee, deployments, peak acceleration and angular rate) and leaves the interpretation to you; for failure analysis, `--window peak --highlight-peak` zooms to the peak-g instant and marks everything after it.

![CI](https://github.com/rex-mcall/Shred-O-Vision/actions/workflows/ci.yml/badge.svg)
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)

## What it does

- Reads a matched pair of Blue Raven CSV exports: the 500 Hz **HR** file (quaternions, accel, gyro) and the 50 Hz **LR** file (baro altitude/velocity, pyro voltages, tilt/roll, flight-state flags). The LR file is optional — without it you still get the 3D orientation view plus HR-derived acceleration/gyro panels.
- Spins a 3D model (your own `.obj`, or a built-in rocket glyph with fins, a colored nose, and one fin + a matching body stripe painted a marker color so roll/spin is visible during playback) using the logged quaternions, with a moving time cursor synced across every panel. The matplotlib backend real-time-shades every face from its own current normal (not matplotlib's `shade=True`, which freezes lighting at the model's starting orientation and never updates it as the mesh rotates) - real geometric detail like fins reads clearly even in flat gray, not just with pyvista's textures. A real OBJ (which has no per-part labels to build a marker from the way the built-in glyph does) gets the same roll-visibility treatment a different way - a stripe painted by angular position around the model's own long axis, so it rotates rigidly with the mesh regardless of what the geometry actually represents. `--record`ed exports also render your OBJ at much higher detail than interactive playback does (playback decimates a large mesh aggressively to stay responsive; a one-time export isn't racing a live frame budget, so it keeps ~5x more geometry).
- Auto-detects liftoff, burnout, apogee, drogue/main-charge fire, peak acceleration (max thrust on a nominal flight; the break on a failure) and peak angular rate, and marks them on every plot.
- Prints a console flight report summarizing every detected event plus max velocity/Mach and peak altitude - data only, no interpretation. Max velocity is taken over the ascent, and Mach is reported at that same instant: the Blue Raven's inertial velocities drift badly once the airframe is tumbling or under canopy, so a whole-flight maximum reports descent noise as flight performance.
- Interactive playback (Play/Pause, Step◀/▶, Restart, drag-to-scrub, keyboard shortcuts) is paced to the wall clock, so it tracks real time (at `--speed 1`, the default) even if a frame takes longer to render than its nominal slot - it catches up rather than falling into slow motion. The matplotlib backend blits (redraws only what changed - the mesh, cursors, and the slider itself - instead of the whole figure, ticks and all, every frame), measured at ~450 ms per interaction before, ~16 ms after. Exports a real-time-accurate MP4/GIF for sharing.
- Two rendering backends — pick whichever fits what you need:

  | | `--renderer matplotlib` (default) | `--renderer pyvista` |
  |---|---|---|
  | Dependencies | numpy/pandas/matplotlib only | + [PyVista](https://pyvista.org)/VTK (`pip install blueraven-visualizer[pyvista]`) |
  | Telemetry dashboard | Yes, full 4-panel layout | No — 3D orientation view only |
  | OBJ textures/materials | No (real-time-lit gray, not textured) | Yes, real per-part textures via VTK's OBJ importer |
  | Rendering | Software (matplotlib `mplot3d`) | Hardware-accelerated |
  | Camera | Interactive (mouse-orbit) | Fixed and locked - a ground plane + a launch-vertical reference line stay put while the rocket tumbles, so it's easy to tell what's actually moving |

## No command-line experience? Start here

Run `blueraven-visualizer` with nothing after it (or double-click `run.py`, or hit Run in VS Code) and you get a guided, plain-language wizard instead of flags to remember - it asks a few yes/no and numbered questions in the terminal and pops a file-picker window at the right moments:

```
============================================================
  Blue Raven Visualizer - guided setup
============================================================
A file picker will open for each file below - Cancel skips an
optional one. Then answer a couple of quick questions.

Opening file picker: HIGH-RATE (HR) Blue Raven CSV...
```

Everything below this point (the `--flag` examples) is for when you want more control.

## Install

```bash
git clone https://github.com/rex-mcall/Shred-O-Vision.git
cd Shred-O-Vision
pip install -e .
```

That installs the command `blueraven-visualizer` (with `shred-o-vision` as an
alias for it - they're the same program).

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

# launch through apogee (the default) as an MP4
blueraven-visualizer HR.csv LR.csv --obj my_rocket.obj --record flight.mp4

# the entire recording, descent included
blueraven-visualizer HR.csv LR.csv --obj my_rocket.obj --window full --record flight.mp4

# zoom to the peak-acceleration instant, and flag everything after it
# (failure analysis: "--window shred" still works as an alias for "peak")
blueraven-visualizer HR.csv LR.csv --obj my_rocket.obj --window peak --pad 2.0 --highlight-peak --record breakup.mp4

# textured, hardware-accelerated 3D view (needs the pyvista extra)
blueraven-visualizer HR.csv LR.csv --obj my_rocket.obj --renderer pyvista
```

Run `blueraven-visualizer --help` for the full flag list (playback speed/FPS, model-nose axis, mesh decimation for the matplotlib backend, etc).

> Playback defaults to **launch through apogee** - on the bundled flight the descent is 88 s of a 105 s recording and shows little in the orientation view. Use `--window full` for everything.
>
> `--record` renders frames by blitting (reusing the static background and redrawing only the mesh, HUD and time cursors) and streams them straight to the encoder, rather than doing a full figure redraw per frame - measured at ~30 ms/frame instead of ~280-710 ms.
>
> `--record` to `.mp4` or `.gif` works out of the box on both backends with nothing installed system-wide: encoding goes through the ffmpeg binary bundled by `imageio-ffmpeg`, which is a core dependency.

From Python:

```python
from blueraven_visualizer import visualize
visualize("HR.csv", "LR.csv", obj="my_rocket.obj", record="flight.mp4")
visualize("HR.csv", "LR.csv", obj="my_rocket.obj", window="peak", highlight_peak=True)
```

## Try it on the bundled example

`examples/mothman_avenged/` ships a real flight (WVUER's "Mothman Avenged," IREC 2026) with its matched, fully-textured OBJ model, plus a [reference output video](examples/mothman_avenged/reference_output.mp4) so you can see what to expect before running anything:

```bash
cd examples/mothman_avenged
blueraven-visualizer "BlRv_wvuer1_HR_06-17-2026_09_25_14.csv" "BlRv_wvuer1_LR_06-17-2026_09_25_14.csv" --obj mmavenged.obj --window peak --pad 2
```

Every run also prints a flight report to the console:

```
----------------------------------------------------------
  BLUE RAVEN FLIGHT REPORT
----------------------------------------------------------
  Liftoff               T+   0.00 s
  Burnout (flag)        T+   6.30 s
  Max velocity          T+   6.28 s    1602 ft/s  (Mach 1.39)
  Peak altitude AGL     T+   6.56 s    9214 ft
----------------------------------------------------------
  Peak acceleration     T+   6.47 s    309 g
  Peak angular rate     T+   6.69 s    3779 deg/s  (10.5 rev/s)
----------------------------------------------------------
  Baro apogee           T+  13.88 s
  Drogue/Apo fired      T+  15.42 s
  Main fired            T+  88.80 s
----------------------------------------------------------
```

> **A note on OBJ models:** because the rocket tumbles freely, the matplotlib backend's 3D view has to fit your whole model at any rotation without distorting it - so a very slender model (long body, small diameter), or one exported as an "exploded" CAD diagram with gaps between parts, will still show some empty margin no matter how the camera is tuned. That's a framing limit, not a detail one, though - real per-face shading and a much higher default mesh-decimation budget mean the model's actual shape (fins, seams, taper) reads clearly either way. The pyvista backend doesn't have the framing limit at all (its camera fits the model once rather than guaranteeing every rotation stays in frame) and adds real textures on top, so it's still the better choice for a highly detailed or very slender model - an assembled, non-exploded model looks best in either backend.
>
> **Model orientation is detected automatically.** CAD tools and exporters disagree about which axis is "up", and guessing wrong is visually dramatic rather than subtle - the rocket gets rotated about the wrong axis and is drawn lying sideways, or (seen straight down the tube) as nothing but fins radiating around an almost invisible body, all while the HUD still reports a near-zero tilt. The long axis is taken from the direction the geometry is most spread out along (deliberately not the bounding box - a few big fins can make a rocket wider than it is long) and the nose end from whichever end tapers; `--model-nose` overrides it if a model ever fools the heuristic.
>
> It also warns when a model looks like a single component rather than a whole airframe - if you export just the fin set from OpenRocket, that's all the tool has to draw, and the note tells you so instead of leaving you guessing.
>
> The OBJ reader handles the formatting variations real exporters actually emit - tab-separated face lines, a UTF-8 BOM, quads/n-gons, negative indices, `v/vt/vn` forms - and tells you if it had to skip anything, rather than silently dropping whole components.
>
> The pyvista backend also automatically works around two real, confirmed limitations in VTK's own OBJ importer that show up with unmodified real-world CAD/OpenRocket exports (a `-clamp on`/`off` texture option that corrupts the filename after it, and material names that are long or punctuated enough that VTK fails to match them between the `.obj` and `.mtl`) - it rewrites temporary sanitized copies before handing them to VTK and never touches your original files.

## CSV format

Expects the raw, unmodified CSV exports from the Featherweight Blue Raven app: the HR file needs `Flight_Time_(s)`, `Quat_1..4`, `Accel_X/Y/Z`, `Gyro_X/Y/Z`; the LR file (if used) needs `Flight_Time_(s)`, the pyro-channel `*_Volts` columns, `Baro_Altitude_AGL_(feet)`, `Velocity_Up`, `Tilt_Angle_(deg)`, `Roll_Angle_(deg)`, and the flight-state flag columns (`Liftoff`, `Burnout_Coast`, `Apogee`, `Apo_fired`, `Main_fired`). `Velocity_DR/CR` and `Temperature_(F)` are used for the report's Mach number if present, but optional. If a column's missing you'll get an error naming the file and column, rather than a raw crash.

## Contributing / issues

Bug reports and PRs welcome — this was built for one team's flight data, so if your Blue Raven export doesn't parse cleanly, please open an issue with (a redacted sample of) the CSV header.

## License

[MIT](LICENSE)
