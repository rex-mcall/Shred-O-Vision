"""Console flight report, ported from blueraven_visualizer.m's console output.

Summarizes the key events (liftoff, burnout, max velocity/Mach, peak
altitude, shred, tumble onset, apogee, drogue/main fire). This reports the
data only - no diagnosis/interpretation is printed; that judgment call is
left to the person reading it.
"""

import numpy as np

from .events import first_true_time, detect_shred, detect_tumble_onset
from .atmosphere import mach_number

_RULE = "-" * 58


def _rep(lines, label, t, extra=""):
    if t is None or np.isnan(t):
        lines.append(f"  {label:<20}   --")
    elif not extra:
        lines.append(f"  {label:<20}  T+{t:7.2f} s")
    else:
        lines.append(f"  {label:<20}  T+{t:7.2f} s    {extra}")


def build_report(t_hr, accel_mag, gyro_mag, *, t_lr=None, lc=None):
    """Build the flight report text.

    t_hr/accel_mag/gyro_mag: HR arrays (always available).
    t_lr: LR time array, or None if no LR file was loaded.
    lc: the LR column accessor from io.load_blueraven, or None.
    """
    tShred, gPeak, _ = detect_shred(t_hr, accel_mag)
    tSpin, wPeak, _ = detect_tumble_onset(t_hr, gyro_mag)

    lines = [_RULE, "  BLUE RAVEN FLIGHT REPORT", _RULE]

    tBurn = None
    machMax = None
    has_lr = t_lr is not None and lc is not None

    if has_lr:
        vup = lc("Velocity_Up")
        vdr = lc("Velocity_DR", required=False)
        vcr = lc("Velocity_CR", required=False)
        tempF = lc("Temperature_(F)", required=False)
        alt = lc("Baro_Altitude_AGL_(feet)")

        tLift = first_true_time(t_lr, lc("Liftoff") > 0.5)
        tBurn = first_true_time(t_lr, lc("Burnout_Coast") > 0.5)
        tApo = first_true_time(t_lr, lc("Apogee") > 0.5)
        tApoF = first_true_time(t_lr, lc("Apo_fired") > 0.5)
        tMainF = first_true_time(t_lr, lc("Main_fired") > 0.5)

        iVmax = int(np.argmax(vup))
        vMax, tVmax = float(vup[iVmax]), float(t_lr[iVmax])
        iAlt = int(np.argmax(alt))
        altMax, tAlt = float(alt[iAlt]), float(t_lr[iAlt])

        if vdr is not None and vcr is not None and tempF is not None:
            mach = mach_number(vup, vdr, vcr, tempF)
            machMax = float(np.nanmax(mach))

        vel_extra = f"{vMax:.0f} ft/s" + (f"  (Mach {machMax:.2f})" if machMax is not None else "")
        _rep(lines, "Liftoff", tLift)
        _rep(lines, "Burnout (flag)", tBurn)
        _rep(lines, "Max velocity", tVmax, vel_extra)
        _rep(lines, "Peak altitude AGL", tAlt, f"{altMax:.0f} ft")
        lines.append(_RULE)

    _rep(lines, ">> SHRED (peak g)", tShred, f"{gPeak:.0f} g")
    _rep(lines, "   Tumble onset", tSpin, f"{wPeak:.0f} deg/s  ({wPeak / 360:.1f} rev/s)")

    if has_lr:
        lines.append(_RULE)
        _rep(lines, "Baro apogee", tApo)
        _rep(lines, "Drogue/Apo fired", tApoF)
        _rep(lines, "Main fired", tMainF)

    lines.append(_RULE)

    return "\n".join(lines)
