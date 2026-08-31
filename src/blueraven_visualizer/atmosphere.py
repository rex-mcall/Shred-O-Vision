"""Atmospheric helpers, ported from blueraven_visualizer.m."""

import numpy as np


def speed_of_sound_ftps(temp_f):
    """Speed of sound [ft/s] from measured air temperature [deg F]."""
    temp_k = (np.asarray(temp_f, float) - 32) * 5 / 9 + 273.15
    return np.sqrt(1.4 * 287 * temp_k) / 0.3048


def mach_number(v_up, v_dr, v_cr, temp_f):
    """Mach number from the three inertial velocity components [ft/s] and
    measured air temperature [deg F] (speed of sound varies with altitude)."""
    speed_fts = np.sqrt(np.asarray(v_up, float) ** 2 +
                         np.asarray(v_dr, float) ** 2 +
                         np.asarray(v_cr, float) ** 2)
    return speed_fts / speed_of_sound_ftps(temp_f)
