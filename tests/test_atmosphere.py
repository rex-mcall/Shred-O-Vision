import numpy as np
import pytest

from blueraven_visualizer.atmosphere import speed_of_sound_ftps, mach_number


def test_speed_of_sound_at_standard_sea_level_temp():
    # ISA sea-level standard temp is 59 F (15 C); speed of sound there is
    # ~1116 ft/s.
    assert speed_of_sound_ftps(59.0) == pytest.approx(1116.4, abs=1.0)


def test_speed_of_sound_increases_with_temperature():
    assert speed_of_sound_ftps(80.0) > speed_of_sound_ftps(20.0)


def test_mach_number_pure_vertical_velocity():
    a = speed_of_sound_ftps(59.0)
    mach = mach_number(v_up=a, v_dr=0.0, v_cr=0.0, temp_f=59.0)
    assert mach == pytest.approx(1.0, rel=1e-6)


def test_mach_number_combines_components():
    temp_f = 59.0
    a = speed_of_sound_ftps(temp_f)
    mach = mach_number(v_up=0.6 * a, v_dr=0.8 * a, v_cr=0.0, temp_f=temp_f)
    assert mach == pytest.approx(1.0, rel=1e-6)


def test_mach_number_array_input():
    temp_f = np.array([59.0, 59.0])
    a = speed_of_sound_ftps(59.0)
    mach = mach_number(v_up=np.array([a, 2 * a]), v_dr=np.zeros(2), v_cr=np.zeros(2),
                        temp_f=temp_f)
    assert mach[0] == pytest.approx(1.0, rel=1e-6)
    assert mach[1] == pytest.approx(2.0, rel=1e-6)
