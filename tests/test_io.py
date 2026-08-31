import os

import pytest

from blueraven_visualizer.io import load_blueraven, BlueRavenFormatError

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
HR_SAMPLE = os.path.join(FIXTURES, "hr_sample.csv")
LR_SAMPLE = os.path.join(FIXTURES, "lr_sample.csv")


def test_load_blueraven_hr_reads_expected_columns():
    df, col = load_blueraven(HR_SAMPLE)
    t = col("Flight_Time_(s)")
    assert len(t) == len(df)
    assert len(t) > 100
    # covers liftoff through past the shred instant (see tests/fixtures generation)
    assert t[0] < 0 < t[-1]


def test_load_blueraven_lr_reads_expected_columns():
    df, col = load_blueraven(LR_SAMPLE)
    liftoff = col("Liftoff")
    assert (liftoff > 0.5).any()


def test_missing_required_column_raises_actionable_error():
    _, col = load_blueraven(HR_SAMPLE)
    with pytest.raises(BlueRavenFormatError) as exc_info:
        col("Not_A_Real_Column")
    msg = str(exc_info.value)
    assert "Not_A_Real_Column" in msg
    assert "hr_sample.csv" in msg


def test_missing_optional_column_returns_none():
    _, col = load_blueraven(HR_SAMPLE)
    assert col("Not_A_Real_Column", required=False) is None
