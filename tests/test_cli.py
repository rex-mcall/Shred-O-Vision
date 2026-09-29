import pytest

import blueraven_visualizer.cli as cli


def test_no_args_triggers_menu(monkeypatch):
    calls = {}
    monkeypatch.setattr("blueraven_visualizer.menu.run_menu", lambda: {
        "hr_csv": "hr.csv", "lr_csv": None, "obj": None,
        "renderer": "matplotlib", "window": "peak", "record": None,
        "highlight_peak": False,
    })
    monkeypatch.setattr(cli, "_run", lambda *a, **k: calls.update(args=a, kwargs=k))

    cli.main([])

    assert calls["args"] == ("hr.csv", None, None, "matplotlib", "peak", None)


def test_menu_flag_triggers_menu_even_with_other_args(monkeypatch):
    monkeypatch.setattr("blueraven_visualizer.menu.run_menu", lambda: {
        "hr_csv": "hr.csv", "lr_csv": None, "obj": None,
        "renderer": "matplotlib", "window": None, "record": None,
        "highlight_peak": False,
    })
    called = {"run": False}
    monkeypatch.setattr(cli, "_run", lambda *a, **k: called.update(run=True))

    cli.main(["--menu"])

    assert called["run"]


def test_explicit_args_skip_menu_and_use_flag_path(monkeypatch):
    menu_called = {"yes": False}
    monkeypatch.setattr("blueraven_visualizer.menu.run_menu",
                        lambda: menu_called.update(yes=True))
    calls = {}
    monkeypatch.setattr(cli, "_run", lambda *a, **k: calls.update(args=a, kwargs=k))

    cli.main(["hr.csv", "lr.csv", "--window", "peak", "--record", "out.mp4"])

    assert not menu_called["yes"]
    assert calls["args"][0] == "hr.csv"
    assert calls["args"][1] == "lr.csv"
    assert calls["args"][4] == "peak"     # window
    assert calls["args"][5] == "out.mp4"  # record


def test_lr_none_word_becomes_none():
    assert cli._parse_optional("none") is None
    assert cli._parse_optional("None") is None
    assert cli._parse_optional("lr.csv") == "lr.csv"


def test_window_shred_passthrough_vs_range_parsing():
    assert cli._parse_window("peak") == "peak"
    assert cli._parse_window("shred") == "shred"   # back-compat alias
    assert cli._parse_window(None) is None
    assert cli._parse_window("1.5,3.0") == [1.5, 3.0]


# --- two-flight comparison ---

def test_compare_flag_dispatches_both_flights(monkeypatch):
    from blueraven_visualizer.render_compare import SAME_AS_A
    calls = {}
    monkeypatch.setattr(cli, "_run_compare", lambda *a, **k: calls.update(args=a, kwargs=k))

    cli.main(["hr_a.csv", "lr_a.csv", "--obj", "none", "--compare", "hr_b.csv", "lr_b.csv",
              "--label-b", "Kansas", "--record", "cmp.mp4"])

    hr_a, lr_a, obj_a, compare, window, record = calls["args"]
    assert (hr_a, lr_a, obj_a) == ("hr_a.csv", "lr_a.csv", None)
    assert (compare["hr_csv"], compare["lr_csv"]) == ("hr_b.csv", "lr_b.csv")
    assert compare["obj"] is SAME_AS_A          # default: reuse flight A's model
    assert compare["label_b"] == "Kansas"
    assert record == "cmp.mp4"


def test_compare_obj_b_none_means_the_built_in_glyph(monkeypatch):
    calls = {}
    monkeypatch.setattr(cli, "_run_compare", lambda *a, **k: calls.update(args=a))
    cli.main(["a.csv", "b.csv", "--obj", "none", "--compare", "c.csv", "d.csv", "--obj-b", "none"])
    assert calls["args"][3]["obj"] is None


@pytest.mark.parametrize("extra", [["--renderer", "pyvista"], ["--no-3d"], ["--highlight-peak"]])
def test_compare_refuses_single_flight_only_options(monkeypatch, extra):
    monkeypatch.setattr(cli, "_run_compare", lambda *a, **k: pytest.fail("should not dispatch"))
    with pytest.raises(SystemExit):
        cli.main(["a.csv", "b.csv", "--obj", "none", "--compare", "c.csv", "d.csv", *extra])


def test_compare_needs_the_first_flights_lr_file(monkeypatch):
    monkeypatch.setattr(cli, "_run_compare", lambda *a, **k: pytest.fail("should not dispatch"))
    with pytest.raises(SystemExit):
        cli.main(["a.csv", "none", "--obj", "none", "--compare", "c.csv", "d.csv"])


@pytest.mark.parametrize("flag", ["--obj-b", "--label-a", "--label-b"])
def test_compare_only_options_are_rejected_without_compare(monkeypatch, flag):
    monkeypatch.setattr(cli, "_run", lambda *a, **k: pytest.fail("should not dispatch"))
    with pytest.raises(SystemExit):
        cli.main(["a.csv", "b.csv", "--obj", "none", flag, "x"])


def test_menu_compare_choice_dispatches_to_compare(monkeypatch):
    called = {}
    monkeypatch.setattr("blueraven_visualizer.menu.run_menu", lambda: {
        "hr_csv": "a", "lr_csv": "b", "obj": None, "renderer": "matplotlib",
        "window": None, "record": None, "highlight_peak": False,
        "compare": {"hr_csv": "c", "lr_csv": "d", "obj": None},
    })
    monkeypatch.setattr(cli, "_run_compare", lambda *a, **k: called.update(compare=a))
    monkeypatch.setattr(cli, "_run", lambda *a, **k: called.update(single=True))
    cli.main([])
    assert "compare" in called and "single" not in called


def test_compare_passes_each_flights_roll_axis(monkeypatch):
    calls = {}
    monkeypatch.setattr(cli, "_run_compare", lambda *a, **k: calls.update(kwargs=k))
    cli.main(["a.csv", "b.csv", "--obj", "none", "--compare", "c.csv", "d.csv",
              "--roll-axis", "+y", "--roll-axis-b", "-z"])
    assert calls["kwargs"]["roll_axis_a"] == "+y"
    assert calls["kwargs"]["roll_axis_b"] == "-z"

    cli.main(["a.csv", "b.csv", "--obj", "none", "--compare", "c.csv", "d.csv"])
    assert calls["kwargs"]["roll_axis_a"] == "auto"
    assert calls["kwargs"]["roll_axis_b"] == "auto"


def test_roll_axis_b_needs_compare(monkeypatch):
    monkeypatch.setattr(cli, "_run", lambda *a, **k: pytest.fail("should not dispatch"))
    with pytest.raises(SystemExit):
        cli.main(["a.csv", "b.csv", "--obj", "none", "--roll-axis-b", "+y"])


@pytest.mark.parametrize("flag,value", [("--roll-axis", "-y"), ("--model-nose", "-z"),
                                        ("--roll-axis", "+y")])
def test_negative_axis_values_parse_as_written(monkeypatch, flag, value):
    """Regression guard: argparse read `-y` as the next option, so
    `--roll-axis -y` (the spelling the README suggests) failed with
    "expected one argument"."""
    calls = {}
    monkeypatch.setattr(cli, "_run", lambda *a, **k: calls.update(kwargs=k))
    cli.main(["a.csv", "b.csv", "--obj", "none", flag, value])
    key = "roll_axis" if flag == "--roll-axis" else "model_nose"
    assert calls["kwargs"][key] == value
