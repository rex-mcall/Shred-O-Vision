import blueraven_visualizer.cli as cli


def test_no_args_triggers_menu(monkeypatch):
    calls = {}
    monkeypatch.setattr("blueraven_visualizer.menu.run_menu", lambda: {
        "hr_csv": "hr.csv", "lr_csv": None, "obj": None,
        "renderer": "matplotlib", "window": "shred", "record": None,
    })
    monkeypatch.setattr(cli, "_run", lambda *a, **k: calls.update(args=a, kwargs=k))

    cli.main([])

    assert calls["args"] == ("hr.csv", None, None, "matplotlib", "shred", None)


def test_menu_flag_triggers_menu_even_with_other_args(monkeypatch):
    monkeypatch.setattr("blueraven_visualizer.menu.run_menu", lambda: {
        "hr_csv": "hr.csv", "lr_csv": None, "obj": None,
        "renderer": "matplotlib", "window": None, "record": None,
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

    cli.main(["hr.csv", "lr.csv", "--window", "shred", "--record", "out.mp4"])

    assert not menu_called["yes"]
    assert calls["args"][0] == "hr.csv"
    assert calls["args"][1] == "lr.csv"
    assert calls["args"][4] == "shred"    # window
    assert calls["args"][5] == "out.mp4"  # record


def test_lr_none_word_becomes_none():
    assert cli._parse_optional("none") is None
    assert cli._parse_optional("None") is None
    assert cli._parse_optional("lr.csv") == "lr.csv"


def test_window_shred_passthrough_vs_range_parsing():
    assert cli._parse_window("shred") == "shred"
    assert cli._parse_window(None) is None
    assert cli._parse_window("1.5,3.0") == [1.5, 3.0]
