"""The menu wizard is interactive (input() + native file dialogs), so these
tests drive it by mocking both - verifying the question flow and the choices
it hands back to the CLI, not the actual dialog/terminal behavior."""

import blueraven_visualizer.menu as menu


def test_menu_all_defaults_skips_lr_and_obj(monkeypatch):
    def fake_pick_file(title, *a, **k):
        return "/fake/hr.csv" if "HIGH-RATE" in title else None   # Cancel on LR and OBJ

    inputs = iter([
        "",    # display mode -> default (matplotlib)
        "",    # window -> default (entire flight)
        "",    # mark peak? -> default (no)
        "",    # action -> default (watch)
    ])
    monkeypatch.setattr("builtins.input", lambda *a: next(inputs))
    monkeypatch.setattr(menu, "pick_file", fake_pick_file)

    choices = menu.run_menu()

    assert choices["hr_csv"] == "/fake/hr.csv"
    assert choices["lr_csv"] is None
    assert choices["obj"] is None
    assert choices["renderer"] == "matplotlib"
    assert choices["window"] is None          # entire flight is the default now
    assert choices["highlight_peak"] is False  # no failure-implying red by default
    assert choices["record"] is None


def test_menu_picks_lr_and_obj_and_exports(monkeypatch):
    picked = {"calls": []}

    def fake_pick_file(title, *a, **k):
        picked["calls"].append(title)
        if "HIGH-RATE" in title:
            return "/fake/hr.csv"
        if "LOW-RATE" in title:
            return "/fake/lr.csv"
        return "/fake/model.obj"

    inputs = iter([
        "1",   # display mode -> matplotlib (still first option)
        "2",   # window -> zoom to peak acceleration
        "1",   # mark peak? -> no
        "2",   # action -> export
        "1",   # format -> mp4
        "",    # filename -> default
    ])
    monkeypatch.setattr("builtins.input", lambda *a: next(inputs))
    monkeypatch.setattr(menu, "pick_file", fake_pick_file)

    choices = menu.run_menu()

    assert choices["hr_csv"] == "/fake/hr.csv"
    assert choices["lr_csv"] == "/fake/lr.csv"
    assert choices["obj"] == "/fake/model.obj"
    assert choices["window"] == "peak"
    assert choices["record"] == "blueraven_clip.mp4"
    # HR, LR, and OBJ dialogs all fire unprompted - no yes/no gating first
    assert len(picked["calls"]) == 3


def test_menu_prompts_for_all_three_files_unconditionally(monkeypatch):
    """The whole point of the latest revision: no "do you have an LR/OBJ
    file?" gating question - all three file dialogs always fire."""
    seen_titles = []

    def fake_pick_file(title, *a, **k):
        seen_titles.append(title)
        return "/fake/hr.csv" if "HIGH-RATE" in title else None

    monkeypatch.setattr("builtins.input", lambda *a: "")
    monkeypatch.setattr(menu, "pick_file", fake_pick_file)

    menu.run_menu()

    assert any("HIGH-RATE" in t for t in seen_titles)
    assert any("LOW-RATE" in t for t in seen_titles)
    assert any("OBJ" in t for t in seen_titles)


def test_menu_export_filename_gets_the_right_extension_even_without_one(monkeypatch):
    """Regression guard: typing a custom filename without an extension (an
    easy mistake - the prompt shows a default *with* one, but nothing stops
    you typing over it with a bare name) used to reach the renderer and
    crash deep inside a third-party library with a bare `KeyError: None`."""
    def fake_pick_file(title, *a, **k):
        return "/fake/hr.csv" if "HIGH-RATE" in title else None

    inputs = iter(["", "", "", "2", "1", "my_clip"])   # export, mp4, no extension typed
    monkeypatch.setattr("builtins.input", lambda *a: next(inputs))
    monkeypatch.setattr(menu, "pick_file", fake_pick_file)

    choices = menu.run_menu()

    assert choices["record"] == "my_clip.mp4"


def test_menu_export_filename_wrong_extension_gets_corrected(monkeypatch):
    def fake_pick_file(title, *a, **k):
        return "/fake/hr.csv" if "HIGH-RATE" in title else None

    inputs = iter(["", "", "", "2", "2", "clip.mp4"])   # export, GIF chosen, but typed .mp4
    monkeypatch.setattr("builtins.input", lambda *a: next(inputs))
    monkeypatch.setattr(menu, "pick_file", fake_pick_file)

    choices = menu.run_menu()

    assert choices["record"] == "clip.gif"


def test_menu_exits_cleanly_if_no_hr_file_selected(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "")
    monkeypatch.setattr(menu, "pick_file", lambda *a, **k: None)

    try:
        menu.run_menu()
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert exc.code == 0
