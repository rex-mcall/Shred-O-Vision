"""The menu wizard is interactive (input() + native file dialogs), so these
tests drive it by mocking both - verifying the question flow and the choices
it hands back to the CLI, not the actual dialog/terminal behavior."""

import blueraven_visualizer.menu as menu


def test_menu_all_defaults_skips_lr_and_obj(monkeypatch):
    inputs = iter([
        "",    # press Enter to open the HR file dialog
        "2",   # LR file? -> No
        "",    # OBJ model? -> default (No, use generic shape)
        "",    # display mode -> default (matplotlib)
        "",    # window -> default (shred)
        "",    # action -> default (watch)
    ])
    monkeypatch.setattr("builtins.input", lambda *a: next(inputs))
    monkeypatch.setattr(menu, "pick_file", lambda *a, **k: "/fake/hr.csv")

    choices = menu.run_menu()

    assert choices["hr_csv"] == "/fake/hr.csv"
    assert choices["lr_csv"] is None
    assert choices["obj"] is None
    assert choices["renderer"] == "matplotlib"
    assert choices["window"] == "shred"
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
        "",    # press Enter to open the HR file dialog
        "1",   # LR file? -> Yes
        "",    # press Enter to open the LR file dialog
        "2",   # OBJ model? -> Yes
        "",    # press Enter to open the OBJ file dialog
        "1",   # display mode -> matplotlib (still first option)
        "2",   # window -> entire flight
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
    assert choices["window"] is None
    assert choices["record"] == "blueraven_clip.mp4"
    assert len(picked["calls"]) == 3


def test_menu_exits_cleanly_if_no_hr_file_selected(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "")
    monkeypatch.setattr(menu, "pick_file", lambda *a, **k: None)

    try:
        menu.run_menu()
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert exc.code == 0
