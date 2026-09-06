"""Tests for the screen_view tool: hook plumbing, headless honesty, and a
real capture against a live OpenCodeTUI in Textual's test harness.
"""

from __future__ import annotations

import pytest

from opencode_py.tools.screen_view import set_capture_fn, tool


@pytest.fixture(autouse=True)
def clean_hook():
    """Every test starts and ends with no capture hook installed."""
    set_capture_fn(None)
    yield
    set_capture_fn(None)


def _run(**kw) -> dict:
    return tool().run(kw)


# ------------------------------------------------------------------ plumbing


def test_registered_in_registry():
    from opencode_py.tools import build_registry

    reg = build_registry()
    t = reg.get("screen_view")
    assert t is not None
    assert t.permission == "screen_view"


def test_headless_reports_no_screen():
    res = _run(action="text")
    assert res.get("error") is True
    assert "No TUI" in res["output"] or "headless" in res["output"]


def test_unknown_action_errors():
    res = _run(action="explode")
    assert res.get("error") is True


def test_hook_passthrough_text_and_info():
    calls = []

    def fake(action: str) -> dict:
        calls.append(action)
        if action == "text":
            return {"output": "FAKE SCREEN", "metadata": {}}
        return {"output": "INFO", "metadata": {}}

    set_capture_fn(fake)
    assert _run(action="text")["output"] == "FAKE SCREEN"
    assert _run(action="info")["output"] == "INFO"
    assert calls == ["text", "info"]


# ------------------------------------------------------- live TUI integration


async def test_live_tui_widgets_tree():
    """ui_probe: the bones under the screen."""
    from opencode_py.tui.app import OpenCodeTUI

    app = OpenCodeTUI()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        res = _run(action="widgets")
        assert res.get("error") is not True, res["output"]
        out = res["output"]
        # real chrome present with geometry
        assert "ChatView" in out
        assert "InputBar" in out
        assert "StatusBar" in out
        assert "#chat-stack" in out
        assert "(0,0 100×30)" in out or "(0,0 100x30)" in out.replace("×", "x")
        # focused marker lands on the input area's inner text widget
        assert "▸FOCUSED" in out
        assert "PromptTextArea" in out
        # footer summary sane
        assert "[21 widgets" in out or "[22 widgets" in out or "widgets, screen 100x30" in out


async def test_live_tui_widgets_hidden_marker():
    from textual.widgets import Static

    from opencode_py.tui.app import OpenCodeTUI

    app = OpenCodeTUI()
    async with app.run_test(size=(90, 26)) as pilot:
        await pilot.pause()
        ghost = Static("ghost")
        ghost.display = "none"
        await app.screen.mount(ghost)
        await pilot.pause()
        res = _run(action="widgets")
        assert "✗hidden" in res["output"]


def test_headless_widgets_reports_no_screen():
    res = _run(action="widgets")
    assert res.get("error") is True
    assert "No TUI" in res["output"] or "headless" in res["output"]


# ------------------------------------------------------------------ plumbing


def test_hook_passthrough_text_and_info():
    calls = []

    def fake(action: str) -> dict:
        calls.append(action)
        if action in ("text", "widgets"):
            return {"output": f"FAKE {action.upper()}", "metadata": {}}
        return {"output": "INFO", "metadata": {}}

    set_capture_fn(fake)
    assert _run(action="text")["output"] == "FAKE TEXT"
    assert _run(action="widgets")["output"] == "FAKE WIDGETS"
    assert _run(action="info")["output"] == "INFO"
    assert calls == ["text", "widgets", "info"]


async def test_live_tui_text_capture_shows_layout():
    from opencode_py.tui.app import OpenCodeTUI

    app = OpenCodeTUI()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # on_mount must have installed the eyes
        res = _run(action="text")
        assert res.get("error") is not True, res["output"]
        lines = [ln for ln in res["output"].splitlines() if ln.strip()]
        assert lines, "screen capture was empty"
        assert len(lines) <= 400
        assert "[100x30 cells]" in res["output"]
        # the rendered grid must actually contain visible chrome
        assert any("opencode" in ln.lower() for ln in lines)


async def test_live_tui_info_capture():
    from opencode_py.tui.app import OpenCodeTUI

    app = OpenCodeTUI()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        res = _run(action="info")
        assert res.get("error") is not True
        assert "Terminal: 80x24 cells" in res["output"]
        # focus lives on the input area inside InputBar; report what is real
        assert "Focused: PromptTextArea" in res["output"]
        assert "prompt-input" in res["output"]
        assert res["metadata"]["width"] == 80


async def test_capture_reflects_content_changes():
    """The point of the tool: what the model sees tracks what is on screen."""
    from opencode_py.tui.chat_view import ChatView, MessageBubble
    from opencode_py.tui.app import OpenCodeTUI

    app = OpenCodeTUI()
    async with app.run_test(size=(90, 26)) as pilot:
        await pilot.pause()
        chat = app.query_one(ChatView)
        marker = "XYZZY-CAPTURE-MARKER"
        chat.append_meta(f"{marker} hello")
        await pilot.pause(0.3)
        res = _run(action="text")
        assert marker in res["output"]


# ------------------------------------------------------------- screenshot


def test_screenshot_without_binary_reports_honestly(monkeypatch):
    import opencode_py.tools.screen_view as sv

    if sv._screencap_binary() is None:
        res = _run(action="screenshot")
        assert res.get("error") is True
        assert "screencap" in res["output"].lower()


def test_tui_labels_registered():
    from opencode_py.tui.chat_view import TOOL_ICONS, TOOL_NAMES

    assert TOOL_NAMES.get("screen_view") == "Screen"
    assert "screen_view" in TOOL_ICONS