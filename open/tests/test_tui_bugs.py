"""TUI bug regressions (trimmed): the fast, harness-free unit tests that
cover interrupt wiring, unmounted-widget no-ops and the write-tool metadata.
The full headless Textual runtime coverage lives in this file's git history."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from opencode_py.agent.loop import AgentLoop
from opencode_py.config import Config
from opencode_py.tools import build_registry
from opencode_py.tools.write import _write
from opencode_py.tui.model_picker import ModelPicker
from opencode_py.tui.settings_screen import SettingsScreen


class FakeEngine:
    agent = "build"
    permission = SimpleNamespace(mode="auto")


# --------------------------------------------------------------------------
# Bug 5: registry interrupt check must follow a re-wired engine.interrupt —
# wires engine.interrupt AFTER construction, so a plain attribute froze the
# hook to the init default (a `lambda: False`) and ESC kept being ignored while
# a command ran.
# --------------------------------------------------------------------------

def test_wiring_updates_registry_interrupt_check():
    reg = build_registry(Config())
    engine = AgentLoop(cfg=Config(), registry=reg, directory=Path("."))
    flag = {"on": False}
    engine.interrupt = lambda: flag["on"]  # what _wire_engine does
    assert reg.interrupt_check() is False
    flag["on"] = True  # what a 2nd ESC press does
    assert reg.interrupt_check() is True


def test_engine_interrupt_honors_shared_flag():
    """A sub-agent spawned from the app engine must share the interrupt flag."""
    parent = AgentLoop(cfg=Config(), registry=SimpleNamespace(), directory=Path("."))
    flag = {"requested": False}
    parent.interrupt = lambda: flag["requested"]
    assert parent.interrupt() is False
    flag["requested"] = True
    assert parent.interrupt() is True


# --------------------------------------------------------------------------
# Bug 1: ModelPicker / SettingsScreen deferred refresh after dismissal
# --------------------------------------------------------------------------

def test_model_picker_populate_when_not_attached_noop():
    picker = ModelPicker()
    assert picker.is_attached is False
    picker.populate({})  # must not raise NoMatches


def test_model_picker_set_loading_when_not_attached_noop():
    picker = ModelPicker()
    picker.set_loading()  # must not raise


def test_settings_render_when_not_attached_noop():
    screen = SettingsScreen(cfg=Config(), engine=FakeEngine(), auth=None)
    assert screen.is_attached is False
    screen._render_settings()  # must not raise NoMatches
    screen._keep_selection_visible()  # must not raise


# --------------------------------------------------------------------------
# Bug 11: write tool returns the written content for the TUI block
# --------------------------------------------------------------------------

def test_write_tool_returns_content_metadata(tmp_path):
    target = tmp_path / "hello.py"
    result = _write(str(target), "print('hi')\n")
    assert result["output"] == "Wrote file successfully."
    assert result["metadata"]["content"] == "print('hi')\n"
    assert result["metadata"]["filePath"] == str(target.resolve())
    assert target.read_text() == "print('hi')\n"
