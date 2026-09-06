"""TUI bug-fix regressions (trimmed): the fast, harness-free unit tests —
raw config-key preservation, picker label/format helpers, MessageBubble
constructor behavior. The full headless Textual runtime coverage lives in
this file's git history."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from opencode_py.config import Config, save_config
from opencode_py.tui.chat_view import MessageBubble, collapse_tool_output
from opencode_py.tui.settings_screen import SettingsScreen


class FakeEngine:
    agent = "build"
    permission = type("P", (), {"mode": "auto"})()


# --------------------------------------------------------------------------
# Config: save_config must preserve unknown raw keys (mcpServers/plugins/tools).
# --------------------------------------------------------------------------

def test_save_config_preserves_raw_keys(tmp_path):
    cfg = Config.from_dict(
        {
            "model": "opencode/foo",
            "mcpServers": {"local": {"command": "npx"}},
            "plugins": ["@opencode/plugin-ts"],
            "tools": {"bash": {"deny": "*"}},
        },
        Path("."),
    )
    p = tmp_path / "opencode.json"
    save_config(cfg, path=p)
    data = json.loads(p.read_text())
    assert data["model"] == "opencode/foo"
    assert data["mcpServers"] == {"local": {"command": "npx"}}
    assert data["plugins"] == ["@opencode/plugin-ts"]
    assert data["tools"] == {"bash": {"deny": "*"}}


def test_save_config_known_keys_override_raw():
    cfg = Config.from_dict({"model": "opencode/old", "theme": "solarized"}, Path("."))
    cfg.theme = "opencode"
    p = Path(tempfile.mkdtemp()) / "opencode.json"
    save_config(cfg, path=p)
    data = json.loads(p.read_text())
    assert data["theme"] == "opencode"


# --------------------------------------------------------------------------
# Model picker: "128k"-style context strings must not crash int().
# --------------------------------------------------------------------------

def test_format_context_handles_k_and_junk():
    from opencode_py.tui.model_picker import _format_context

    assert _format_context(128000) == "128,000"
    assert _format_context("128k") == "128,000"
    assert _format_context("1m") == "1,000,000"
    assert _format_context("junk") == "junk"
    assert _format_context(None) == "?"
    assert _format_context(0) == "0"


# --------------------------------------------------------------------------
# Chat view: long write output collapses.
# --------------------------------------------------------------------------

def test_write_render_collapses_long_content():
    long = "\n".join(f"line {i}" for i in range(200))
    collapsed = collapse_tool_output(long, 10, 10 * 80)
    assert collapsed["overflow"] is True
    assert "line 199" not in collapsed["output"]
    short = collapse_tool_output("tiny", 10, 10 * 80)
    assert short["overflow"] is False
    assert short["output"] == "tiny"


# --------------------------------------------------------------------------
# Settings: the "small model" picker must not retarget the app engine.
# --------------------------------------------------------------------------

def test_small_model_row_does_not_propagate():
    screen = SettingsScreen(cfg=Config(), engine=FakeEngine(), auth=None)
    rows = screen._build_rows()
    model_row = next(r for r in rows if r.label == "model")
    small_row = next(r for r in rows if r.label == "small model")
    assert model_row.propagate is True
    assert small_row.propagate is False


# --------------------------------------------------------------------------
# Model picker row labels: FREE tag, current-model bullet.
# --------------------------------------------------------------------------

def test_picker_row_label_free_and_current():
    from opencode_py.tui.model_picker import _model_row_label

    row = _model_row_label("opencode/deepseek-v4-flash-free", {"id": "deepseek-v4-flash-free", "name": "DeepSeek V4 Flash", "free": True}, "opencode/deepseek-v4-flash-free")
    text = row.render().plain
    assert "FREE" in text
    assert "DeepSeek V4 Flash" in text
    # current model marked with the bullet
    assert "\u25cf" in text


def test_picker_row_label_free_sort_key():
    from opencode_py.tui.model_picker import _model_row_label

    free_row = _model_row_label("p/a", {"id": "a", "free": True}, "")
    paid_row = _model_row_label("p/b", {"id": "b", "free": False}, "")
    assert "FREE" in free_row.render().plain
    assert "FREE" not in paid_row.render().plain


def test_picker_does_not_shadow_textual_render():
    from opencode_py.tui.model_picker import ModelPicker

    assert ModelPicker._render.__qualname__ == "Widget._render"
    # the method that builds the list rows must exist under the renamed id
    assert hasattr(ModelPicker, "_populate_list")


def test_picker_css_is_attached_to_class():
    from opencode_py.tui.model_picker import ModelPicker

    assert hasattr(ModelPicker, "CSS") and ModelPicker.CSS.strip()
    assert "#models-search" in ModelPicker.CSS
    assert "group-header" in ModelPicker.CSS


# --------------------------------------------------------------------------
# MessageBubble constructor args were silently ignored (dead code below a
# `return {}`), so queued/streaming/focus never applied.
# --------------------------------------------------------------------------

def test_message_bubble_constructor_applies_queued_streaming_focus():
    b = MessageBubble("reasoning", "**Title**\n\nbody", queued=True, streaming=True)
    assert b.can_focus is True
    assert b.queued is True
    assert b.streaming is True


def test_message_bubble_only_reasoning_is_focusable():
    assert MessageBubble("reasoning", "").can_focus is True
    assert MessageBubble("assistant", "hi").can_focus is False
    assert MessageBubble("user", "hi").can_focus is False
