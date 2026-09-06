"""screen_view tool: let the model SEE the TUI it is building/debugging.

The model writes a full-screen Textual app but is blind to its own layout —
"the header is misaligned" used to be pure guesswork from CSS classes. This
tool bridges that gap with four kinds of eyes:

- ``text``      – the CURRENT rendered screen as plain text (exact character
                  grid, borders and alignment included), via the compositor.
- ``widgets``   – the live widget TREE underneath (ui_probe view): type, id,
                  classes, position/size per node, focused/hidden markers.
- ``info``      – terminal size, focused widget, app/screen title.
- ``screenshot`` – best-effort real PNG of the device display via Android's
                  screencap binaries; the path can be opened with `read`.

The capture function is installed by the TUI at startup (module-level, one
screen per process). Outside the TUI the tool answers honestly that there is
nothing to look at instead of pretending.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..globals import Path as GPath
from .registry import Tool, schema_with

_LOCK = threading.Lock()
_CAPTURE_FN: Callable[[str], dict[str, Any]] | None = None

_MAX_TEXT_LINES = 400


def set_capture_fn(fn: Callable[[str], dict[str, Any]] | None) -> None:
    """Install/remove the live capture hook (called by the TUI at startup)."""
    global _CAPTURE_FN
    with _LOCK:
        _CAPTURE_FN = fn


def _capture(action: str) -> dict[str, Any] | None:
    with _LOCK:
        fn = _CAPTURE_FN
    return fn(action) if fn is not None else None


# ---------------------------------------------------------------------------
# Android screenshot back-end (best effort)
# ---------------------------------------------------------------------------

def _screencap_binary() -> str | None:
    for cand in ("termux-screencap",):
        path = shutil.which(cand)
        if path:
            return path
    if os.path.exists("/system/bin/screencap"):
        return "/system/bin/screencap"
    return None


def _take_screenshot() -> dict[str, Any]:
    binary = _screencap_binary()
    if binary is None:
        return {
            "output": (
                "No screencap binary available (need termux-screencap or "
                "/system/bin/screencap). Use action=text instead."
            ),
            "error": True,
        }
    out_dir = GPath.data / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"screen_{int(time.time())}.png"
    try:
        proc = subprocess.run(
            [binary, "-p", str(out_path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return {"output": f"screencap failed: {e}", "error": True}
    if proc.returncode != 0 or not Path(out_path).exists():
        err = (proc.stderr or proc.stdout or "").strip()[:200]
        return {
            "output": (
                f"screencap failed (code {proc.returncode})"
                + (f": {err}" if err else "")
                + ". On many devices this needs root or the Termux:API "
                "permission; use action=text instead."
            ),
            "error": True,
        }
    size_kb = Path(out_path).stat().st_size // 1024
    return {
        "output": (
            f"Screenshot saved: {out_path} ({size_kb} KB).\n"
            "Open it with the read tool (returns an image attachment when "
            "the provider supports vision)."
        ),
        "metadata": {"path": str(out_path), "bytes": size_kb * 1024},
    }


# ---------------------------------------------------------------------------
# tool
# ---------------------------------------------------------------------------

_ACTIONS = ("text", "widgets", "info", "screenshot")


def tool() -> Tool:
    description = """Shows you the CURRENT rendered TUI screen, so you can see the layout you are debugging.

Actions (via `action`):
- text (default): the whole visible screen as plain text — exact rows/columns,
  box borders, spacing. Answers "does it LOOK right?"
- widgets: the live widget TREE underneath — every button/box/label with its
  type, #id, CSS classes, exact position+size in cells, and markers:
  ▸FOCUSED (where keyboard input goes), ✗hidden (exists but not displayed),
  ␀zero-size (collapsed). Answers "WHICH piece is broken and why?" — use it
  whenever text shows a layout problem, to find the culprit widget before
  editing CSS.
- info: terminal width/height in cells, focused widget, app title.
- screenshot: best-effort real PNG of the device display (needs Android
  permission); returns a file path you can open with `read`.

Debugging loop: text (see the symptom) → widgets (find the broken bone) →
fix → text again to confirm.

Only meaningful while the TUI is running; headless runs report that honestly."""

    def run(input: dict) -> dict:
        action = str(input.get("action") or "text").strip().lower()
        if action not in _ACTIONS:
            return {
                "output": f"Unknown action {action!r} (want one of {', '.join(_ACTIONS)}).",
                "error": True,
            }
        if action == "screenshot":
            return _take_screenshot()
        result = _capture(action)
        if result is None:
            return {
                "output": (
                    "No TUI is attached to this process (headless mode?), so "
                    "there is no screen to show. Run inside the opencode_py "
                    "TUI to use this tool."
                ),
                "error": True,
            }
        return result

    return Tool(
        name="screen_view",
        description=description,
        parameters=schema_with(
            {
                "action": {
                    "type": "string",
                    "enum": list(_ACTIONS),
                    "description": "What to capture (default text).",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="screen_view",
    )


__all__ = ["tool", "set_capture_fn"]