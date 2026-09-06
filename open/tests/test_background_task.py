"""Tests for the background_task tool: lifecycle, incremental read, stop/wait.

Uses short-lived shell commands so the suite stays fast; the only "long" task
is a sleep that is explicitly stopped.
"""

from __future__ import annotations

import time

import pytest


@pytest.fixture
def bgt(monkeypatch):
    """Fresh task registry per test (clears globals)."""
    from opencode_py.tools import background as bg

    bg._TASKS.clear()
    return bg


def _run(**kw) -> dict:
    from opencode_py.tools.background import tool

    return tool().run(kw)


def _wait_output(bgt, task_id: str, want: str, timeout_s: float = 8.0) -> dict:
    """Poll `read` until `want` appears (reader thread is async)."""
    deadline = time.monotonic() + timeout_s
    last = {}
    while time.monotonic() < deadline:
        last = _run(action="read", task_id=task_id)
        if want in last["output"]:
            return last
        time.sleep(0.05)
    raise AssertionError(f"output {want!r} not seen; last={last!r}")


# ------------------------------------------------------------------ lifecycle


def test_start_returns_id_and_runs(bgt):
    res = _run(action="start", command="sleep 30")
    assert res.get("error") is not True, res["output"]
    tid = res["metadata"]["task_id"]
    st = _run(action="status", task_id=tid)
    assert "RUNNING" in st["output"]
    # cleanup
    _run(action="stop", task_id=tid)


def test_quick_command_finishes_with_exit_code_and_tail(bgt):
    res = _run(action="start", command="echo hello-bg")
    tid = res["metadata"]["task_id"]
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        st = _run(action="status", task_id=tid)
        if "FINISHED" in st["output"]:
            break
        time.sleep(0.05)
    w = _run(action="wait", task_id=tid, timeout=5000)
    assert w.get("error") is not True
    assert "Exit code: 0" in w["output"]
    assert "hello-bg" in w["output"]


def test_incremental_read_returns_only_new_bytes(bgt):
    res = _run(action="start", command="printf 'first\\n'; sleep 1.2; printf 'second\\n'")
    tid = res["metadata"]["task_id"]
    r1 = _wait_output(bgt, tid, "first")
    assert "second" not in r1["output"]
    r2 = _wait_output(bgt, tid, "second")
    assert "first" in r2["output"] or r2["metadata"]["bytes_returned"] > 0
    assert "second" in r2["output"]


# ------------------------------------------------------------------ control


def test_stop_kills_running_task(bgt):
    res = _run(action="start", command="sleep 60")
    tid = res["metadata"]["task_id"]
    time.sleep(0.15)  # let it spawn
    s = _run(action="stop", task_id=tid)
    assert s.get("error") is not True, s["output"]
    assert s["metadata"]["stopped"] is True
    st = _run(action="status", task_id=tid)
    assert "FINISHED" in st["output"] or "reaping" in st["output"]


def test_stop_all_kills_everything_running(bgt):
    """2nd-ESC force-stop: stop_all() ends all running tasks, never raises."""
    from opencode_py.tools import background as bg

    assert bg.stop_all() == 0
    tids = []
    for _ in range(3):
        tids.append(_run(action="start", command="sleep 60")["metadata"]["task_id"])
    time.sleep(0.2)
    assert bg.stop_all() == 3
    assert bg.stop_all() == 0
    for tid in tids:
        assert "FINISHED" in _run(action="status", task_id=tid)["output"]


def test_wait_returns_promptly_after_finish(bgt):
    """Event wait (not 100ms polling): wait() returns right after exit."""
    res = _run(action="start", command="echo done")
    tid = res["metadata"]["task_id"]
    out = _run(action="wait", task_id=tid, timeout_ms=10000)
    assert out.get("error") is not True
    assert "done" in out["output"]


def test_stop_already_finished_reports_cleanly(bgt):
    res = _run(action="start", command="true")
    tid = res["metadata"]["task_id"]
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if "FINISHED" in _run(action="status", task_id=tid)["output"]:
            break
        time.sleep(0.05)
    s = _run(action="stop", task_id=tid)
    assert s.get("error") is not True
    assert "already finished" in s["output"]


def test_wait_timeout_still_running(bgt, monkeypatch):
    res = _run(action="start", command="sleep 45")
    tid = res["metadata"]["task_id"]
    try:
        w = _run(action="wait", task_id=tid, timeout=1500)
        assert w["metadata"].get("timed_out") is True
        assert "STILL RUNNING" in w["output"]
    finally:
        _run(action="stop", task_id=tid)


# ------------------------------------------------------------------- queries


def test_list_shows_tasks(bgt):
    r = _run(action="start", command="echo listed")
    tid = r["metadata"]["task_id"]
    lst = _run(action="list")
    assert tid in lst["output"]
    _run(action="wait", task_id=tid, timeout=8000)


def test_unknown_task_id_errors(bgt):
    for action in ("status", "read", "stop", "wait"):
        res = _run(action=action, task_id="bg999")
        assert res.get("error") is True, action


def test_empty_start_errors(bgt):
    res = _run(action="start", command="   ")
    assert res.get("error") is True


def test_bad_workdir_errors(bgt):
    res = _run(action="start", command="true", workdir="/no/such/dir/xyz")
    assert res.get("error") is True


def test_unknown_action_errors(bgt):
    res = _run(action="explode")
    assert res.get("error") is True


def test_running_cap_refuses_ninth(bgt, monkeypatch):
    monkeypatch.setattr(bgt, "MAX_RUNNING", 1)
    r1 = _run(action="start", command="sleep 30")
    assert r1.get("error") is not True
    r2 = _run(action="start", command="sleep 30")
    assert r2.get("error") is True
    assert "already running" in r2["output"]
    _run(action="stop", task_id=r1["metadata"]["task_id"])


# ------------------------------------------------------------ registry glue


def test_registered_in_registry():
    from opencode_py.tools import build_registry

    reg = build_registry()
    t = reg.get("background_task")
    assert t is not None
    assert t.permission == "background_task"


def test_tui_labels():
    from opencode_py.tui.chat_view import TOOL_ICONS, TOOL_NAMES

    assert TOOL_NAMES.get("background_task") == "Background"
    assert "background_task" in TOOL_ICONS