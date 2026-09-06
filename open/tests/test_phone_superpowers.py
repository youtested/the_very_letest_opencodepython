"""Tests for the device, history_search, and checkpoint tools.

- device: fake Termux binaries on PATH, sysfs battery fallback, honest errors.
- history_search: real-format session files under a patched data dir; search
  ranking (AND terms), list, read-by-prefix.
- checkpoint: snapshot -> mutate -> diff -> rollback round-trip with exact
  bytes, safety snapshot, cross-root refusal, retention pruning.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from opencode_py.globals import Path as GPath


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_fake_bin(tmp_path: Path, scripts: dict[str, str]) -> Path:
    """Create executable shell-script 'binaries' and return the dir."""
    import shutil as _shutil

    bindir = tmp_path / "fakebin"
    bindir.mkdir(exist_ok=True)
    shell = os.environ.get("SHELL") or _shutil.which("sh") or _shutil.which("bash")
    assert shell, "no shell available for fake binaries"
    for name, body in scripts.items():
        p = bindir / name
        # Termux has no /bin/sh — point the shebang at a shell that exists
        p.write_text(f"#!{shell}\n" + body, encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bindir


@pytest.fixture
def fake_path(tmp_path, monkeypatch):
    def install(scripts: dict[str, str]) -> Path:
        bindir = _make_fake_bin(tmp_path, scripts)
        monkeypatch.setenv("PATH", str(bindir))
        return bindir

    return install


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------

class TestDevice:
    def test_wake_lock_and_unlock(self, fake_path):
        from opencode_py.tools.device import _WAKE_HELD, tool

        fake_path({
            "termux-wake-lock": "exit 0\n",
            "termux-wake-unlock": "exit 0\n",
        })
        t = tool()
        r1 = t.run({"action": "wake_lock"})
        assert r1.get("error") is not True
        assert _WAKE_HELD.is_set()
        r2 = t.run({"action": "wake_unlock"})
        assert r2.get("error") is not True
        assert not _WAKE_HELD.is_set()

    def test_battery_via_termux_api(self, fake_path):
        from opencode_py.tools.device import tool

        fake_path({
            "termux-battery-status": 'echo \'{"percentage": 77, "status": "DISCHARGING"}\'\n',
        })
        res = tool().run({"action": "battery"})
        assert res.get("error") is not True
        assert "77%" in res["output"]
        assert res["metadata"]["source"] == "termux-api"

    def test_battery_sysfs_fallback(self, tmp_path, monkeypatch):
        import opencode_py.tools.device as dev

        ps = tmp_path / "power_supply" / "battery"
        ps.mkdir(parents=True)
        (ps / "capacity").write_text("42\n")
        (ps / "status").write_text("Charging\n")
        monkeypatch.setattr(dev, "_SYSFS_POWER", str(ps.parent))
        monkeypatch.setattr(dev.shutil, "which", lambda _: None)

        res = dev.tool().run({"action": "battery"})
        assert res.get("error") is not True
        assert "42%" in res["output"]
        assert res["metadata"]["source"] == "sysfs"

    def test_missing_binaries_report_install_hint(self, tmp_path, monkeypatch):
        from opencode_py.tools.device import tool

        empty = tmp_path / "nobin"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))
        t = tool()
        for action in ("wake_lock", "vibrate"):
            res = t.run({"action": action})
            assert res.get("error") is True
            assert "pkg install" in res["output"]

    def test_vibrate_duration_clamped(self, fake_path):
        from opencode_py.tools.device import tool

        fake_path({"termux-vibrate": "exit 0\n"})
        res = tool().run({"action": "vibrate", "duration": 99999})
        assert res.get("error") is not True
        assert "5000 ms" in res["output"]

    def test_unknown_action(self):
        from opencode_py.tools.device import tool

        assert tool().run({"action": "fly"}).get("error") is True


# ---------------------------------------------------------------------------
# history_search
# ---------------------------------------------------------------------------

def _write_session(sid: str, title: str, created: float, messages: list) -> None:
    data = {
        "id": sid,
        "title": title,
        "created": created,
        "completed": created + 60,
        "directory": "/tmp/proj",
        "provider": "p",
        "model": "m",
        "agent": "build",
        "messages": messages,
    }
    path = GPath.sessions_dir() / f"{sid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "data", tmp_path)
    from opencode_py.session import clear_session_cache

    clear_session_cache()
    yield tmp_path / "sessions"
    clear_session_cache()


class TestHistorySearch:
    def test_search_finds_and_ranks(self, sessions_dir):
        from opencode_py.tools.history_search import tool

        old = time.time() - 86400 * 3
        _write_session(
            "aaa11111", "sse timeout hunt", old,
            [
                {"role": "user", "content": "fix the SSE timeout bug in providers"},
                {"role": "assistant",
                 "content": "the SSE timeout was raised to 300s and we added a keepalive"},
            ],
        )
        _write_session(
            "bbb22222", "later fix", time.time(),
            [
                {"role": "user", "content": "again about sse timeout"},
                {"role": "assistant", "content": "sse timeout patched twice here"},
            ],
        )
        _write_session(
            "ccc33333", "unrelated", time.time() - 100,
            [{"role": "user", "content": "what is love"}],
        )
        res = tool().run({"query": "sse timeout"})
        assert res.get("error") is not True
        meta = res["metadata"]
        assert meta["matches"] == 2
        # newer session first on equal score; bbb has score 4 vs aaa 3 anyway
        assert meta["items"][0]["id"].startswith("bbb")

    def test_search_all_terms_required(self, sessions_dir):
        from opencode_py.tools.history_search import tool

        _write_session("ddd44444", "only one term", time.time(),
                       [{"role": "user", "content": "talk about walrus"}])
        res = tool().run({"query": "walrus zanzibar"})
        assert res["metadata"]["matches"] == 0

    def test_list_shows_sessions(self, sessions_dir):
        from opencode_py.tools.history_search import tool

        _write_session("eee55555", "a session", time.time(),
                       [{"role": "user", "content": "hi"}])
        res = tool().run({"action": "list"})
        assert "eee55555" in res["output"]

    def test_read_by_id_prefix(self, sessions_dir):
        from opencode_py.tools.history_search import tool

        _write_session("fff66666", "readable one", time.time(), [
            {"role": "user", "content": "tell me about yaks"},
            {"role": "assistant", "content": "yaks are shaggy oxen of the himalayas"},
        ])
        res = tool().run({"action": "read", "id": "fff666"})
        assert res.get("error") is not True
        assert "[user]" in res["output"] and "yaks" in res["output"]

    def test_read_ambiguous_errors(self, sessions_dir):
        from opencode_py.tools.history_search import tool

        now = time.time()
        _write_session("aaa77777", "t1", now, [])
        _write_session("aaa88888", "t2", now - 10, [])
        res = tool().run({"action": "read", "id": "aaa"})
        assert res.get("error") is True

    def test_empty_archive(self, sessions_dir):
        from opencode_py.tools.history_search import tool

        res = tool().run({"query": "anything"})
        assert res.get("error") is not True
        assert res["metadata"]["matches"] == 0


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

@pytest.fixture
def ckpt_env(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "data", tmp_path)
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    (repo / "a.txt").write_bytes(b"one\ntwo\nthree\n")
    (repo / "sub" / "c.txt").write_bytes(b"crlf\r\nlines\r\n")
    monkeypatch.chdir(repo)
    from opencode_py.tools.checkpoint import _load_manifest

    return repo


def _ckpt(**kw):
    from opencode_py.tools.checkpoint import tool

    return tool().run(kw)


class TestCheckpoint:
    def test_take_diff_rollback_roundtrip(self, ckpt_env):
        take = _ckpt(action="take", label="before chaos")
        assert take.get("error") is not True, take["output"]
        cid = take["metadata"]["id"]

        # chaos: modify one, create one, delete one
        (ckpt_env / "a.txt").write_bytes(b"CHANGED\n")
        (ckpt_env / "new.txt").write_bytes(b"brand new\n")
        (ckpt_env / "sub" / "c.txt").unlink()

        diff = _ckpt(action="diff")
        out = diff["output"]
        assert "modified (1)" in out and "a.txt" in out
        assert "added (1)" in out and "new.txt" in out
        assert "deleted (1)" in out and "c.txt" in out

        rb = _ckpt(action="rollback")
        assert rb.get("error") is not True, rb["output"]
        # exact byte-level restoration (CRLF preserved)
        assert (ckpt_env / "a.txt").read_bytes() == b"one\ntwo\nthree\n"
        assert (ckpt_env / "sub" / "c.txt").read_bytes() == b"crlf\r\nlines\r\n"
        assert not (ckpt_env / "new.txt").exists()
        # safety snapshot exists so the rollback itself is reversible
        assert rb["metadata"]["safety_id"]
        names = json.dumps(_ckpt(action="list")["output"])
        assert "pre-rollback" in names

    def test_rollback_of_rollback(self, ckpt_env):
        _ckpt(action="take", label="first")
        (ckpt_env / "a.txt").write_bytes(b"second state\n")
        rb1 = _ckpt(action="rollback")
        assert rb1.get("error") is not True
        # now undo that rollback via its safety snapshot (newest)
        rb2 = _ckpt(action="rollback")
        assert rb2.get("error") is not True
        assert (ckpt_env / "a.txt").read_bytes() == b"second state\n"

    def test_clean_tree_reports_no_changes(self, ckpt_env):
        take = _ckpt(action="take")
        diff = _ckpt(action="diff")
        assert "matches the checkpoint exactly" in diff["output"]
        assert take["metadata"]["truncated"] is False

    def test_cross_project_refusal_and_force(self, ckpt_env, tmp_path, monkeypatch):
        _ckpt(action="take", label="in repo")
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)
        res = _ckpt(action="rollback")
        assert res.get("error") is True
        assert "Refusing" in res["output"]
        ok = _ckpt(action="rollback", force=True)
        assert ok.get("error") is not True

    def test_drop_removes_checkpoint(self, ckpt_env):
        take = _ckpt(action="take")
        cid = take["metadata"]["id"]
        res = _ckpt(action="drop", id=cid[:12])
        assert res.get("error") is not True
        assert cid[:14] not in _ckpt(action="list")["output"]

    def test_retention_prunes_oldest(self, ckpt_env, monkeypatch):
        from opencode_py.tools import checkpoint as cp

        monkeypatch.setattr(cp, "MAX_CHECKPOINTS", 3)
        ids = []
        for i in range(5):
            (ckpt_env / "a.txt").write_bytes(f"state {i}\n".encode())
            r = _ckpt(action="take", label=f"s{i}")
            ids.append(r["metadata"]["id"])
        listing = _ckpt(action="list")["output"]
        assert len(cp._load_manifest()) == 3
        assert ids[0] not in listing
        assert ids[1] not in listing
        assert ids[-1] in listing

    def test_binary_files_skipped(self, ckpt_env):
        (ckpt_env / "blob.bin").write_bytes(b"\x00\x01\x02\xff\xfe\x00" * 10)
        res = _ckpt(action="take")
        assert res.get("error") is not True
        # binary never enters the snapshot: rollback won't touch it
        (ckpt_env / "blob.bin").write_bytes(b"\x09\x08\x07\x06\x05\x04" * 10)
        _ckpt(action="rollback")
        assert (ckpt_env / "blob.bin").read_bytes() == b"\x09\x08\x07\x06\x05\x04" * 10

    def test_registered_in_registry(self):
        from opencode_py.tools import build_registry

        reg = build_registry()
        assert reg.get("checkpoint") is not None
        assert reg.get("device") is not None
        assert reg.get("history_search") is not None