"""Regression tests for the /sessions picker + storage hardening fixes.

Fast, harness-free unit tests: pure session-module logic plus headless
SessionList mounts (no full-app run_test).
"""

import json
import time

import pytest

import opencode_py.session as session_mod
from opencode_py.globals import Path as GPath


@pytest.fixture(autouse=True)
def _isolated_sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    session_mod.clear_session_cache()
    yield
    session_mod.clear_session_cache()


# ---------------------------------------------------------------------------
# storage hardening
# ---------------------------------------------------------------------------


def test_list_sessions_survives_corrupt_message_item(tmp_path):
    """A hand-corrupted body with a non-dict inside `messages` must not crash
    list_sessions — and must not poison it permanently (the index used to
    never get written, so EVERY later call re-crashed)."""
    (tmp_path / "bad1.json").write_text(
        json.dumps(
            {
                "id": "bad1",
                "title": "",
                "created": time.time(),
                "messages": ["junk", {"role": "user", "content": "hi"}],
            }
        )
    )
    first = session_mod.list_sessions()
    second = session_mod.list_sessions()
    assert len(first) == len(second) == 1
    # the junk item was dropped, the real message kept
    msgs = first[0].messages
    assert msgs == [{"role": "user", "content": "hi"}]


def test_suggested_title_skips_non_dict_messages():
    title = session_mod.suggested_title(
        {"title": "", "messages": ["junk", {"role": "user", "content": "hello"}]}
    )
    assert title == "hello"


def test_group_sessions_tolerates_nan_and_overflow_created():
    """NaN passes a naive isinstance check but poisons sorting AND explodes
    datetime.fromtimestamp; ±huge finite values overflow the platform."""
    rows = [
        {"id": "nan", "created": float("nan")},
        {"id": "neg", "created": -1e307},
        {"id": "ok", "created": time.time()},
    ]
    groups = session_mod.group_sessions(rows)  # must not raise
    labels = [label for label, _ in groups]
    assert any("Today" in l for l in labels)
    assert any("Unknown date" in l or "1970" in l for l in labels)


def test_session_path_is_injective_for_weird_ids():
    """'a/b', 'a.b' and 'ab' used to all sanitize to ab.json — one save
    silently overwrote another and load_session handed back wrong bodies."""
    p1 = session_mod.session_path("a/b")
    p2 = session_mod.session_path("a.b")
    p3 = session_mod.session_path("ab")
    assert len({p1.name, p2.name, p3.name}) == 3


def test_session_path_keeps_uuid_ids_stable():
    hex_id = "0123456789abcdef0123456789abcdef"
    assert session_mod.session_path(hex_id).name == f"{hex_id}.json"


def test_session_path_traversal_contained():
    name = session_mod.session_path("../../etc/passwd").name
    assert "/" not in name and ".." not in name


def test_save_load_roundtrip_distinct_collision_ids(tmp_path):
    x = session_mod.Session({"id": "a/b", "messages": [{"role": "user", "content": "one"}]})
    y = session_mod.Session({"id": "a.b", "messages": [{"role": "user", "content": "TWO"}]})
    session_mod.save_session(x)
    session_mod.save_session(y)
    assert session_mod.load_session("a/b").messages[0]["content"] == "one"
    assert session_mod.load_session("a.b").messages[0]["content"] == "TWO"


def test_delete_session_cascades_children(tmp_path):
    parent = session_mod.new_session(title="parent")
    kid = session_mod.new_session(title="kid", parent_id=parent.id)
    grand = session_mod.new_session(title="grand", parent_id=kid.id)
    session_mod.save_session(parent)
    session_mod.save_session(kid)
    session_mod.save_session(grand)

    assert session_mod.delete_session(parent.id) is True
    ids = {s.id for s in session_mod.list_sessions()}
    assert kid.id not in ids
    assert grand.id not in ids


def test_delete_session_cleans_ghost_index_entry(tmp_path):
    """A body already missing used to return False WITHOUT cleaning its index
    row — a ghost entry that lingered until the next full rescan."""
    s = session_mod.new_session(title="ghost")
    session_mod.save_session(s)
    s.path.unlink()  # body gone, index row still there

    assert session_mod.delete_session(s.id) is False  # nothing to delete...
    idx = json.loads((tmp_path / ".sessions-index.json").read_text())
    assert s.path.name not in idx["files"]  # ...but the ghost is cleaned


def test_delete_session_cycle_guarded(tmp_path):
    """Hand-edited parent_id cycles must not recurse forever."""
    a = session_mod.new_session(title="a")
    b = session_mod.new_session(title="b", parent_id=a.id)
    b.parent_id = b.id  # self-cycle
    session_mod.save_session(a)
    session_mod.save_session(b)
    assert session_mod.delete_session(a.id) is True


# ---------------------------------------------------------------------------
# registry / command layer
# ---------------------------------------------------------------------------


def test_registry_alias_never_steals_canonical_name():
    from opencode_py.commands import Command, CommandContext, CommandRegistry

    reg = CommandRegistry()
    reg.register(Command("resume", ["sessions"], "Resume", lambda c, a: None))
    reg.register(Command("sessions", [], "List", lambda c, a: None))
    assert reg.get("sessions").name == "sessions"
    assert reg.get("resume").name == "resume"


def test_built_in_registry_has_no_alias_trap():
    from opencode_py.commands import build_registry

    reg = build_registry()
    assert reg.get("sessions").name == "sessions"  # not /resume's alias anymore
    assert "sessions" not in reg.get("resume").aliases


def test_export_preview_only_writes_nothing(tmp_path):
    from opencode_py.commands import (
        CommandContext,
        attach_registry,
        build_registry,
        handle_command,
    )

    sess = session_mod.new_session(directory=str(tmp_path), title="exp")
    sess.messages = [{"role": "user", "content": "hello"}]
    session_mod.save_session(sess)

    class FakeEngine:
        session_id = sess.id

    out = []
    ctx = CommandContext(
        config=None,
        auth=None,
        engine=FakeEngine(),
        worktree=str(tmp_path),
        reply=out.append,
    )
    attach_registry(build_registry(), ctx)
    ctx.preview_only = True
    handle_command(build_registry(), ctx, "/export")
    assert not list(tmp_path.glob("opencode-session-*.md"))
    assert "Will export" in out[0]

    ctx.preview_only = False
    handle_command(build_registry(), ctx, "/export")
    assert list(tmp_path.glob("opencode-session-*.md"))


def test_resume_handler_uses_ctx_callback():
    from opencode_py.commands import CommandContext, attach_registry, build_registry, handle_command

    calls = []
    ctx = CommandContext(config=None, auth=None, reply=lambda t: None, resume=calls.append)
    attach_registry(build_registry(), ctx)
    handle_command(build_registry(), ctx, "/resume abc123")
    assert calls == ["abc123"]


# ---------------------------------------------------------------------------
# history_search tool
# ---------------------------------------------------------------------------


def test_history_search_scans_whole_archive(tmp_path, monkeypatch):
    """The 200-newest cap made old sessions invisible despite the 'ALL
    transcripts' contract."""
    from opencode_py.tools.history_search import tool

    target = session_mod.new_session(title="ancient")
    target.created = time.time() - 400 * 86400
    target.messages = [{"role": "user", "content": "XYZZY_marker secret"}]
    session_mod.save_session(target)
    filler_created = time.time()
    for i in range(210):
        f = session_mod.new_session(title=f"filler{i}")
        f.created = filler_created + i
        f.messages = [{"role": "user", "content": f"filler {i}"}]
        session_mod.save_session(f)
    session_mod.clear_session_cache()

    out = tool().run({"action": "search", "query": "XYZZY_marker"})
    assert "XYZZY_marker" in str(out.get("snippet", "")) or "matching session" in out["output"]


# ---------------------------------------------------------------------------
# delete-on-first-try + stale prune marker (TUI app level, headless)
# ---------------------------------------------------------------------------


def _make_app(tmp_path):
    from opencode_py.config import Config
    from opencode_py.tui.app import OpenCodeTUI

    return OpenCodeTUI(cfg=Config())


def test_delete_removes_live_session_file_on_first_call(tmp_path):
    """A RESUMED session is live in RAM *and* on disk. The old code skipped
    the disk delete for anything live (`True if was_live else ...`), so the
    file survived and the picker's refresh resurrected the row — the user had
    to Ctrl+D twice. One call must be enough."""
    import asyncio

    sess = session_mod.new_session(title="resumed one")
    sess.messages = [{"role": "user", "content": "hello"}]
    session_mod.save_session(sess)
    assert session_mod.load_session(sess.id) is not None

    async def scenario():
        app = _make_app(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.1)
            # register it exactly like _resume_session does (live + chat view)
            app._sessions[sess.id] = sess
            chat = app._chat_for(sess.id)
            assert chat is not None
            ok = app._delete_session(sess.id)
            await pilot.pause(0.05)
            return ok

    ok = asyncio.run(scenario())
    assert ok is True
    session_mod.clear_session_cache()
    assert session_mod.load_session(sess.id) is None, (
        "the durable copy must be gone after ONE delete"
    )


def test_stale_prune_marker_with_existing_file_is_resumable(tmp_path):
    """`_pruned` means 'in-memory chat torn down', NOT 'history vanished'.
    A prune marker plus a live file must resume (and heal the marker) instead
    of answering 'That sub-agent session is finished and closed.'"""
    import asyncio

    sess = session_mod.new_session(title="ghost with body")
    sess.messages = [{"role": "user", "content": "still here"}]
    session_mod.save_session(sess)

    resumed = []

    async def scenario():
        app = _make_app(tmp_path)
        app._pruned.add(sess.id)  # poisoned by an earlier failed delete
        orig_switch = app._switch_session

        def spy(sid):
            resumed.append(sid)
            return orig_switch(sid)

        app._switch_session = spy
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.1)
            app._resume_session(sess.id)
            await pilot.pause(0.2)

    asyncio.run(scenario())
    assert sess.id in resumed
    # marker healed so future navigation works too


def test_pruned_without_any_file_still_refused(tmp_path):
    """Genuinely pruned sub-agent chats (nothing on disk) keep the old
    'finished and closed' refusal."""
    import asyncio

    ran = []
    messages = []

    async def scenario():
        app = _make_app(tmp_path)
        app._pruned.add("gone-child")
        app.notify = lambda m, **k: messages.append(str(m))
        orig = app._switch_session

        def spy(sid):
            ran.append(sid)
            return orig(sid)

        app._switch_session = spy
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.1)
            app._resume_session("gone-child")
            await pilot.pause(0.1)

    asyncio.run(scenario())
    assert ran == []
    assert any("no longer exists" in m for m in messages)


# ---------------------------------------------------------------------------
# picker header buttons (were dead: no on_button_pressed on SessionList)
# ---------------------------------------------------------------------------


def test_session_list_header_buttons_work(tmp_path):
    """Clicking Select must enter select mode ([ ] checkboxes), Save must hit
    the callback for the HIGHLIGHTED row, Delete-sel must open the confirm
    dialog. All three were click-dead — only Ctrl+E/S/D worked."""
    import asyncio

    from textual.app import App
    from textual.widgets import Button
    from opencode_py.tui.session_list import ConfirmDeleteDialog, SessionList

    saved = []
    pushed = []
    rows = [
        {"id": "cur", "title": "cur", "agent": "build", "created": time.time(), "status": ""},
        {"id": "old", "title": "old", "agent": "build", "created": 1000.0, "status": ""},
    ]

    async def scenario():
        class T(App):
            def notify(self, m, **k):
                pass

        app = T()
        async with app.run_test(size=(80, 24)) as pilot:
            sl = SessionList(
                [dict(r) for r in rows],
                current="cur",
                on_delete=lambda sid: True,
                on_save=lambda sid: saved.append(sid) or True,
            )
            await app.push_screen(sl, None)
            await pilot.pause(0.1)

            # Select button -> select mode ON, rows show checkboxes
            await pilot.click("#btn-select")
            await pilot.pause(0.1)
            assert sl.select_mode is True

            # toggle off again via the same button
            await pilot.click("#btn-select")
            await pilot.pause(0.1)
            assert sl.select_mode is False and sl.selected == set()

            # Save button acts on the highlighted row (highlight 'old')
            idx = next(i for i, (rid, _) in enumerate(sl._rows()) if rid == "old")
            sl.query_one("#session-list").highlighted = idx
            await pilot.pause(0.05)
            sl.query_one("#btn-save", Button).press()
            await pilot.pause(0.1)
            assert saved == ["old"], saved

            # Delete-sel button opens the confirmation popup
            def capture(screen, *a, **k):
                pushed.append(type(screen).__name__)

            app.push_screen = capture  # type: ignore[method-assign]
            sl.query_one("#btn-del-sel", Button).press()
            await pilot.pause(0.1)

    asyncio.run(scenario())
    assert pushed == ["ConfirmDeleteDialog"], pushed
