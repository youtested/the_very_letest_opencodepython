"""opencode_py Textual TUI app.

Mirrors opencode's session screen: header status bar (agent/model/provider/
permission), scrollable chat with live tool blocks + diff rendering, and a
prompt input bar. The engine runs in a worker thread; events are bridged to
the UI via call_from_thread.
"""

from __future__ import annotations

import asyncio
import copy
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical

_MAX_LINES_CAP = 400  # screen_view text capture: rows before truncation
_MAX_WIDGET_LINES = 250  # screen_view widgets capture: tree rows before truncation
_DIALOG_TIMEOUT = 30.0  # seconds to wait for permission/question dialog before defaulting

from ..config import load_config, save_config
from .chat_view import ChatView, MessageBubble
from .input_bar import (
    AgentToggleRequested,
    CommandSelected,
    InputBar,
    ModelsRequested,
    PromptSubmitted,
    RotationLockToggled,
    SessionNavRequested,
    format_duration,
)
from .status_bar import StatusBar
from .subagent_footer import NavRequested, SubagentFooter

if TYPE_CHECKING:
    from ..agent.loop import AgentLoop
    from ..config import Config
    from ..question import QuestionInfo


def _prewarm_heavy_deps() -> None:
    """Import the engine chain + provider internals on a background thread so
    they don't sit on the first-turn path (best effort — lazy imports re-run
    normally if any of this fails)."""
    try:
        import opencode_py.agent.loop  # noqa: F401
        import opencode_py.agent.compaction  # noqa: F401
        import opencode_py.commands  # noqa: F401
        import opencode_py.tools  # noqa: F401
        import opencode_py.providers.zen  # noqa: F401
        import opencode_py.providers.openai_compat  # noqa: F401
        import opencode_py.providers.anthropic  # noqa: F401
    except Exception:  # pragma: no cover - best-effort warm-up
        pass


# Read-only slash commands allowed while a turn is running. Everything else
# (undo/clear/compact/agent/model/theme/...) mutates engine state or the running
# turn and is blocked until the current request finishes. `thinking` is safe:
# on/off/show/hide/last are pure UI, and an effort change only saves config +
# marks rotation dirty — the engine picks it up at the next provider step
# (see _stream), never mid-chunk.
_SAFE_WHILE_BUSY = {
    "help",
    "config",
    "permissions",
    "sessions",
    "ls",
    "models",
    "review",
    "connect",
    "resume",
    "continue",  # /resume's alias — the pair must behave identically
    "thinking",
    "mcp",  # list/test are read-only; add/remove save config + reload next turn
}

# Honest usage lines for arg-taking commands (the popup used to fabricate
# "Usage: /x [args]" even for no-arg commands like /help).
_COMMAND_USAGE = {
    "resume": "Usage: /resume <session-id>",
    "export": "Usage: /export [session-id]",
    "model": "Usage: /model <model-id>",
    "agent": "Usage: /agent build|plan|explore",
    "theme": "Usage: /theme <name>",
    "config": "Usage: /config print|validate",
    "connect": "Usage: /connect [provider]",
    "mcp": "Usage: /mcp [list|add <name> -- <command...>|remove <name>|test [name]]",
    "thinking": "Usage: /thinking [show|hide|last]",
}


def _probe_online() -> bool:
    """True when the provider route is reachable (reconnect watcher).

    Tiny raced probes (~5s max total): ANY HTTP status (even 500) means the
    route is alive — only transport exceptions mean still offline. Several
    independent hosts race so one dead provider can't fake an outage. Never
    raises.
    """
    urls = [
        "https://opencode.ai/zen/v1/models",
        "https://api.groq.com/openai/v1/models",
        "https://openrouter.ai/api/v1/models",
    ]
    try:
        import httpx
    except ImportError:
        return False

    def _hit(url: str) -> bool:
        try:
            httpx.get(url, timeout=5.0, follow_redirects=True)
            return True
        except Exception:
            return False

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(_hit, u) for u in urls]
            try:
                for fut in as_completed(futs, timeout=6.0):
                    try:
                        if fut.result():
                            return True
                    except Exception:
                        continue
            except Exception:
                pass
    except Exception:
        pass
    return False


class OpenCodeTUI(App):
    TITLE = "opencode"
    SUB_TITLE = "opencode_py"

    ENABLE_COMMAND_PALETTE = False  # ctrl+p is bound to Settings instead

    CSS = """
    Screen {
        background: $background;
        color: $text;
    }
    #root {
        layout: vertical;
        height: 1fr;
    }
    #chat-stack {
        layout: vertical;
        height: 1fr;
    }
    ChatView {
        width: 100%;
        height: 1fr;
        padding: 0 2;
        background: $background;
        scrollbar-size-vertical: 0;
        scrollbar-size-horizontal: 0;
    }
    .chat-welcome-logo {
        width: 100%;
        height: 100%;
        content-align: center middle;
    }
    SubagentFooter {
        width: 100%;
        height: auto;
        background: $panel;
        padding: 0 2;
    }
    #subagent-info {
        width: 1fr;
        height: 1;
        padding: 0 1 0 2;
    }
    #subagent-nav {
        height: 1;
        padding: 0 1;
        background: $panel;
    }
    .subagent-gap {
        height: 1;
    }
    InputBar {
        width: 100%;
        height: auto;
        padding: 0 1;
        background: $background;
    }
    .prompt-frame {
        height: auto;
        background: $background;
    }
    #prompt-accent {
        width: 1;
        height: 3;
        background: $primary;
    }
    .prompt-body {
        width: 1fr;
        height: auto;
        padding: 0 0 0 1;
    }
    #prompt-input {
        width: 1fr;
        background: $surface;
        color: $text;
        border: none;
        outline: none;
        padding: 0 1;
        height: 3;
        min-height: 3;
        content-align: left middle;
    }
    #prompt-input:focus {
        border: none;
        outline: none;
    }
    #prompt-title {
        width: 1fr;
        height: 1;
        margin-top: 1;
        margin-bottom: 1;
        padding: 0 1;
        color: $text-muted;
    }
    #prompt-meta {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    #prompt-status {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    #prompt-status.hidden {
        display: none;
    }
    #suggestions {
        height: auto;
        max-height: 10;
        overflow-y: auto;
        scrollbar-size-vertical: 1;
        padding: 0 1;
        background: $surface;
        color: $text;
        border: round $accent;
        margin: 0 1 1 2;
    }
    #suggestions.hidden {
        display: none;
    }
    StatusBar {
        width: 100%;
        height: 1;
        padding: 0 1;
        background: $background;
        color: $text-muted;
    }
    .cmd-popup {
        width: 74;
        height: auto;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    .cmd-popup.settings {
        width: 78;
        max-height: 92%;
    }
    .cmd-popup-title {
        height: 1;
        text-style: bold;
        color: $accent;
        background: $surface;
    }
    .settings-title {
        height: 1;
        text-style: bold;
        color: $accent;
        background: $surface;
    }
    .settings-count {
        height: 1;
        color: $text-muted;
        background: $surface;
        padding: 0 1;
    }
    .settings-scroll {
        height: 40;
        min-height: 4;
        max-height: 70%;
        border: none;
    }
    .settings-body {
        padding: 1 2;
        color: $text;
    }
    #settings-list {
        height: auto;
        max-height: 62vh;
        border: tall $accent-muted;
        background: $background;
        padding: 0 1;
        scrollbar-size-horizontal: 0;
    }
    .settings-edit {
        margin: 0 2;
        height: 3;
    }
    .settings-hint {
        dock: bottom;
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    .cmd-popup-actions {
        height: 3;
        align: center middle;
        background: $surface;
        padding: 0 2;
    }
    .cmd-popup.session-popup {
        max-height: 90%;
    }
    OptionList > .option--highlighted {
        background: $block-cursor-background;
        color: $block-cursor-foreground;
        text-style: bold;
    }
    ListView > .list-item--highlighted {
        background: $block-cursor-background;
        color: $block-cursor-foreground;
    }
    #session-list {
        height: auto;
        max-height: 60vh;
        border: none;
        padding: 0;
        scrollbar-size-horizontal: 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt"),
        Binding("ctrl+r", "resume", "Resume"),
        Binding("ctrl+t", "toggle_agent", "Switch agent"),
        Binding("ctrl+m", "models", "Models"),
        Binding("escape", "interrupt_escape", "Interrupt (press twice)"),
        Binding("ctrl+p", "settings", "Settings"),
        Binding("ctrl+s", "settings", "Settings"),
        Binding("ctrl+shift+e", "toggle_thought", "Expand/collapse thought"),
        # session routing between parallel sub-agents (opencode's
        # session.parent / session.child.next / session.child.previous /
        # session.child.first). Non-priority so the prompt keeps the arrow keys
        # for cursor movement while it is non-empty.
        Binding("up", "fd_parent", "Parent session", priority=False),
        Binding("left", "fd_prev", "Previous subagent", priority=False),
        Binding("right", "fd_next", "Next subagent", priority=False),
        Binding("ctrl+down", "fd_first", "View subagents"),
        # HOME/END/PgUp/PgDn always scroll the conversation, no matter where
        # focus is (the input box normally eats them for text editing). Prior
        # bindings win over the focused widget, so they work right out of the
        # gate — no need to click a message to hand focus to the chat first.
        Binding("home", "chat_home", "Scroll to top", priority=True),
        Binding("end", "chat_end", "Scroll to bottom", priority=True),
        Binding("pageup", "chat_page_up", "Scroll page up", priority=True),
        Binding("pagedown", "chat_page_down", "Scroll page down", priority=True),
    ]

    def __init__(
        self,
        cfg: Config | None = None,
        engine: AgentLoop | None = None,
        directory: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg or load_config()
        # Live theme selection: every widget resolves colors via active_theme()
        # at render time, so this one call styles the whole app.
        from .theme import set_active_theme

        set_active_theme(getattr(self.cfg, "theme", "") or "opencode")
        self.directory = directory or Path.cwd()
        # Engine-chain imports (agent.loop, tools, commands, session, auth) are
        # deferred out of module scope so `import opencode_py.tui` (and thus
        # first paint) doesn't pay ~0.4s before the app even exists. The engine
        # is built on demand: a background thread warms it right after on_mount
        # for the real app, and any synchronous first access (tests, sub-agent
        # spawn) builds it inline. `engine=` still injects a prebuilt engine.
        self._engine: AgentLoop | None = engine
        self._engine_lock = threading.Lock()
        # Single dialog queue: permission AND question modals from every agent
        # (parent + parallel children) serialize here, so two engine threads
        # can never stack screens or steal each other's 30s answer window.
        # Each waiter gets its full window once its dialog is actually shown.
        self._dialog_lock = threading.Lock()
        # Force-stop flag: set by the 2nd ESC while work is still running.
        # Dialog waits watch it so a stuck modal can't trap the worker after
        # the user demanded a stop. Cleared once nothing is busy anymore.
        self._force_stop = threading.Event()
        # Reconnect watchers (auto-resume): sid -> stop Event for the
        # background thread watching connectivity after a network-killed turn.
        self._reconnect_watchers: dict[str, threading.Event] = {}
        if engine is not None:
            self._wire_engine(engine)
        from ..globals import Path as GPath
        from ..auth import Auth

        self.auth = Auth(auth_file=GPath.auth_file())
        # Per-session interrupt flags: each session has its own flag so interrupting
        # one session doesn't affect others. Keyed by session_id.
        self._interrupt_flags: dict[str, bool] = {}
        from ..commands import build_registry as build_command_registry

        self.command_registry = build_command_registry()
        from ..session import new_session

        self.session = new_session(
            directory=str(self.directory),
            provider=self.cfg.provider,
            model=self.cfg.model,
            agent=self.cfg.default_agent or "build",
        )
        if engine is not None:
            engine.session_id = self.session.id
        self._chats: dict[str, ChatView] = {}
        self._sessions: dict[str, Any] = {self.session.id: self.session}
        self._engines: dict[str, AgentLoop] = {}
        if engine is not None:
            self._engines[self.session.id] = engine
        self._main_engine: AgentLoop | None = engine
        self._current_session_id = self.session.id
        self._active_turn_session_id = self.session.id
        self._busy = False
        self._busy_sessions: set[str] = set()
        self._running_agents: dict[str, str] = {}
        # Per-session turn bookkeeping. A single global set of had_text/
        # had_reasoning/... flags is WRONG: a sub-agent's events (keyed by the
        # child session id) or a prompt submitted on another session while this
        # one streams would overwrite the active turn's flags, so _turn_done
        # could finalize the wrong bubble / show the wrong runtime. Every
        # event handler writes to the SESSION's slot; _turn_done reads the
        # slot of the turn that actually finished.
        self._turn: dict[str, dict[str, Any]] = {}
        self._esc_presses = 0
        self._esc_timer: Any = None
        # Auto-refocus timer: dragging the prompt cursor back after the focus
        # lands somewhere else (e.g. a tapped reasoning bubble), so typing keeps
        # working without having to tap the input box again.
        self._refocus_timer: Any = None
        self._main_screen: Any = None
        self._pruned: set[str] = set()
        # Deferred renders: child chats created before their widget is mounted
        # (task-row click on a fresh open) render right after the switch.
        self._pending_render: dict[str, bool] = {}
        # Live sub-agent family tree (mirrors the official store): parent_id ->
        # ordered list of child records {id, title, agent, created, status}.
        # Kept even after a child finishes so `(2 of N)` counts stay correct and
        # finished children stay reviewable.
        self._children: dict[str, list[dict[str, Any]]] = {}
        self._child_parent: dict[str, str] = {}
        # Last sub-agent viewed under each parent, so returning to a parent and
        # pressing ctrl+down resumes that same agent (official opencode keeps
        # your previous sub-agent selection per parent).
        self._last_selection: dict[str, str] = {}
        self._task_start: dict[str, float] = {}
        self._usage: dict[str, dict[str, int]] = {}
        # delta batching: text/reasoning deltas are queued (non-blocking) and
        # flushed on a short timer so a fast stream isn't re-rendered per token.
        self._pending: dict[str, dict[str, list[str]]] = {}
        self._pending_bg: dict[str, dict[str, list[str]]] = {}
        self._delta_timer: Any = None
        # Periodic autosave while a turn streams: the engine keeps the in-flight
        # assistant reply live in its history, so this persists the conversation
        # up to the very last token. If Termux/Android kills the app suddenly
        # (no graceful exit), the session file is at most a few seconds stale
        # and the picker resumes where the user left off.
        self._autosave_timer: Any = None
        self._exit_requested = threading.Event()
        # Invalidates any in-flight streaming autosave when its turn ends or the
        # app saves-all (exit/teardown), so a stale worker can never overwrite a
        # newer durable copy. See _autosave_in_flight.
        self._autosave_generation = 0
        self._autosave_thread: threading.Thread | None = None

    @property
    def engine(self) -> AgentLoop:
        """The main engine, built lazily on first access.

        Building happens synchronously here (so tests / synchronous handlers see
        a fully-wired engine), and a background thread warmed it after mount for
        the real app — whichever builds first, the lock ensures one instance.
        """
        self._ensure_engine()
        assert self._main_engine is not None
        return self._main_engine

    def _wire_engine(self, engine: AgentLoop) -> None:
        engine.on_event = self._on_engine_event
        # Per-session interrupt: each engine gets its own flag so interrupting
        # one session doesn't affect others.
        sid = engine.session_id

        def _make_interrupt_checker(session_id: str):
            def check() -> bool:
                return self._interrupt_flags.get(session_id, False)
            return check

        engine.interrupt = _make_interrupt_checker(sid)
        # Permission "ask" mode: bridge the engine thread to a modal dialog.
        # Sub-agents share the same PermissionEngine instance, so one hook works
        # for all sessions.
        engine.permission.ask_callback = self._permission_ask
        # Question "ask" mode: bridge the engine thread's question.ask to a
        # modal dialog, mirroring the official TUI's question popup. Sub-agents
        # share the same QuestionService instance, so one hook works for all.
        engine.question_service.ask_callback = self._question_ask

    def _ensure_engine(self) -> None:
        if self._main_engine is not None:
            return
        with self._engine_lock:
            if self._main_engine is not None:
                return
            from ..agent.loop import AgentLoop
            from ..tools import build_registry as build_tool_registry

            engine = AgentLoop(
                cfg=self.cfg,
                registry=build_tool_registry(self.cfg),
                directory=self.directory,
                auth=self.auth,
                agent=self.cfg.default_agent or "build",
            )
            self._wire_engine(engine)
            engine.session_id = self.session.id
            self._main_engine = engine
            self._engines.setdefault(self.session.id, engine)

    def _warm_engine(self) -> None:
        """Background engine build launched after mount: keeps first paint fast
        while the (~0.4s) engine-chain import runs off the UI thread."""
        try:
            self._ensure_engine()
        except Exception as e:
            # A failed warm-up must not kill the app's message loop; the engine
            # is rebuilt synchronously on first real use anyway.
            sys.stderr.write(f"[tui] engine warm-up failed: {e}\n")
            return
        # Pre-warm the per-model context/output lookups (and their lazy provider
        # imports: zen.py / openai_compat ~180ms) so the FIRST turn isn't held
        # up by a one-time lookup on the request path.
        try:
            from ..providers import model_context_size, model_output_limit

            model_context_size(self.cfg.provider, self.cfg.model, auth=self.auth)
            model_output_limit(self.cfg.provider, self.cfg.model)
        except Exception:
            pass
        # Thread-safe schedule via post_message; returns False (no-op) when the
        # app's message pump isn't running yet/anymore instead of dropping a
        # never-awaited coroutine.
        self.call_later(self._update_header)

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            with Vertical(id="chat-stack"):
                yield ChatView()
            yield SubagentFooter()
            yield InputBar(
                # Aliases get their own dropdown entry so typing /q, /clear or
                # /continue takes the SAME popup/confirm path as the canonical
                # name — aliases used to be raw-submitted, skipping the Run/
                # Cancel safety net (/q quit instantly with no confirmation).
                commands=self._dropdown_commands()
            )
            yield StatusBar()

    def _dropdown_commands(self) -> list[dict[str, str]]:
        cmds: list[dict[str, str]] = []
        names: set[str] = set()
        for c in self.command_registry.list():
            if c.hidden:
                continue
            cmds.append({"name": c.name, "description": c.description})
            names.add(c.name)
        for c in self.command_registry.list():
            if c.hidden:
                continue
            for alias in c.aliases:
                if alias in names:
                    continue  # canonical names always win
                names.add(alias)
                cmds.append({"name": alias, "description": c.description})
        return cmds

    def on_mount(self) -> None:
        self._thread_id = threading.get_ident()
        self._main_screen = self.screen
        # Register the Textual design-token theme from the active palette and
        # keep it live: every later set_active_theme() (picker, /theme,
        # Settings) re-applies it so CSS chrome restyles instantly.
        from .theme import set_theme_applier

        self._apply_textual_theme()
        set_theme_applier(self._apply_textual_theme)
        # Give the model eyes: screen_view tool captures THIS rendered screen.
        from ..tools.screen_view import set_capture_fn

        set_capture_fn(self._capture_for_model)
        self._update_header()
        status = self.query_one(StatusBar)
        status.set_directory(str(self.directory))
        self._main_chat = self.query_one(ChatView)
        self._chats[self.session.id] = self._main_chat
        self._footer = self.query_one(SubagentFooter)
        # First open: show the opencode logo banner until the first message
        # starts a real conversation (opencode shows its logo on the launch
        # screen, then it disappears once you begin typing/chatting).
        self._main_chat.show_logo()
        self.query_one(InputBar).focus()
        if self._main_engine is None:
            # First paint first: the engine chain imports ~0.4s of heavy modules
            # (agent.loop, tools, commands, …) that don't touch the widgets on
            # screen. Build that on a background thread so the frame is up while
            # it warms; _update_header shows cfg defaults until it's ready.
            threading.Thread(target=self._warm_engine, daemon=True).start()

    def _apply_textual_theme(self) -> None:
        """Rebuild + reapply the Textual design-token theme from the active
        palette. Registering a fresh name each time guarantees the reactive
        `theme` setter fires and every $variable-driven CSS rule re-resolves."""
        from .theme import build_textual_theme

        t = build_textual_theme()
        self.register_theme(t)
        self.theme = t.name

    # -- screen capture (screen_view tool) ---------------------------------

    def _capture_widget_tree(self) -> dict:
        """The bones under the screen: one line per widget, depth-first.

        This is the ui_probe view: type, #id, CSS classes, exact position and
        size in cells, plus FOCUSED / hidden / zero-size markers — so a layout
        bug can be traced to the specific broken widget instead of guessed
        from pixels. Read-only and bounded; a misbehaving widget's properties
        can never crash the capture.
        """
        focused = self.focused
        lines: list[str] = []
        count = 0

        def _safe(fn, default):
            try:
                return fn()
            except Exception:  # pragma: no cover - defensive per-widget guard
                return default

        try:
            nodes = [self.screen] + list(self.screen.walk_children())
        except Exception as e:  # pragma: no cover - defensive
            return {"output": f"Widget tree walk failed: {e}", "error": True}

        for depth, w in enumerate(nodes):
            if count >= _MAX_WIDGET_LINES:
                lines.append(f"… ({len(nodes) - count} more widgets not shown)")
                break
            indent = "  " * min(depth, 12)
            name = type(w).__name__
            ident = f" #{w.id}" if getattr(w, "id", None) else ""
            classes = list(_safe(lambda: w.classes, []) or [])
            css = (" ." + ".".join(classes)) if classes else ""
            region = _safe(lambda: w.region, None)
            geo = ""
            if region is not None:
                geo = f"  ({region.x},{region.y} {region.width}×{region.height})"
            markers = ""
            if focused is not None and w is focused:
                markers += " ▸FOCUSED"
            display = _safe(lambda: str(w.styles.display), "block")
            if display == "none":
                markers += " ✗hidden"
            elif region is not None and (region.width == 0 or region.height == 0):
                markers += " ␀zero-size"
            lines.append(f"{indent}{name}{ident}{css}{geo}{markers}")
            count += 1
        return {
            "output": (
                "\n".join(lines)
                + f"\n[{count} widgets, screen {self.size.width}x{self.size.height}]"
            ),
            "metadata": {
                "count": count,
                "focused": type(focused).__name__ if focused else None,
            },
        }

    def _capture_for_model(self, action: str) -> dict:
        """Bridge a worker-thread tool call to the app thread.

        Tool calls run on engine worker threads; the compositor must be read
        on the app thread. When the caller already IS the app thread (tests,
        on_mount-time queries), render directly instead of deadlocking.
        """
        if getattr(self, "_thread_id", None) == threading.get_ident():
            return self._capture_on_app_thread(action)
        return self.call_from_thread(self._capture_on_app_thread, action)

    def _capture_on_app_thread(self, action: str) -> dict:
        if action == "widgets":
            return self._capture_widget_tree()
        if action == "info":
            focused = self.focused
            return {
                "output": (
                    f"Terminal: {self.size.width}x{self.size.height} cells\n"
                    f"App title: {self.title}\n"
                    f"Screen: {type(self.screen).__name__}\n"
                    f"Focused: {type(focused).__name__}"
                    + (f" (id={focused.id})" if focused is not None and focused.id else "")
                ),
                "metadata": {
                    "width": self.size.width,
                    "height": self.size.height,
                    "focused": type(focused).__name__ if focused else None,
                },
            }
        # action == "text": render the full visible screen as plain text rows
        try:
            strips = self.screen._compositor.render_strips()
        except Exception as e:  # pragma: no cover - defensive
            return {"output": f"Screen render failed: {e}", "error": True}
        lines = [strip.text for strip in strips]
        # trim trailing blank rows / right padding so the model sees layout,
        # not hundreds of spaces
        while lines and not lines[-1].strip():
            lines.pop()
        lines = [ln.rstrip() for ln in lines]
        truncated = False
        if len(lines) > _MAX_LINES_CAP:
            lines = lines[:_MAX_LINES_CAP]
            truncated = True
        body = "\n".join(lines) if lines else "(empty screen)"
        footer = (
            f"\n[{self.size.width}x{self.size.height} cells"
            + (", truncated" if truncated else "")
            + "]"
        )
        return {
            "output": body + footer,
            "metadata": {"width": self.size.width, "height": self.size.height},
        }

    # -- session routing --------------------------------------------------
    def _chat_for(self, session_id: str) -> ChatView:
        """Chat view for a session, creating (hidden) one on first use so a
        spawned sub-agent has a live, switchable conversation."""
        chat = self._chats.get(session_id)
        if chat is not None:
            return chat
        chat = ChatView()
        self._chats[session_id] = chat
        try:
            self.query_one("#chat-stack", Vertical).mount(chat, after=self._main_chat)
        except Exception:
            pass
        chat.display = "none"
        return chat

    def _active_engine(self) -> AgentLoop:
        return self._engines.get(self._current_session_id, self.engine)

    def _active_session(self) -> Any:
        return self._sessions.get(self._current_session_id, self.session)

    def _collect_picker_rows(self) -> list[dict[str, Any]]:
        """Build the picker's rows fresh: live sessions first (running
        sub-agents marked), then persisted sessions from disk. Called both
        when the popup opens and by its refresh timer, so the list is never a
        stale snapshot.

        Deliberately NOT scoped to this project's directory: sessions are
        shared across projects on purpose here, and an exact-path filter hid
        every session saved before the workspace folder was renamed
        (opencode_in_python -> opencode_python). `list_sessions(directory=...)`
        stays available for anyone who wants upstream-style scoping."""
        from ..session import list_sessions, suggested_title

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _collect(sid: str, title: str, agent: str, created: float | None, status: str) -> None:
            if not sid or sid in seen:
                return
            seen.add(sid)
            rows.append(
                {
                    "id": sid,
                    "title": title,
                    "agent": agent,
                    "created": created,
                    "status": status,
                }
            )

        for sid, sess in self._sessions.items():
            if getattr(sess, "parent_id", None):
                continue  # launched agents live INSIDE their parent, not here
            status = "running" if sid in self._busy_sessions else ""
            _collect(sid, sess.title, sess.agent, getattr(sess, "created", None), status)
        self._rebuild_family_from_disk()
        seen_count = 0
        for sess in list_sessions():
            if not getattr(sess, "has_messages", True):
                continue  # opened but never chatted — not a session to resume
            if getattr(sess, "parent_id", None):
                continue  # launched agents live INSIDE their parent, not here
            _collect(sess.id, sess.title or suggested_title(sess), sess.agent, sess.created, "")
            seen_count += 1
            if seen_count >= 200:
                break
        return rows

    def _children_of(self, parent_id: str) -> list[dict[str, Any]]:
        """Launched agents of one parent session, oldest first.

        Merges the live tree with saved children from disk so the in-parent
        list is complete even after a restart. Each entry: id/title/agent/
        created/status. Never raises.
        """
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        try:
            for c in self._sibling_records(parent_id):
                cid = c.get("id")
                if cid and cid not in seen:
                    seen.add(cid)
                    out.append({
                        "id": cid,
                        "title": c.get("title") or "sub-agent",
                        "agent": c.get("agent") or "build",
                        "created": c.get("created") or 0.0,
                        "status": c.get("status") or ("running" if cid in self._busy_sessions else "completed"),
                    })
        except Exception:
            pass
        try:
            from ..session import list_sessions as _list_all
            for sess in _list_all():
                try:
                    if getattr(sess, "parent_id", None) != parent_id:
                        continue
                    sid = getattr(sess, "id", "")
                    if not sid or sid in seen:
                        continue
                    seen.add(sid)
                    out.append({
                        "id": sid,
                        "title": getattr(sess, "title", "") or "sub-agent",
                        "agent": getattr(sess, "agent", "") or "build",
                        "created": getattr(sess, "created", 0.0) or 0.0,
                        "status": "running" if sid in self._busy_sessions else "completed",
                    })
                except Exception:
                    continue
        except Exception:
            pass
        out.sort(key=lambda c: c.get("created") or 0)
        return out

    def _rebuild_family_from_disk(self) -> None:
        """Rebuild the parent/child navigation maps from saved sessions.

        The live maps only know children spawned THIS run; after a restart
        (or a resume) the arrows died because _parent_of found nothing. Every
        saved session with a parent_id re-registers here, so ←/→/↑ keep
        working across restarts and resumed parents find their children.
        Never raises; never duplicates existing live entries.
        """
        try:
            from ..session import list_sessions as _list_all
        except Exception:
            return
        try:
            saved = _list_all()
        except Exception:
            return
        for sess in saved:
            try:
                sid = getattr(sess, "id", "")
                pid = getattr(sess, "parent_id", None)
                if not sid or not pid or sid == pid:
                    continue
                if sid not in self._child_parent:
                    self._child_parent[sid] = pid
                records = self._children.setdefault(pid, [])
                if not any(c.get("id") == sid for c in records):
                    records.append({
                        "id": sid,
                        "title": getattr(sess, "title", "") or "sub-agent",
                        "agent": getattr(sess, "agent", "") or "build",
                        "created": getattr(sess, "created", 0.0) or 0.0,
                        "status": "completed",
                    })
            except Exception:
                continue

    def action_sessions(self) -> None:
        """Ctrl+R / `/sessions`: open the opencode-style picker (Today /
        Yesterday / older sections)."""
        from .session_list import SessionList

        sl = SessionList(
            self._collect_picker_rows(),
            current=self._current_session_id,
            on_rename=self._rename_session,
            on_delete=self._delete_session,
            on_save=self._save_current_session,
            on_refresh=self._collect_picker_rows,
            on_agents=self._children_of,
        )
        sl.on_open_agents = self._open_agents_view
        self.push_screen(sl, self._on_session_picked)

    def _open_agents_view(self, parent_id: str) -> None:
        """Open the in-parent agents list for one session.

        Shows ONLY the models launched inside that parent — Enter resumes
        one, Esc returns to the sessions picker.
        """
        from .session_list import AgentsView

        title = ""
        try:
            sess = self._sessions.get(parent_id)
            title = str(getattr(sess, "title", "") or "")
            if not title:
                from ..session import load_session, suggested_title

                loaded = load_session(parent_id)
                if loaded is not None:
                    title = loaded.title or suggested_title(loaded)
        except Exception:
            pass
        self.push_screen(
            AgentsView(
                parent_id,
                parent_title=title,
                agents=self._children_of(parent_id),
                current=self._current_session_id,
                on_refresh=lambda pid=parent_id: self._children_of(pid),
            ),
            self._on_agent_picked,
        )

    def _on_agent_picked(self, choice: str | None) -> None:
        if choice:
            self._resume_session(choice)
        try:
            self.query_one(InputBar).input.focus()
        except Exception:
            pass

    def action_models(self) -> None:
        """Ctrl+M: launch the model picker (same as `/models`)."""
        self._open_model_picker()

    def _on_session_picked(self, choice: str | None) -> None:
        if choice:
            self._resume_session(choice)
        try:
            self.query_one(InputBar).input.focus()
        except Exception:
            pass

    def _switch_session(self, session_id: str) -> None:
        if session_id == self._current_session_id:
            return
        if session_id in self._pruned:
            # A pruned id whose file still exists on disk is a STALE marker
            # (failed delete / prune-then-resume race) — heal it and open the
            # session instead of refusing with "finished and closed". Only a
            # genuinely file-less id keeps the refusal.
            try:
                from ..session import load_session as _load_pruned
                revived = _load_pruned(session_id)
            except Exception:
                revived = None
            if revived is not None:
                try:
                    self._pruned.discard(session_id)
                except Exception:
                    pass
                self._sessions[session_id] = revived
                try:
                    self._rebuild_family_from_disk()
                except Exception:
                    pass
                chat = self._chat_for(session_id)
                try:
                    self._render_history(chat, revived.messages)
                except Exception:
                    pass
            else:
                self.notify("That sub-agent session is finished and closed.")
                try:
                    self.query_one(InputBar).focus()
                except Exception:
                    pass
                return
        old = self._chats.get(self._current_session_id)
        if old is None:
            # the current chat vanished (delete-flow edge): fall back to the
            # main chat instead of crashing the keypress mid-navigation
            old = self._main_chat
            self._chats.setdefault(self.session.id, self._main_chat)
            if session_id != self.session.id:
                self._current_session_id = self.session.id
                self._switch_session(session_id)
                return
        self._current_session_id = session_id
        new = self._chat_for(session_id)
        if not new.is_attached:
            # mount failed inside _chat_for: retry once now that we are on the
            # UI thread; an unmounted widget would leave a BLANK screen while
            # internal state believes the switch happened
            try:
                self.query_one("#chat-stack", Vertical).mount(new, after=self._main_chat)
            except Exception:
                self.notify("Could not open that session's view.", severity="error")
                return
        # Sibling-arrow switches land on chats that were created empty (never
        # rendered): fill from the session NOW, or ←/→ shows a blank screen
        # even though the transcript exists on disk/in memory.
        try:
            if not list(new.query("*")):
                sess = self._sessions.get(session_id)
                msgs = getattr(sess, "messages", None) if sess is not None else None
                if not msgs:
                    from ..session import load_session as _load_sw
                    loaded = _load_sw(session_id)
                    if loaded is not None:
                        self._sessions[session_id] = loaded
                        msgs = loaded.messages
                if msgs:
                    self._render_history(new, msgs)
        except Exception:
            pass
        old.display = "none"
        new.display = "block"
        # Flush a deferred render (task-row click created this chat before it
        # was mounted): without this the child opens EMPTY and the click
        # looks dead.
        try:
            if self._pending_render.pop(session_id, False):
                sess = self._sessions.get(session_id)
                msgs = getattr(sess, "messages", None) if sess is not None else None
                if msgs:
                    self._render_history(new, msgs)
        except Exception:
            pass
        # While a sub-agent chat has focus, ↑/←/→ route to session navigation
        # (parent / siblings) instead of scrolling the message list.
        new._session_is_child = bool(self._parent_of(session_id))
        # Remember the last sub-agent viewed under its parent so returning and
        # pressing ctrl+down resumes it (official keeps the selection).
        parent_of = self._parent_of(session_id)
        if parent_of:
            self._last_selection[parent_of] = session_id
        sess = self._sessions.get(session_id)
        if sess and sess.title:
            self.notify(f"Session: {sess.title}", markup=False)
        self._update_header()
        self._update_footer()
        # The footer's context hint (`12,345 (6%)`) must follow the session
        # now on screen — otherwise the previous session's token counter stays
        # painted on the status bar after a switch/resume.
        try:
            self.query_one(StatusBar).set_usage(self._usage.get(session_id) or {})
        except Exception:
            pass
        # Streaming/busy indicators belong to the VIEWED session too: watching
        # an idle chat while another one streams used to keep the spinner (and
        # a locked input) on screen for no reason.
        self._sync_streaming_visuals()
        self.query_one(InputBar).focus()

    def _sync_streaming_visuals(self) -> None:
        """Reflect the CURRENTLY VIEWED session's busy state in the status bar
        and input bar. Busy/streaming visuals were global: viewing idle session
        B while A streamed showed A's spinner over B, and only A's turn end
        cleared it. Safe to call from anywhere on the UI thread."""
        here = self._current_session_id in self._busy_sessions
        try:
            self.query_one(StatusBar).set_streaming(here)
        except Exception:
            pass
        try:
            self.query_one(InputBar).set_busy(here)
        except Exception:
            pass

    def on_open_task_session(self, event: Any) -> None:
        sid = getattr(event, "sid", None)
        if not sid:
            return
        # A finished sub-agent from a previous run is only persisted on disk;
        # reopening its task row must load that history instead of switching
        # to an empty chat. Children deleted from disk report clearly instead
        # of silently doing nothing (the "click does nothing" symptom).
        if sid not in self._sessions and sid not in self._pruned:
            from ..session import load_session

            sess = load_session(sid)
            if sess is not None:
                self._sessions[sid] = sess
                try:
                    self._rebuild_family_from_disk()
                except Exception:
                    pass
                chat = self._chat_for(sid)
                if chat.is_attached:
                    self._render_history(chat, sess.messages)
                else:
                    # chat not mounted yet (fresh popup-less open): defer the
                    # render until after the switch mounts it, else mount()
                    # raises and the click appears dead.
                    try:
                        self._pending_render[sid] = True
                    except Exception:
                        pass
            else:
                self.notify("Sub-agent session not found on disk.", severity="error")
                try:
                    self.query_one(InputBar).focus()
                except Exception:
                    pass
                return
        self._switch_session(sid)

    # -- sub-agent navigation (official session.child.*) ------------------
    def _sibling_records(self, parent_id: str) -> list[dict[str, Any]]:
        """Every sub-agent the parent spawned, oldest first (official numbers
        ``(2 of 4)`` by creation time across ALL of the parent's children)."""
        return sorted(self._children.get(parent_id, []), key=lambda c: c.get("created") or 0)

    def _session_nav_active(self) -> bool:
        """Arrow-key session routing must not fight the prompt's cursor/history
        keys: while the user is typing a non-empty prompt the arrows belong to
        the input (official's input scope wins over the session scope)."""
        from .input_bar import PromptTextArea

        focused = self.focused
        if isinstance(focused, PromptTextArea):
            return not focused.text.strip()
        return True

    def _parent_of(self, session_id: str) -> str | None:
        """The parent session id of a session: from the live children registry
        (authoritative while a sub-agent is/was running) or the saved parent_id."""
        parent = self._child_parent.get(session_id)
        if parent:
            return parent
        sess = self._sessions.get(session_id)
        pid = getattr(sess, "parent_id", None)
        if pid:
            return pid
        # Reverse-index fallback: the registry entry may have been dropped
        # (restart, prune) while the parent's sibling list still holds us.
        # Heal the fast path so the next lookup doesn't rescan.
        for _pid, records in self._children.items():
            if any(c.get("id") == session_id for c in records):
                self._child_parent[session_id] = _pid
                return _pid
        return None

    def _go_parent(self) -> bool:
        parent_id = self._parent_of(self._current_session_id)
        if parent_id:
            self._switch_session(parent_id)
            return True
        return False

    def _go_prev(self) -> None:
        self._move_sibling(-1)

    def _go_next(self) -> None:
        self._move_sibling(1)

    def _move_sibling(self, direction: int) -> bool:
        parent_id = self._parent_of(self._current_session_id)
        if not parent_id:
            return False
        siblings = self._sibling_records(parent_id)
        if len(siblings) <= 1:
            # Single child: heal a stale registry (restart/prune wiped the
            # record) from the in-memory session's parent_id so the arrows
            # at least stay on this child instead of dying silently.
            sess = self._sessions.get(self._current_session_id)
            pid = getattr(sess, "parent_id", None) if sess else None
            if pid and parent_id == pid:
                records = self._children.setdefault(parent_id, [])
                if not any(c.get("id") == self._current_session_id for c in records):
                    records.append({"id": self._current_session_id, "created": time.time(), "status": "running"})
            return False
        try:
            index = next(i for i, c in enumerate(siblings) if c.get("id") == self._current_session_id)
        except StopIteration:
            # Registry drift: current child missing from the sibling list
            # (record evicted but parent link intact). Re-add ourselves so
            # the NEXT press can move instead of dying forever.
            self._children.setdefault(parent_id, []).append(
                {"id": self._current_session_id, "created": time.time(), "status": "running"})
            return False
        target = (index + direction) % len(siblings)
        target_sid = siblings[target]["id"]
        # Never land on a pruned (closed/deleted) child: skip straight to
        # the next live sibling. Landing on one hit the "finished and closed"
        # wall and LOOKED like ←/→ doing nothing.
        if target_sid in self._pruned:
            for step in range(1, len(siblings)):
                cand = siblings[(index + direction * step) % len(siblings)]["id"]
                if cand not in self._pruned:
                    target_sid = cand
                    break
            else:
                return False
        self._switch_session(target_sid)
        return True

    def _go_first_child(self) -> None:
        # open the in-parent agents list so the user touches and picks the
        # launched agent inside this session — except a single child, which
        # resumes directly (keeps the old single-agent flow instant).
        kids = self._children_of(self._current_session_id)
        if not kids:
            return
        if len(kids) == 1:
            self._switch_session(kids[0]["id"])
            return
        last = self._last_selection.get(self._current_session_id)
        if last and any(c.get("id") == last for c in kids):
            self._switch_session(last)
            return
        # no prior selection: resume the first child directly (old instant
        # flow); the agents popup stays one Enter away on the picker row.
        self._switch_session(kids[0]["id"])

    def action_fd_parent(self) -> None:
        if self._session_nav_active():
            self._go_parent()

    def action_fd_prev(self) -> None:
        if self._session_nav_active():
            self._go_prev()

    def action_fd_next(self) -> None:
        if self._session_nav_active():
            self._go_next()

    def action_fd_first(self) -> None:
        self._go_first_child()

    def _chat_scroll_target(self) -> ChatView | None:
        """The chat of the session currently on screen (used by HOME/END/PgUp/
        PgDn so they scroll the conversation no matter what widget has focus)."""
        return self._chats.get(self._current_session_id)

    def action_chat_home(self) -> None:
        chat = self._chat_scroll_target()
        if chat is not None:
            chat.scroll_home(animate=False)

    def action_chat_end(self) -> None:
        chat = self._chat_scroll_target()
        if chat is not None:
            chat.scroll_end(animate=False)

    def action_chat_page_up(self) -> None:
        chat = self._chat_scroll_target()
        if chat is not None:
            chat.scroll_page_up(animate=False)

    def action_chat_page_down(self) -> None:
        chat = self._chat_scroll_target()
        if chat is not None:
            chat.scroll_page_down(animate=False)

    def on_nav_requested(self, event: NavRequested) -> None:
        if event.action == "parent":
            self._go_parent()
        elif event.action == "prev":
            self._go_prev()
        elif event.action == "next":
            self._go_next()

    def on_session_nav_requested(self, event: SessionNavRequested) -> None:
        """Arrow keys pressed with an empty prompt (posted from the input bar).
        `↑` parent (or previous prompt when there is no parent), `←`/`→` cycle
        the parallel sub-agent siblings, `↓` recalls the next prompt / draft."""
        key = event.direction
        if key == "up":
            bar = self.query_one(InputBar)
            if (
                bar.input.value == ""
                and bar._hist_index == len(bar._history)
                and bar._draft
            ):
                # the final ↓ cleared the box: the next ↑ restores what we were
                # typing instead of session-navigating away
                bar.input.value = bar._draft
                bar.input.cursor_position = len(bar._draft)
                return
            if not self._go_parent():
                # No parent → fall back to prompt history (main session AND a
                # parent-of-agents: being a parent gives ↑ no navigation
                # target, so sent chats must still recall). Only a CHILD whose
                # parent record vanished stays silent (no popup, no pasted
                # prompts).
                if self._parent_of(self._current_session_id) is None:
                    bar.recall_history("up")
        elif key == "down":
            self.query_one(InputBar).recall_history("down")
        elif key == "left":
            self._go_prev()
        elif key == "right":
            self._go_next()

    def _update_footer(self) -> None:
        """Show the subagent footer only while viewing a child session
        (opencode's SubagentFooter: `Build (2 of 4)` + usage + parent/prev/next)."""
        footer = getattr(self, "_footer", None)
        if footer is None:
            return
        current = self._current_session_id
        self._mark_selected_task()
        parent_id = self._parent_of(current)
        if not parent_id or not getattr(self.cfg, "subagent_footer", False):
            # removed by default (cfg.subagent_footer=False): the bar cost a
            # screen line on phones and duplicated what ↑ / ← / → already do.
            # Task-row highlighting above still runs — it is independent.
            footer.hide()
            return
        sess = self._sessions.get(current)
        siblings = self._sibling_records(parent_id)
        if not siblings:
            footer.hide()
            return
        index = next(
            (i for i, c in enumerate(siblings) if c.get("id") == current),
            0,
        )
        footer.show(
            label=str(getattr(sess, "agent", "") or "build").title(),
            index=index + 1,
            total=len(siblings),
            usage=self._usage.get(current),
        )

    def _mark_selected_task(self) -> None:
        """Highlight which sub-agent is currently selected on the parent's task
        rows: the child being viewed, or — while sitting at the parent — the
        child you last opened (official opencode marks the active sub-agent).

        Only the PARENT chat's task bubbles are touched (task rows live there),
        and a no-op selection change returns early: this ran per footer update
        over EVERY chat x EVERY bubble, which visibly janked long sessions on
        armv7 mid-stream."""
        current = self._current_session_id
        parent_id = self._parent_of(current)
        if parent_id:
            target_sid, parent_chat_id = current, parent_id
        else:
            target_sid = self._last_selection.get(current) or ""
            parent_chat_id = current
        marked = getattr(self, "_marked_tasks", None)
        if marked == (parent_chat_id, target_sid):
            return
        self._marked_tasks = (parent_chat_id, target_sid)
        chat = self._chats.get(parent_chat_id)
        if chat is None:
            return
        try:
            for bubble in chat.query(MessageBubble):
                if bubble.role != "tool" or not isinstance(bubble.content, dict) or bubble.content.get("tool") != "task":
                    continue
                meta = bubble.content.get("metadata") or {}
                bubble.selected = str(meta.get("sessionId") or "") == target_sid
        except Exception:
            pass

    def _resume_session(self, session_id: str) -> None:
        """Switch to a live session or load a persisted one (engine + chat
        rebuilt around its saved history) so the conversation can continue."""
        if session_id in self._sessions:
            self._switch_session(session_id)
            return
        from ..session import load_session, suggested_title
        from ..agent.loop import AgentLoop
        from ..tools import build_registry as build_tool_registry

        sess = load_session(session_id)
        if sess is not None:
            # A prune marker with a live file on disk is STALE — it means the
            # in-memory chat was torn down, not that history vanished. The old
            # check order refused to open resurrected rows ("That sub-agent
            # session is finished and closed") after a failed delete left the
            # body behind. Disk presence wins; heal the marker. A resumed
            # CHILD specifically must never stay pruned, or every later click
            # on its task row silently does nothing.
            try:
                if session_id in self._pruned:
                    self._pruned.discard(session_id)
            except Exception:
                pass
        if sess is None:
            if session_id in self._pruned:
                self.notify(
                    "That session no longer exists (it was closed or deleted)."
                )
            else:
                self.notify("Session not found.")
            return
        if not sess.title:
            sess.title = suggested_title(sess)
        engine = AgentLoop(
            cfg=self.cfg,
            registry=build_tool_registry(self.cfg),
            directory=Path(sess.directory) if sess.directory else self.directory,
            auth=self.auth,
            agent=sess.agent or "build",
            provider_id=sess.provider or self.cfg.provider,
            model_id=sess.model or self.cfg.model,
            session_id=session_id,
        )
        engine.on_event = self._on_engine_event
        engine.interrupt = self._interrupt_requested
        engine.permission.ask_callback = self._permission_ask
        engine.question_service.ask_callback = self._question_ask
        # the pin is a workspace-wide choice: a resumed session must not
        # silently come back UNLOCKED and fail over on its first error
        engine.rotation_locked = bool(getattr(self.cfg, "rotation_lock", False))
        engine.set_history(sess.messages)
        self._engines[session_id] = engine
        self._sessions[session_id] = sess
        # a resumed child must rejoin the family tree so the arrows keep
        # working (the live maps were empty after a restart).
        try:
            self._rebuild_family_from_disk()
        except Exception:
            pass
        chat = self._chat_for(session_id)
        chat.display = "none"
        self._render_history(chat, sess.messages)
        # a resumed PARENT's task rows must be clickable too: re-link every
        # known child session onto its row once the bubbles exist. Rendering
        # mounts in 15-row chunks via call_later, so the rows may not exist
        # yet — retry until linked or the chat switches away.
        try:
            self._schedule_relink(session_id, attempts=40)
        except Exception:
            pass
        self._switch_session(session_id)
        # restore the context-usage hint immediately (it used to stay blank
        # until the next turn completed): estimate tokens like the engine does
        try:
            from ..agent import compaction as compact_mod

            est = sum(
                compact_mod.estimate_tokens(compact_mod.msg_text(m))
                for m in sess.messages
            )
            if est:
                usage = {"input_tokens": est, "output_tokens": 0, "total_tokens": est}
                ctx_size = 0
                from ..providers import model_context_size

                ctx_size = model_context_size(
                    sess.provider or self.cfg.provider,
                    sess.model or self.cfg.model,
                    auth=self.auth,
                )
                if ctx_size:
                    usage["context_size"] = ctx_size
                self._usage[session_id] = usage
                self.query_one(StatusBar).set_usage(usage)
        except Exception:
            pass
        self.notify(f"Resumed: {sess.title or session_id}", markup=False)

    def _rename_session(self, session_id: str, title: str) -> str | None:
        """Persist a rename; returns an error message or None on success."""
        from ..session import load_session, save_session

        try:
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = load_session(session_id)
                if sess is None:
                    return "Session not found."
            sess.title = title
            # Renames only stick to disk when the session actually has a
            # conversation — a never-chatted scratch session must not be
            # materialised into a file just because it was titled. Say so,
            # instead of silently dropping the title on the next launch.
            if getattr(sess, "messages", None):
                save_session(sess)
            else:
                self.notify(
                    "Rename kept for this run — the title saves once the session has messages.",
                    severity="warning",
                    timeout=5,
                )
        except Exception as e:
            return f"Rename failed: {e}"
        return None

    def _save_current_session(self, session_id: str = "") -> bool:
        """Save the highlighted session right now (Ctrl+S in the picker). An
        empty id means "the session I'm in" (the old popup-wide Save). Includes
        the engine's live history so the durable copy matches what's on
        screen. Returns False when there was nothing to save / it failed."""
        from ..session import save_session

        sid = session_id or self._current_session_id
        sess = self._sessions.get(sid)
        if sess is None:
            from ..session import load_session

            sess = load_session(sid)
            if sess is None:
                self.notify("Session not found.", severity="error")
                return False
        engine = self._engines.get(sid)
        try:
            if engine is not None:
                history = engine.get_history()
                if history:
                    sess.messages = history
            if not getattr(sess, "messages", None):
                self.notify("Nothing to save — this session has no conversation.")
                return False
            save_session(sess)
        except Exception as e:
            self.notify(f"Save failed: {e}", severity="error", markup=False)
            return False
        self.notify(f"Session saved: {sess.title or '(untitled)'}", markup=False)
        return True

    def _action_new(self) -> None:
        """`/new` / `/clear`: start a brand-new session in place.

        The old conversation is durably saved first (it stays resumable from
        the picker), then the workspace resets exactly like the delete-main-
        session flow: same engine instance re-registered under a new session
        id with empty history, chat cleared, header/footer/usage refreshed."""
        from ..session import new_session

        try:
            self._save_all_live_sessions()
        except Exception:
            pass
        old_id = self.session.id
        old_chat = self._chats.get(old_id)
        self.session = new_session(
            directory=str(self.directory),
            provider=self.cfg.provider,
            model=self.cfg.model,
            agent=self.engine.agent,
        )
        self.engine.session_id = self.session.id
        try:
            self.engine.set_history([])
        except Exception:
            pass
        self.engine.clear_prompts()
        self._engines.pop(old_id, None)
        self._sessions.pop(old_id, None)
        self._turn.pop(old_id, None)
        if old_chat is not None and old_chat is not self._main_chat:
            try:
                old_chat.remove()
            except Exception:
                pass
            self._chats.pop(old_id, None)
        else:
            self._main_chat.clear()
        self._chats[self.session.id] = self._main_chat
        self._sessions[self.session.id] = self.session
        self._engines[self.session.id] = self.engine
        self._current_session_id = self.session.id
        self._active_turn_session_id = self.session.id
        self._usage.pop(old_id, None)
        try:
            self.query_one(StatusBar).set_usage({})
        except Exception:
            pass
        self._main_chat.show_logo()
        self._update_header()
        self._update_footer()
        self.query_one(InputBar).focus()

    def _delete_session(self, session_id: str) -> bool:
        """Delete a session from disk (and its live registrations). Any session
        is deletable — deleting the one you're in resets the workspace to a
        brand-new session so a later save doesn't resurrect the deleted file.
        A session mid-turn is still protected (it owns a running engine)."""
        from ..session import delete_session, new_session

        if session_id in self._busy_sessions:
            return False  # never drop a running turn out from under it
        if session_id == self.session.id:
            # deleting the workspace itself -> start a fresh session in place
            old_id = self.session.id
            viewing_old = self._current_session_id == old_id
            self.session = new_session(
                directory=str(self.directory),
                provider=self.cfg.provider,
                model=self.cfg.model,
                agent=self.engine.agent,
            )
            self.engine.session_id = self.session.id
            try:
                self.engine.set_history([])
            except Exception:
                pass
            self._chats.pop(old_id, None)
            self._sessions.pop(old_id, None)
            self._engines.pop(old_id, None)
            self._turn.pop(old_id, None)
            self.engine.clear_prompts()
            self._chats[self.session.id] = self._main_chat
            self._sessions[self.session.id] = self.session
            self._engines[self.session.id] = self.engine
            self._pruned.discard(old_id)
            if self._active_turn_session_id == old_id:
                self._active_turn_session_id = self.session.id
            # Same cleanup /new does: drop the old token counter, reset the
            # status bar, clear + logo the main chat.
            self._usage.pop(old_id, None)
            try:
                self.query_one(StatusBar).set_usage({})
            except Exception:
                pass
            self._main_chat.clear()
            if viewing_old or self._current_session_id not in self._sessions:
                # Land in the fresh workspace. (The old build repointed
                # _current_session_id unconditionally WITHOUT touching what was
                # on screen: watching another chat while deleting the main one
                # left prompts streaming into a display:none widget, and the
                # batch flow could blank the pane entirely.)
                self._current_session_id = self.session.id
                self._main_chat.display = "block"
                self._main_chat._session_is_child = False
                self._main_chat.show_logo()
            # else: the user is watching another LIVE chat — leave it exactly
            # where it is instead of yanking them into the empty workspace.
            self._update_header()
            self._update_footer()
            self._sync_streaming_visuals()
            # Do NOT persist the fresh replacement session here: saving an empty
            # session on every delete is how 0-message ghost files accumulate.
            # It only gets its own file once the user actually chats and one of
            # the save conditions fires (crash-safety autosave / exit / close).
            self.notify("Session deleted — starting a new one.")
            delete_session(old_id)
            return True
        # a non-main session (resumed, sub-agent, …)
        was_live = session_id in self._sessions
        was_current = session_id == self._current_session_id
        if was_current:
            self._switch_session(self.session.id)  # hop back to a live session
        # drop the deleted session from the sub-agent family tree
        self._children.pop(session_id, None)  # its own children first
        for pid, records in list(self._children.items()):
            self._children[pid] = [r for r in records if r.get("id") != session_id]
        self._child_parent.pop(session_id, None)
        self._task_start.pop(session_id, None)
        self._usage.pop(session_id, None)
        self._sessions.pop(session_id, None)
        self._engines.pop(session_id, None)
        chat = self._chats.pop(session_id, None)
        self._turn.pop(session_id, None)
        self._pruned.add(session_id)
        self._running_agents.pop(session_id, None)
        self._refresh_running_agents()
        if chat is not None:
            try:
                chat.remove()
            except Exception:
                pass
        if was_current:
            self.notify("Session deleted.")
        # ALWAYS drop the durable copy too. The old `True if was_live else
        # delete_session(...)` skipped the disk delete for any session that was
        # live in RAM — but a RESUMED session was loaded from a file, so the
        # file survived: the picker's 2s refresh resurrected the row, the next
        # Ctrl+D was needed (and only then actually deleted it), and until then
        # clicking the row hit the _pruned wall ("finished and closed").
        deleted_on_disk = delete_session(session_id)
        return True if (was_live or deleted_on_disk) else False

    def _render_history(self, chat: ChatView, messages: list[dict[str, Any]]) -> None:
        """Replay saved messages into a ChatView (tool calls paired with their
        results so each run reads as one row). Tool `arguments` arrive as JSON
        strings in history but the renderers expect dicts, so parse them.

        Mounting one bubble costs ~6ms of Textual widget machinery, so a long
        saved session is filled in progressively (first chunk synchronous, the
        rest queued on the event loop) instead of freezing the UI for seconds.
        """
        import json as _json

        def _args(raw: Any) -> dict[str, Any]:
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str):
                try:
                    parsed = _json.loads(raw)
                except _json.JSONDecodeError:
                    return {}
                return parsed if isinstance(parsed, dict) else {}
            return {}

        def _reasoning_text(raw: Any) -> str:
            if isinstance(raw, str):
                return raw
            if isinstance(raw, list):
                parts = []
                for p in raw:
                    if isinstance(p, dict):
                        parts.append(str(p.get("text") or p.get("content") or ""))
                    elif isinstance(p, str):
                        parts.append(p)
                return "\n".join(p for p in parts if p)
            return str(raw or "")

        tasks: list[Callable[[], None]] = []  # one deferred bubble-mount per message
        pending: list[tuple[str, str, dict[str, Any]]] = []  # (call_id, toolname, input)
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                continue
            if role == "compaction" or msg.get("compaction"):
                text = str(content or "")
                tasks.append(lambda t=text: chat.append_compaction(t))
                continue
            if role == "user":
                text = str(content or "")
                agent = str(msg.get("agent") or "build")
                tasks.append(lambda t=text, a=agent: chat.append_user(t, agent=a))
                continue
            if role == "assistant":
                reasoning = msg.get("reasoning_content")
                if reasoning:
                    rtext = _reasoning_text(reasoning)
                    tasks.append(lambda t=rtext: chat.append_reasoning(t))
                if content:
                    ctext = str(content)
                    tasks.append(lambda t=ctext: chat.append_assistant(t))
                for call in msg.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    pending.append(
                        (call.get("id", ""), fn.get("name", "tool"), _args(fn.get("arguments")))
                    )
                continue
            if role == "tool":
                call_id = msg.get("tool_call_id") or ""
                tool = msg.get("name") or "tool"
                tool_input: dict[str, Any] = {}
                for j, (cid, name, args) in enumerate(pending):
                    if cid == call_id:
                        tool, tool_input = name, args
                        pending.pop(j)
                        break
                else:
                    if pending:
                        # call_id missing (out-of-order results / poisoned save):
                        # match the OLDEST pending call for the SAME tool name so
                        # parallel tool results don't get swapped between
                        # different tools; only fall back to the oldest call when
                        # no same-name call is still waiting. Use the pending
                        # call's own id/name/args so the row stays accurate.
                        idx = 0
                        for j, (cid2, name2, args2) in enumerate(pending):
                            if name2 == tool:
                                idx = j
                                break
                        call_id, tool, tool_input = pending.pop(idx)
                metadata: dict[str, Any] = {}
                # Re-attach a persisted child-session link (see run_turn):
                # resumed parents render task rows from saved history, and
                # the click handler reads metadata.sessionId.
                try:
                    psid = msg.get("session_id") or msg.get("sessionId")
                    if tool == "task" and psid:
                        metadata["sessionId"] = psid
                except Exception:
                    pass
                if tool == "todowrite":
                    # history stores the todos as JSON text; the renderer reads
                    # them from metadata.todos
                    try:
                        parsed = _json.loads(str(content or ""))
                        if isinstance(parsed, list):
                            metadata["todos"] = parsed
                    except _json.JSONDecodeError:
                        pass
                payload: dict[str, Any] = {
                    "id": call_id,
                    "tool": tool,
                    "input": dict(tool_input),
                    # "completed" (not "done"): tool renderers + the gray
                    # block frame key off this exact value
                    "status": "completed",
                    "done": True,
                    "output": content or "",
                    # copy: every lambda below closes over `payload` by NAME,
                    # so without this all rows share the LAST dict and every
                    # task row opens the same (last) child.
                    "metadata": dict(metadata),
                }
                tasks.append(lambda p=dict(payload, metadata=dict(metadata), input=dict(tool_input)): chat.append_tool(p))
                continue
            if content:
                text = str(content)
                tasks.append(lambda t=text: chat.append_meta(t))

        # A session cut off mid-tool-run (interrupt / sudden kill / old
        # poisoned save) ends with an assistant tool_calls message whose results
        # never arrived. Render those as interrupted rows so resuming shows
        # exactly where the turn stopped instead of silently dropping them.
        for call_id, name, tool_input in pending:
            payload = {
                "id": call_id,
                "tool": name,
                "input": dict(tool_input) if isinstance(tool_input, dict) else tool_input,
                "status": "interrupted",
                "done": False,
                "output": "",
                "metadata": {},
            }
            tasks.append(lambda p=payload: chat.append_tool(dict(p)))

        def _mount(i: int) -> None:
            step = 15
            for t in tasks[i : i + step]:
                t()
            nxt = i + step
            if nxt < len(tasks):
                self.call_later(lambda n=nxt: _mount(n))

        _mount(0)

    # -- running agents ---------------------------------------------------
    def _refresh_running_agents(self) -> None:
        """Show the launched sub-agents in the status line above the prompt,
        like opencode's `Delegating...` indicator (transient, no sidebar)."""
        try:
            bar = self.query_one(InputBar)
        except Exception:
            return
        bar.set_running_agents(list(self._running_agents.values()))

    # -- engine event bridge ---------------------------------------------
    def _on_engine_event(self, event: dict[str, Any]) -> None:
        # Called from the engine thread; hop to the UI thread.
        if getattr(self, "_thread_id", None) == threading.get_ident():
            # Already on the UI thread (e.g. a /command handler that makes the
            # engine emit an event, like /undo): call_from_thread would raise
            # RuntimeError, so handle inline instead.
            self._handle_event(event)
            return
        # Never block engine (or sub-agent) worker threads on the UI render:
        # every event rides the async bridge (FIFO on the app loop, so ordering
        # with the text deltas is preserved) and the worker keeps streaming.
        # Before this, each tool/subagent event did a blocking call_from_thread
        # rendezvous — N parallel children meant an N-way UI-thread storm that
        # stalled all streams on every tool row.
        self._schedule_async(event)
        return

    def _schedule_async(self, event: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            # App not running / already tearing down (Textual resets its
            # _loop to None at teardown) — dropping deltas is correct here.
            return
        try:
            asyncio.run_coroutine_threadsafe(self._async_handle(event), loop)
        except RuntimeError:
            pass  # loop closed mid-shutdown — dropping is correct
        except Exception as e:
            # Anything else must stay visible: a silent drop looks exactly
            # like the model freezing mid-stream.
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write(f"[tui] delta schedule failed: {e!r}\n")

    async def _async_handle(self, event: dict[str, Any]) -> None:
        try:
            with self._context():
                self._handle_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A delta-render failure must never vanish into a dropped future
            # (``run_coroutine_threadsafe`` swallows coroutine exceptions and
            # would leave the screen silently frozen mid-stream). Log it so the
            # bug is actually visible in the console.
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write(f"[tui] async event handler error: {e!r}\n")

    def _queue_delta(self, session_id: str, text: str, kind: str) -> None:
        if not text:
            # drop empty deltas entirely — an empty text chunk must not create
            # an assistant bubble that never gets filled
            return
        ui_thread = getattr(self, "_thread_id", None)
        if ui_thread is not None and threading.get_ident() != ui_thread:
            # Single-writer invariant: _pending and _delta_timer are only
            # touched by the UI thread (deltas arrive via _schedule_async on
            # the app loop; flushes run on the flush timer). If a future caller
            # ever queues off-thread, marshal to the UI thread instead of
            # racing the timer/swap instead of corrupting the buffer.
            try:
                self.call_from_thread(self._queue_delta, session_id, text, kind)
            except RuntimeError:
                pass
            return
        # Route to active buffer if this is the actively streaming session,
        # otherwise buffer in background to prevent cross-session bleeding.
        target = self._pending if session_id == self._active_turn_session_id else self._pending_bg
        buf = target.setdefault(session_id, {"text": [], "reasoning": []})
        buf[kind].append(text)
        if self._delta_timer is None:
            self._delta_timer = self.set_timer(0.03, self._flush_deltas)

    def _cancel_delta_timer(self) -> None:
        if self._delta_timer is not None:
            try:
                self._delta_timer.stop()
            except Exception:
                pass
            self._delta_timer = None

    def _flush_deltas(self) -> None:
        """Render any buffered text/reasoning deltas (one render per batch)."""
        self._cancel_delta_timer()
        # Flush active session buffer
        pending = self._pending
        self._pending = {}
        for session_id, buf in pending.items():
            chat = self._chat_for(session_id)
            for t in buf.get("reasoning") or []:
                if t:
                    chat.stream_reasoning_delta(t)
            if buf.get("text"):
                chat.end_reasoning()
                for t in buf["text"]:
                    if t:
                        chat.stream_delta(t)
        # Flush background session buffers (non-active sessions)
        if self._pending_bg:
            bg_pending = self._pending_bg
            self._pending_bg = {}
            for session_id, buf in bg_pending.items():
                chat = self._chat_for(session_id)
                for t in buf.get("reasoning") or []:
                    if t:
                        chat.stream_reasoning_delta(t)
                if buf.get("text"):
                    chat.end_reasoning()
                    for t in buf["text"]:
                        if t:
                            chat.stream_delta(t)

    def _handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        session_id = event.get("session_id") or self.session.id
        chat = self._chat_for(session_id)
        status = self.query_one(StatusBar)
        if kind == "step":
            pass
        elif kind == "prompt_promoted":
            # the engine folded the oldest queued prompt into the running turn
            # (opencode's Session Drain) — drop its ` QUEUED ` badge, finalize
            # the previous reply's bubble, and let the next text stream in as a
            # fresh response to the promoted prompt.
            # the previous reply's text may still be sitting in the delta buffer
            # (deltas are batched on a 30ms flush timer) — empty it BEFORE
            # remove_last_stream_bubble decides whether the stream bubble is
            # "empty", otherwise the trailing bubble can be dropped while its
            # text is still un-rendered.
            self._flush_deltas()
            chat.promote_next_queued()
            chat.end_reasoning()
            chat.remove_last_stream_bubble()
        elif kind == "retry":
            status.set_retry_message(event.get("message", "↻ retrying…"))
            self._on_retry_event(session_id, event)
        elif kind == "error":
            status.set_retry_message("")
            self._clear_task_retry(session_id)
            self._show_error(event.get("error", "unknown error"), retryable=bool(event.get("retryable")), session_id=session_id)
        elif kind == "text_delta":
            # an empty trailing chunk (a failed summary) is real output only
            # if it actually carries text — otherwise it would light up the
            # runtime line and spawn a blank assistant bubble
            text = event.get("text") or ""
            status.set_retry_message("")
            self._clear_task_retry(session_id)
            if text:
                self._turn_state(session_id)["had_text"] = True
            self._queue_delta(session_id, text, "text")
        elif kind == "reasoning_delta":
            text = event.get("text") or ""
            status.set_retry_message("")
            self._clear_task_retry(session_id)
            if text:
                self._turn_state(session_id)["had_reasoning"] = True
            self._queue_delta(session_id, text, "reasoning")
        elif kind == "tool_call":
            # the model is now responding/acting — drop any stale retry hint
            status.set_retry_message("")
            self._clear_task_retry(session_id)
            # render any buffered text first so the tool row lands below it
            self._flush_deltas()
            tool = event.get("tool", "?")
            # the current thought/assistant stream is over — finalize it so a
            # multi-step tool loop doesn't merge every step's text into one
            # bubble or leave a stale ▍ cursor
            chat.end_reasoning()
            chat.remove_last_stream_bubble()
            self._turn_state(session_id)["had_tools"] = True
            chat.append_tool(
                {
                    "tool": tool,
                    "status": "running",
                    "input": event.get("arguments", {}),
                    "call_id": event.get("call_id", ""),
                }
            )
        elif kind == "tool_start":
            tool_run = {
                "tool": event.get("tool", "?"),
                "status": "running",
                "input": event.get("input", {}),
                "call_id": event.get("call_id", ""),
            }
            if not chat.update_tool_bubble(tool_run):
                # No exact call_id match (e.g. the tool_call event never
                # arrived for this session) — append the row instead of leaving
                # a stale placeholder or updating the wrong tool.
                chat.append_tool(tool_run)
        elif kind == "tool_complete":
            run = {
                "tool": event.get("tool", "?"),
                "status": "error" if event.get("status") == "error" else "completed",
                "input": event.get("input", {}),
                "output": event.get("output", ""),
                "metadata": event.get("metadata", {}),
                "call_id": event.get("call_id", ""),
            }
            chat.update_tool_bubble(run)
            if event.get("tool") == "task":
                # enrich the task row with the child's runtime + toolcall count
                # (official's `↳ 3 toolcalls · 12.5s` completion detail).
                self._finalize_task_row(chat, run)
        elif kind == "tool_denied":
            run = {
                "tool": event.get("tool", "?"),
                "status": "error",
                "input": event.get("input", {}),
                "output": event.get("reason") or "permission denied",
                "call_id": event.get("call_id", ""),
            }
            if not chat.update_tool_bubble(run):
                # No preceding tool_call (e.g. a permission denial) — show the
                # denied row so the rejected action is visible.
                chat.append_tool(run)
        elif kind == "interrupted":
            self._flush_deltas()
            self._turn_state(session_id)["interrupted"] = True
            chat.end_reasoning()
            chat.remove_last_stream_bubble()
            chat.end_stream()
            chat.append_meta("⏹ Interrupted")
        elif kind == "usage":
            self._usage[session_id] = event.get("usage") or {}
            if session_id == self._current_session_id:
                status.set_usage(event.get("usage") or {})
        elif kind == "compaction_start":
            # official opencode: show `Compacting conversation…` with a spinner
            # while the session summarizes to recover/avoid context overflow.
            # Only when the compacting session is the one on screen: a
            # sub-agent (task tool) compacting in the background must NOT flash
            # the main bar's spinner — its summary bubble goes to the sub-agent's
            # own (hidden) chat, so the result would otherwise never appear and
            # just look like the current conversation vanished.
            if session_id == self._current_session_id:
                try:
                    self.query_one(InputBar).set_compacting(True)
                except Exception:
                    pass
            # Render any buffered text/reasoning deltas FIRST (same invariant as
            # tool_call/interrupted/prompt_promoted): begin_compaction_stream
            # finalizes the Thought + stream bubbles, so skipping the flush would
            # freeze them mid-sentence (or drop an empty one) and re-emit the
            # buffered tail as a stray bubble below the compaction divider.
            self._flush_deltas()
            # Kick off the live `▸ Compacted summary` bubble (hidden chats too —
            # they get the final divider on switch).
            chat.begin_compaction_stream()
        elif kind == "summary_delta":
            # Stream the anchored summary into the compaction bubble live so the
            # user watches it being written rather than waiting on a spinner.
            text = event.get("text") or ""
            if text:
                chat.stream_compaction_delta(text)
        elif kind == "compacted":
            # opencode renders a ` Session compacted ` divider when the session
            # summarizes to recover/avoid context overflow. Mirror that: flush
            # pending deltas, end the current reasoning bubble, then finalize
            # the live summary bubble into the divider.
            self._flush_deltas()
            if session_id == self._current_session_id:
                try:
                    self.query_one(InputBar).set_compacting(False)
                except Exception:
                    pass
            chat.end_compaction_stream(event.get("summary") or "")
            self._turn_state(session_id)["had_tools"] = True
        elif kind == "rotated":
            # Show the failover popup ONLY when rotation is unlocked. When the
            # selected model is locked the lane can't change, so a "model
            # changed" toast would be noise (or a lie) — suppress it.
            eng = self._engines.get(session_id) or self.engine
            provider = event.get("provider", "?")
            model = event.get("model", "?")
            # reflect the lane that actually answered in the mode line under
            # the input box (deepseek -> nemotron) as soon as it switches.
            eng.provider_id = provider
            eng.model_id = model
            self._update_header()
            if not getattr(eng, "rotation_locked", False):
                reason = event.get("reason") or "provider error"
                self.notify(
                    f"Now using {model} · {provider}\n{reason}",
                    title="Model changed",
                )
        elif kind == "subagent_start":
            self._on_subagent_start(event)
        elif kind == "subagent_done":
            self._on_subagent_done(event)

    def _on_subagent_start(self, event: dict[str, Any]) -> None:
        from ..session import load_session

        sid = event.get("session_id") or ""
        if not sid:
            return
        if sid not in self._sessions:
            sess = load_session(sid)
            if sess is not None:
                self._sessions[sid] = sess
            else:
                # register a placeholder so the fallbacks
                # (`_sessions.get(sid) or self.session`) never route a sub-agent
                # turn's history onto the main session file. The placeholder
                # MUST carry the parent link + title now (not blank): a blank
                # placeholder saved at exit used to mint orphan files with
                # parent_id None that showed in /sessions like real parents.
                from ..session import Session
                parent_hint = self._active_turn_session_id or self._current_session_id
                self._sessions[sid] = Session(
                    {"id": sid,
                     "title": event.get("title") or "sub-agent",
                     "agent": event.get("agent") or "build",
                     "parent_id": parent_hint,
                     "provider": self.cfg.provider,
                     "model": self.cfg.model},
                    directory=str(self.directory))
        sub = self.engine.find_subagent(sid)
        if sub is not None:
            self._engines[sid] = sub
        self._busy_sessions.add(sid)
        chat = self._chat_for(sid)
        # Show the parent's instruction at the very top of the sub-agent's
        # chat, exactly like official opencode renders the task directive as
        # the session's first message.
        prompt = event.get("prompt") or ""
        if prompt and not chat.children:
            chat.append_directive(prompt, title=event.get("title") or "")
        self._chats[sid] = chat
        title = event.get("title") or "sub-agent"
        agent = event.get("agent") or "build"
        sess = self._sessions.get(sid)
        parent_id = getattr(sess, "parent_id", None) or self._active_turn_session_id
        if parent_id:
            # register the child in the parent's family tree (numbered siblings
            # for the footer's `(2 of N)`), and attach its id to the parent
            # chat's running task row so it renders live state / is clickable.
            self._child_parent[sid] = parent_id
            records = self._children.setdefault(parent_id, [])
            if not any(c.get("id") == sid for c in records):
                records.append(
                    {
                        "id": sid,
                        "title": title,
                        "agent": agent,
                        "created": time.time(),
                        "status": "running",
                    }
                )
            self._link_task_row(sid, parent_id, call_id=event.get("call_id") or "")
        self._running_agents[sid] = f"{title} · {agent}"
        self._refresh_running_agents()
        self._update_footer()
        self.notify(f"Sub-agent started: {title}", markup=False)

    def _on_subagent_done(self, event: dict[str, Any]) -> None:
        sid = event.get("session_id") or ""
        sess = self._sessions.get(sid)
        if sess is not None:
            sess.completed = time.time()
            # Adopt the spawn-time identity (parent/title/provider/model):
            # placeholders created at subagent_start may predate the engine
            # save, and without this the file keeps blank identity fields.
            # Parent is looked up through EVERY source (live map → engine
            # session_id chain → active turn): a child saved without it
            # becomes a stray parent-looking row in /sessions, forever.
            try:
                if not getattr(sess, "parent_id", None):
                    sess.parent_id = (
                        self._child_parent.get(sid)
                        or self._active_turn_session_id
                        or self._current_session_id
                        or getattr(sess, "parent_id", None))
                if not getattr(sess, "title", ""):
                    sess.title = event.get("title") or "sub-agent"
                if not getattr(sess, "provider", ""):
                    sess.provider = self.cfg.provider
                if not getattr(sess, "model", ""):
                    sess.model = self.cfg.model
            except Exception:
                pass
            # Keep the in-memory session's messages in sync with what the
            # sub-agent engine actually produced (spawn_task saves to disk, but
            # the app-side object was last loaded with the empty placeholder).
            sub = self._engines.get(sid)
            if sub is not None:
                try:
                    sess.messages = sub.get_history()
                except Exception:
                    pass
        self._busy_sessions.discard(sid)
        chat = self._chats.get(sid)
        if chat is not None:
            chat.end_reasoning()
            # drop an empty streaming cursor if the sub-agent replied with no text
            chat.remove_last_stream_bubble()
            chat.end_stream()
        title = event.get("title") or "sub-agent"
        ok = event.get("ok", True)
        self._running_agents.pop(sid, None)
        self._refresh_running_agents()
        # Keep the finished sub-agent session registered (like the official
        # store) so its task row stays clickable, its chat stays reviewable,
        # and the parent's `(2 of N)` footer count keeps them in the total.
        record = None
        for r in self._children.get(self._child_parent.get(sid, ""), []):
            if r.get("id") == sid:
                record = r
                break
        if record is not None:
            record["status"] = "completed" if ok else "failed"
            record["completed"] = time.time()
        # Clean up in-memory dicts for completed sub-agent to prevent memory leak
        self._engines.pop(sid, None)
        self._turn.pop(sid, None)
        self._task_start.pop(sid, None)
        self._children.pop(sid, None)
        # NOTE: _child_parent[sid] is intentionally KEPT (not popped): it is
        # the only map that answers "who is my parent" for ←/→/↑ after the
        # child finishes. Dropping it killed all arrow navigation the moment
        # a sub-agent completed. Memory cost is one dict entry per child.
        self._update_footer()
        if ok:
            self.notify(f"Sub-agent done: {title}", markup=False)
        else:
            self.notify(f"Sub-agent failed: {title}", severity="error", markup=False)

    # -- task-row enrichment (official Subagent completion detail) --------
    def _schedule_relink(self, parent_id: str, attempts: int = 40) -> None:
        """Retry _relink_task_rows until every task row is linked AND backfilled.

        _render_history mounts bubbles in 15-row chunks via call_later, so a
        single relink right after render finds zero rows on long sessions
        (your 26-message parent renders its task rows seconds later). Each
        pass links what's mounted; passes continue while rows are missing
        links OR missing durations (backfill lands on a later pass than the
        link), as long as the user is still viewing this parent.
        """
        try:
            missing = self._relink_task_rows(parent_id)
        except Exception:
            missing = 1
        try:
            chat = self._chats.get(parent_id)
            unfilled = 0
            if chat is not None:
                for b in chat.query("*"):
                    try:
                        if getattr(b, "role", "") != "tool":
                            continue
                        c = getattr(b, "content", None)
                        if not isinstance(c, dict) or c.get("tool") != "task":
                            continue
                        meta = c.get("metadata") or {}
                        if not meta.get("sessionId") or not meta.get("duration"):
                            unfilled += 1
                    except Exception:
                        continue
            pending = max(missing, unfilled)
        except Exception:
            pending = missing
        if pending > 0 and attempts > 0:
            try:
                # NOTE: no current-session gate here — the resume path calls
                # this BEFORE _switch_session, so _current_session_id still
                # points at the previous session on the first passes. Gating
                # on it killed the whole retry chain (rows mounted later never
                # got linked/backfilled). Retries are cheap and idempotent.
                self.call_later(lambda: self._schedule_relink(parent_id, attempts - 1))
            except Exception:
                pass

    def _relink_task_rows(self, parent_id: str) -> int:
        """Re-attach child session ids onto a resumed parent's task rows.

        Covers ALL subagents — old saves (transcript never stored the link)
        and new ones alike. Matching order: saved transcript link first,
        then live family records by title, then disk children oldest-first
        against still-unlinked task rows. Returns the count of still-unlinked
        task rows (so the scheduler knows whether to retry). Never raises.
        """
        chat = self._chats.get(parent_id)
        if chat is None:
            return 0
        try:
            from ..session import load_session as _load
            parent = self._sessions.get(parent_id) or _load(parent_id)
        except Exception:
            parent = None
        if parent is None:
            return 0
        try:
            messages = getattr(parent, "messages", None) or []
        except Exception:
            messages = []
        # assistant-level spawn stamps (interrupt-proof link source)
        stamped: dict[str, dict] = {}
        try:
            for m in messages:
                if isinstance(m, dict) and isinstance(m.get("task_children"), dict):
                    for k, v in m["task_children"].items():
                        stamped.setdefault(str(k), (v or {}))
        except Exception:
            stamped = {}
        # 1. transcript links (new saves): call_id -> child session id
        linked: dict[str, str] = {}
        order: list[str] = []
        for m in messages:
            try:
                if not isinstance(m, dict) or m.get("role") != "tool" or m.get("name") != "task":
                    continue
                sid = m.get("session_id") or m.get("sessionId")
                if sid and str(sid) not in order:
                    order.append(str(sid))
                cid = m.get("tool_call_id")
                if cid and sid:
                    linked[str(cid)] = str(sid)
            except Exception:
                continue
        # 2. live family records (this run's spawns, incl. title for matching)
        try:
            records = list(self._sibling_records(parent_id))
        except Exception:
            records = []
        # 3. disk children (old saves, restarts): oldest first
        try:
            kids = [dict(c) for c in (self._children_of(parent_id) or [])]
        except Exception:
            kids = []
        try:
            from ..session import load_session as _load2
            for k in kids:
                ksess = self._sessions.get(k["id"]) or _load2(k["id"])
                if ksess is not None and not k.get("title"):
                    k["title"] = getattr(ksess, "title", "") or k.get("title", "")
        except Exception:
            pass
        try:
            bubbles = [b for b in chat.query("*")
                       if getattr(b, "role", "") == "tool"
                       and isinstance(getattr(b, "content", None), dict)
                       and b.content.get("tool") == "task"]
        except Exception:
            return 0
        if not bubbles:
            return 1  # rows not mounted yet — scheduler retries
        used: set[str] = set()
        missing = 0

        def _backfill(bubble: Any, target: str) -> None:
            # Runtime + toolcall count from the child's saved file. Runs for
            # EVERY linked row (not just newly linked ones): resumed parents
            # never saw the live finalize event, so their rows would otherwise
            # show no `↳ N toolcalls · Ns` even with sessionId present.
            try:
                meta = bubble.content.get("metadata") or {}
                if meta.get("duration") and meta.get("toolcalls") is not None:
                    return
                from ..session import load_session as _load3
                ksess = self._sessions.get(target) or _load3(target)
                if ksess is None:
                    return
                created = float(getattr(ksess, "created", 0) or 0)
                completed = float(getattr(ksess, "completed", 0) or 0)
                if completed > created and not meta.get("duration"):
                    bubble.set_tool_metadata("duration", format_duration(completed - created))
                if meta.get("toolcalls") is None:
                    try:
                        n_tools = sum(
                            1 for m in (getattr(ksess, "messages", None) or [])
                            if isinstance(m, dict) and m.get("role") == "tool")
                        bubble.set_tool_metadata("toolcalls", n_tools)
                    except Exception:
                        pass
            except Exception:
                pass

        for b in bubbles:
            try:
                meta = b.content.get("metadata") or {}
                if meta.get("sessionId"):
                    used.add(str(meta.get("sessionId")))
                    _backfill(b, str(meta.get("sessionId")))
                    continue
                bid = b.content.get("id") or b.content.get("call_id") or ""
                target = linked.get(str(bid)) if bid else None
                if target is None or target in used:
                    # title match against family records (description match)
                    inp = b.content.get("input") or {}
                    desc = str(inp.get("description") or meta.get("title") or "").strip().lower()
                    for r in records + kids:
                        rid = str(r.get("id") or "")
                        if not rid or rid in used:
                            continue
                        rt = str(r.get("title") or "").strip().lower()
                        if desc and rt and (desc == rt or desc in rt or rt in desc):
                            target = rid
                            break
                    if target is None:
                        # spawn stamps first (survive interrupts), then
                        # transcript order, then family records.
                        for rid in list(stamped) + order + [str(r.get("id")) for r in records + kids]:
                            if rid and rid not in used:
                                # prefer title agreement when a stamp exists
                                if rid in stamped and desc:
                                    rt = str((stamped[rid] or {}).get("title") or "").strip().lower()
                                    if rt and not (desc == rt or desc in rt or rt in desc):
                                        continue
                                target = rid
                                break
                if target:
                    b.set_tool_metadata("sessionId", target)
                    used.add(str(target))
                    _backfill(b, str(target))
                else:
                    missing += 1
            except Exception:
                missing += 1
                continue
        return missing

    def _link_task_row(self, sid: str, parent_id: str, call_id: str = "") -> None:
        """Attach a sub-agent's session id + start time to the exact task row
        that spawned it (the parent chat's row with the matching tool call id).

        With parallel sub-agents the events arrive concurrently, so "newest
        unattached running row" is wrong under the hood — the call id makes the
        mapping unambiguous (official opencode keys every sub-agent by its task
        call). A missing id keeps the old fallback for compatibility."""
        chat = self._chats.get(parent_id)
        if chat is None:
            return
        if call_id:
            bubble = chat.find_tool("task", call_id)
            if bubble is not None and bubble.content.get("status") == "running" and not (bubble.content.get("metadata") or {}).get("sessionId"):
                bubble.set_tool_metadata("sessionId", sid)
                self._task_start[sid] = time.monotonic()
                return
        for child in reversed(list(chat.query(MessageBubble))):
            if child.role != "tool" or child.content.get("tool") != "task":
                continue
            meta = child.content.get("metadata") or {}
            if child.content.get("status") == "running" and not meta.get("sessionId"):
                child.set_tool_metadata("sessionId", sid)
                self._task_start[sid] = time.monotonic()
                break

    def _finalize_task_row(self, chat: ChatView, run: dict[str, Any]) -> None:
        """Write the completed task row's runtime + toolcall count (the
        `↳ 3 toolcalls · 12.5s` line under a finished sub-agent)."""
        sid = (run.get("metadata") or {}).get("sessionId")
        if not sid:
            return
        bubble = chat.find_task(sid)
        if bubble is None:
            return
        start = self._task_start.pop(sid, None)
        if start is not None:
            bubble.set_tool_metadata("duration", format_duration(time.monotonic() - start))
        child = self._chats.get(sid)
        toolcalls = 0
        if child is not None:
            toolcalls = len([r for r in child.tool_runs() if r.get("status") == "completed"])
        bubble.set_tool_metadata("toolcalls", toolcalls)

    def _on_retry_event(self, session_id: str, event: dict[str, Any]) -> None:
        """A sub-agent's provider lane is retrying — paint its parent task row
        error-red with `↳ Retrying (attempt N) · …` (official Subagent retry)."""
        parent_id = self._child_parent.get(session_id)
        if not parent_id:
            return
        chat = self._chats.get(parent_id)
        if chat is None:
            return
        bubble = chat.find_task(session_id)
        if bubble is not None:
            bubble.set_tool_metadata(
                "retry",
                {"attempt": event.get("attempt", 1), "message": event.get("message", "")},
            )

    def _clear_task_retry(self, session_id: str) -> None:
        """The sub-agent made progress again — remove the retry decoration."""
        parent_id = self._child_parent.get(session_id)
        if not parent_id:
            return
        chat = self._chats.get(parent_id)
        if chat is None:
            return
        bubble = chat.find_task(session_id)
        if bubble is not None:
            bubble.pop_tool_metadata("retry")

    # -- prompt handling -------------------------------------------------
    def on_prompt_submitted(self, event: PromptSubmitted) -> None:
        value = event.value
        if not value.strip():
            return
        sid = self._current_session_id
        chat = self._chat_for(sid)
        engine = self._engines.get(sid) or self.engine
        session = self._sessions.get(sid) or self.session
        # always show what the user typed, then route it
        bubble = chat.append_user(value, agent=engine.agent)
        if value.lstrip().startswith("/"):
            self._run_command(value.lstrip())
            return
        # name the session from its first real message (opencode behaviour)
        if not getattr(session, "title", ""):
            session.title = value.strip()[:60]
        if self._busy:
            if sid != self._active_turn_session_id:
                # The running turn belongs to a DIFFERENT session (user resumed
                # an idle chat mid-stream). Queueing here parked the prompt in
                # an engine with no running turn — stuck "Queued" forever.
                chat.append_meta(
                    "⏳ Another session's request is still running — switch back "
                    "to it, or Ctrl+C to interrupt before sending here."
                )
                self.notify("Busy in another session", severity="warning")
                return
            # opencode's queue-and-promote: never drop a prompt typed while the
            # agent is busy. It goes into the ENGINE's FIFO (thread-safe, one
            # per session) and its bubble shows the ` QUEUED ` badge. The next
            # provider-turn boundary of the SAME running turn folds it in
            # (run_turn drains the queue), so there is no fresh-turn gap —
            # exactly how opencode keeps one Session Drain going.
            bubble.queued = True
            depth = engine.queue_prompt(value)
            self.notify(f"Queued ({depth}) — will run in the current turn")
            return
        self._start_turn(sid, value, engine)

    def _autosave_in_flight(self) -> None:
        """Periodic crash-safety save of the running turn's session.

        The engine keeps the in-flight assistant reply live in get_history()
        as the stream grows, so this captures the very last conversation. Runs
        on the app thread via a timer; a sudden process kill can only lose the
        tokens streamed since the previous tick.

        The durable write (temp file + fsync + atomic rename + .bak replica +
        index) can take 10-25ms on this phone's flash, so it is done on a
        background thread — never on the UI thread where it would hitch the
        render every 2s mid-stream. The thread never mutates the live Session
        (snapshot copy) and its write is guarded by an autosave generation so
        it can't race a newer turn/exit save.
        """
        if not self._busy:
            return
        if self._autosave_thread is not None and self._autosave_thread.is_alive():
            return  # a slower disk can't pile up saves
        try:
            sid = self._active_turn_session_id
            engine = self._engines.get(sid)
            sess = self._sessions.get(sid)
            if engine is not None and sess is not None:
                history = engine.get_history()
                # Only overwrite with a real conversation: the worker may not
                # have appended the prompt yet when the first tick fires, and
                # saving the empty history would destroy the durable copy that
                # _start_turn just wrote.
                if not history:
                    return
                generation = self._autosave_generation
                self._autosave_thread = threading.Thread(
                    target=self._autosave_write,
                    args=(sid, history, generation),
                    daemon=True,
                )
                self._autosave_thread.start()
        except Exception:
            pass

    def _autosave_write(self, sid: str, history: list[dict[str, Any]], generation: int) -> None:
        """Worker thread body: persist a snapshot WITHOUT mutating the live
        Session shared with the UI thread. Skipped via `should_write` if the
        autosave generation moved on (turn ended / exit save-all started)."""
        try:
            from ..session import save_session

            sess = self._sessions.get(sid)
            if sess is None:
                return
            snapshot = copy.copy(sess)
            snapshot.messages = list(history)
            # Double-check generation at write time (not just at start) to avoid
            # a stale worker overwriting a newer durable copy mid-write.
            if self._autosave_generation != generation:
                return
            save_session(
                snapshot,
                should_write=lambda: self._autosave_generation == generation,
            )
        except Exception:
            pass

    def _cancel_autosave(self) -> None:
        self._autosave_generation += 1  # invalidate any in-flight streaming autosave
        if self._autosave_timer is not None:
            try:
                self._autosave_timer.stop()
            except Exception:
                pass
            self._autosave_timer = None

    def _turn_state(self, sid: str) -> dict[str, Any]:
        """The per-session turn bookkeeping slot (created on first touch)."""
        st = self._turn.get(sid)
        if st is None:
            st = {
                "had_text": False,
                "had_reasoning": False,
                "had_error": False,
                "had_tools": False,
                "interrupted": False,
                "started": None,
            }
            self._turn[sid] = st
        return st

    def _start_turn(self, sid: str, value: str, engine: AgentLoop, resume: bool = False) -> None:
        """Start a model turn in a worker thread for the initial prompt.

        Prompts submitted while this turn runs are queued on the engine and
        folded into the SAME turn (one Session Drain) at the next provider-turn
        boundary; this is only the drain's starting point.

        With ``resume=True`` the engine re-runs its last user prompt instead
        (auto-resume after a reconnect) — nothing is appended or duplicated.
        Any reconnect watcher for this session is stopped: a live turn (or a
        fresh user prompt) supersedes waiting.
        """
        self._stop_reconnect_watch(sid)
        chat = self._chat_for(sid)
        # show an eager Thinking… bubble immediately (before the first token
        # arrives) so the UI reacts to Enter instead of sitting silent
        chat.begin_thinking()
        st = self._turn_state(sid)
        st.update(
            had_text=False,
            had_reasoning=False,
            had_error=False,
            had_tools=False,
            interrupted=False,
        )
        st["started"] = time.monotonic()
        # the previous turn's runtime disappears the moment the model starts
        # working again (official opencode shows it only on the final report)
        self._clear_last_duration()
        self._busy = True
        self._busy_sessions.add(sid)
        self._active_turn_session_id = sid
        # Streaming indicators follow the VIEWED session (not a global flag):
        # if the user is watching some other idle chat, it must not inherit
        # this turn's spinner/locked input.
        self._sync_streaming_visuals()
        # Persistence policy: starting a turn NEVER writes a file. A session
        # is only durably stored once one of the save conditions fires —
        # (1) the 2s crash-safety autosave tick while a real conversation
        # streams (phone dies / reboot mid-turn -> at most the last 2 seconds
        # are lost), (2) the final save-all on exit (ctrl+q / /exit), or (3) the
        # SIGTERM/SIGHUP handler when Termux is closed. Merely starting a turn
        # must never mint an empty or half-built session file.
        if self._autosave_timer is None:
            self._autosave_timer = self.set_interval(2.0, self._autosave_in_flight)

        def run():
            result = None
            try:
                result = engine.resume_turn() if resume else engine.run_turn(value)
            except Exception as e:  # never let a worker crash silently
                self.call_from_thread(self._show_error, f"{type(e).__name__}: {e}", False, sid)
            finally:
                # Pass THIS worker's sid: the global _active_turn_session_id can
                # already point at a newer turn (another session's prompt or a
                # promoted queued prompt), and finalizing the wrong session's
                # chat would clear the wrong flags and promote the wrong queue.
                self.call_from_thread(self._turn_done, result, sid)

        self.run_worker(run, thread=True)

    def _show_error(self, message: str, retryable: bool = False, session_id: str | None = None) -> None:
        sid = session_id or self.session.id
        self._turn_state(sid)["had_error"] = True
        chat = self._chat_for(sid)
        self._flush_deltas()
        chat.end_reasoning()
        chat.remove_last_stream_bubble()
        chat.append_meta(f"⚠ {message}")
        hint = " Retry, or check /connect for a model/API key." if retryable else ""
        self.notify(f"error: {message}{hint}", severity="error")
        chat.end_stream()
        # Reset the visual streaming state here too, not only in _turn_done. The
        # worker's `finally` normally calls _turn_done and clears these, but if
        # that call_from_thread ever fails (e.g. app unmount mid-error) the UI
        # must not stay stuck on an "streaming" indicator on the error path.
        self._streaming_visual_reset()

    def _clear_last_duration(self) -> None:
        """Hide the previous turn's runtime on the mode line.

        Official opencode shows the runtime (`▣ Build · model · 1m 12s`) only
        while the final report is displayed; it disappears as soon as the model
        starts doing things again (a new turn begins working / running tools).
        """
        try:
            self.query_one(InputBar).set_last_duration("")
        except Exception:
            pass

    def _streaming_visual_reset(self) -> None:
        """Defensively clear the streaming-indicator UI state (status bar +
        input bar). Does NOT touch _busy/_busy_sessions: those belong to the
        worker thread and are owned by _turn_done's finally, so resetting them
        here could race an active turn on another session. Indicators follow
        the VIEWED session, not a global flag."""
        self._sync_streaming_visuals()

    def _turn_done(self, result: Any = None, sid: str | None = None) -> None:
        if sid is not None:
            self._interrupt_flags[sid] = False
        self._cancel_autosave()
        # Never trust the global _active_turn_session_id here: it can already
        # point at a NEWER turn (another session's prompt, or the promoted
        # queued prompt of a just-finished drain). This worker finalizes the
        # session it actually ran — passed in by _start_turn's run().
        sid = sid or self._active_turn_session_id
        st = self._turn_state(sid)
        engine = self._engines.get(sid) or self.engine
        session = self._sessions.get(sid) or self.session
        interrupted = False
        turn_failed = False
        promote_next = False
        try:
            if result is not None and result.provider_id:
                # reflect the lane/model that actually answered (e.g. openrouter)
                engine.provider_id = result.provider_id
                engine.model_id = result.model_id or engine.model_id
                # count the answered turn toward the most-used default
                try:
                    from ..config import record_model_use
                    record_model_use(result.provider_id, engine.model_id)
                except Exception:
                    pass
                # rebuild_rotation() hits the network (model catalogs) — it
                # must NEVER run here on the UI thread (it froze the whole
                # screen at end of turn whenever the cache was stale). Flag
                # it; run_turn rebuilds on the engine thread before streaming.
                engine.mark_rotation_dirty()
                self._update_header()
            chat = self._chat_for(sid)
            status = self.query_one(StatusBar)
            status.set_retry_message("")
            self._flush_deltas()
            chat.end_reasoning()
            if not st["had_text"] and not st["had_reasoning"] and not st["had_error"] and not st["had_tools"] and not st["interrupted"]:
                # provider returned nothing (no text, no reasoning, no tool call,
                # no error) — drop the empty streaming cursor bubble before
                # end_stream clears its pointer.
                chat.remove_last_stream_bubble()
                chat.append_meta(
                    "(no reply from the model — check your connection and /connect "
                    "for a working model, or switch rotation in /config)"
                )
                self.notify("No reply from the model.", severity="warning")
            chat.end_stream()
            if st["had_text"] and not st["had_error"]:
                # the mode line lives fixed above the prompt box now, not in the chat
                self._update_header()
            # show the finished turn's runtime (`▣ Build · model · 1m 12s`) like
            # opencode's per-message footer, but ONLY on the final report. Official
            # opencode computes the duration when the message finished with a real
            # text answer (`finish` not tool-calls), so a tool-only, errored, or
            # interrupted turn shows no runtime — it appears again on the next
            # turn's last report.
            started = st["started"]
            st["started"] = None
            if started is not None:
                elapsed = time.monotonic() - started
                if st["had_text"] and not st["had_error"] and not st["interrupted"]:
                    try:
                        from .input_bar import format_duration

                        self.query_one(InputBar).set_last_duration(format_duration(elapsed))
                    except Exception:
                        pass
            interrupted = st["interrupted"]
            turn_failed = bool(getattr(result, "error", "")) if result is not None else False
            # opencode's queue-and-promote fallback: normally the ENGINE folds
            # queued prompts into the running turn (one Session Drain) — but a
            # prompt submitted in the tiny window after the turn's last boundary
            # but before it returns sits in the engine queue. Promote it as the
            # start of the next drain so nothing is ever left stuck "queued".
            # An interrupted turn stops the drain for real: the remaining queue
            # stays queued (the ` QUEUED ` badges keep showing) and only runs when
            # the next prompt starts a fresh drain.
            promote_next = (
                not interrupted
                and not turn_failed
                and engine.prompt_pending() > 0
            )
        finally:
            # Guaranteed reset: ANY exception above (widget lookups while a
            # modal screen is up, render errors, engine bookkeeping) used to
            # abort this method BEFORE _busy was cleared — leaving the input
            # locked and every new prompt queued forever ("app stops
            # working"). The busy/streaming reset must survive any failure.
            self._busy = False
            self._busy_sessions.discard(sid)
            # A force-stop is spent once nothing is busy anymore: a stale set
            # flag would instantly reject the next turn's dialogs.
            try:
                if not self._busy_sessions:
                    self._force_stop.clear()
            except Exception:
                pass
            # Indicators reflect the VIEWED session: if the finished turn ran
            # in a background session, its spinner was never on screen — and
            # if another turn is still streaming elsewhere, keep that honest.
            self._sync_streaming_visuals()
            # Keep the in-memory snapshot fresh (a later exit/close save reads it)
            # but persist NOTHING at turn end: per the save policy a conversation
            # file is only written on exit / Termux close / a crash mid-turn (the
            # 2s autosave tick already snapshotted it while streaming).
            try:
                session.messages = engine.get_history()
            except Exception:
                pass
            # this turn's bookkeeping slot is fully consumed — drop it (each future
            # turn / session / sub-agent owns its own slot, so nothing leaks across)
            self._turn.pop(sid, None)
            self._disarm_interrupt_escape()
        if promote_next:
            value = engine.pop_prompt()
            try:
                self._chat_for(sid).promote_next_queued()
            except Exception:
                pass
            self.notify("Running queued request")
            if value:
                self._start_turn(sid, value, engine)
                return
        if turn_failed and engine.prompt_pending() > 0:
            # the drain died mid-way: leave the remaining prompts QUEUED (badges
            # stay) instead of machine-gunning them into a failing provider
            self.notify(
                "Queued requests are waiting — press Enter to continue.",
                severity="warning",
            )
        if getattr(result, "network_failed", False) and not self._exit_requested.is_set():
            # The turn died on transport (disconnect/DNS/timeout), not on a
            # model error: watch connectivity and resume automatically.
            self._start_reconnect_watch(sid)
        try:
            self.query_one(InputBar).focus()
        except Exception:
            pass

    # -- auto-resume after reconnect --------------------------------------
    def _start_reconnect_watch(self, sid: str) -> None:
        """Watch connectivity in the background after a network-killed turn.

        Cheap and fast: short probes with backoff (3s → 60s), one tiny
        request per interval. When the route is back, the failed turn resumes
        by itself. Any new turn on this session (or app exit) stops the watch.
        """
        self._stop_reconnect_watch(sid)
        stop = threading.Event()
        self._reconnect_watchers[sid] = stop
        try:
            self.notify("Connection lost — will resume automatically when back online.")
        except Exception:
            pass

        def _watch() -> None:
            intervals = (3.0, 3.0, 5.0, 5.0, 10.0, 15.0, 20.0, 30.0)
            i = 0
            while not stop.is_set() and not self._exit_requested.is_set():
                if _probe_online():
                    try:
                        self.call_from_thread(self._auto_resume_turn, sid)
                    except Exception:
                        pass
                    return
                wait = intervals[i] if i < len(intervals) else 60.0
                i += 1
                stop.wait(timeout=wait)
            try:
                self._reconnect_watchers.pop(sid, None)
            except Exception:
                pass

        threading.Thread(target=_watch, name=f"reconnect-{sid[:8]}", daemon=True).start()

    def _stop_reconnect_watch(self, sid: str) -> None:
        stop = self._reconnect_watchers.pop(sid, None)
        if stop is not None:
            try:
                stop.set()
            except Exception:
                pass

    def _auto_resume_turn(self, sid: str) -> None:
        """Connectivity is back: resume the network-killed turn by itself."""
        self._reconnect_watchers.pop(sid, None)
        if self._exit_requested.is_set():
            return
        engine = self._engines.get(sid)
        if engine is None:
            return
        if sid in self._busy_sessions or self._busy:
            # user already started something — don't double-run
            return
        if engine.prompt_pending() > 0:
            # user queued follow-ups meanwhile: run those normally instead
            value = engine.pop_prompt()
            if value:
                try:
                    self._chat_for(sid).promote_next_queued()
                except Exception:
                    pass
                self.notify("Back online — running your queued request.")
                self._start_turn(sid, value, engine)
            return
        if sid != self._current_session_id:
            # user moved to another chat: don't hijack it, just report back
            self.notify("Back online — switch back and press Enter to resume.")
            return
        self.notify("Back online — resuming…")
        self._start_turn(sid, "", engine, resume=True)

    # -- command handling -------------------------------------------------
    def _run_command(self, line: str) -> None:
        from ..commands import handle_command
        from ..commands import CommandContext

        # /models is a full-screen, live model list. The bare form is already
        # intercepted by the command popup; with arguments it would fall through
        # to the sync fetch_zen_models() in commands.py and freeze the UI thread,
        # so route it to the picker (which fetches off-thread) instead.
        # /sessions opens the same opencode-style picker as Ctrl+R (the plain
        # /sessions command only prints a text list).
        stripped = line.strip()
        name = (
            stripped[1:].split(maxsplit=1)[0]
            if stripped.startswith("/") and len(stripped) > 1
            else ""
        )
        if name == "models":
            self._open_model_picker()
            return
        if name == "sessions":
            self.action_sessions()
            return
        if name == "mcp":
            # bare /mcp (or list/test) opens the manager popup; add/remove
            # with arguments run inline through the command handler.
            rest = stripped[1 + len(name):].strip()
            first = rest.split(maxsplit=1)[0].lower() if rest else ""
            if not rest or first in ("list", "ls", "test", "check", "ping", "picker", "ui"):
                if not rest:
                    self._open_mcp_picker()
                    return
                if first in ("picker", "ui"):
                    self._open_mcp_picker()
                    return
                # fall through to the handler for list/test variants
            # else fall through to the command handler (add/remove path)
        if name == "thinking":
            # bare /thinking (or with show/hide/last) opens the popup; effort
            # levels passed inline (e.g. `/thinking high`) apply directly.
            rest = stripped[1 + len(name):].strip()
            if not rest or rest.lower() in ("show", "hide", "last", "toggle", "on", "off", "expand", "collapse", "all"):
                self._open_thinking_picker()
                return
            # else fall through to the command handler (effort level path)
        if name in ("new", "clear"):
            # commands._new is headless-shaped (no get_session callback here),
            # so it used to reply "New session." and do NOTHING. Do it for real.
            # BUT never mid-turn: this interception sat before the busy gate,
            # so /clear while streaming wiped the running engine's history out
            # from under its worker thread.
            if self._busy:
                self._chat_for(self._current_session_id).append_meta(
                    "⏳ still working on the previous request…"
                )
                self.notify("Busy — finish or interrupt the running request first (Ctrl+C)")
                return
            self._action_new()
            self._chat_for(self._current_session_id).append_meta(
                "Started a new session — the previous one stays in the picker (Ctrl+R)."
            )
            return
        # Mutating commands must not run mid-turn: they'd race the running
        # engine (e.g. /undo popping the undo stack the worker is appending to).
        if self._busy and name not in _SAFE_WHILE_BUSY:
            self._chat_for(self._current_session_id).append_meta(
                "⏳ still working on the previous request…"
            )
            self.notify("Busy — finish or interrupt the running request first (Ctrl+C)")
            return

        engine = self._active_engine()
        session = self._active_session()

        def reply(text: str) -> None:
            # persistent chat output + a short toast; /models & friends must
            # not vanish into a transient notification
            self._chat_for(self._current_session_id).append_meta(text)
            self.notify(text.splitlines()[0][:60] if text else "", timeout=3, markup=False)

        ctx = CommandContext(
            config=self.cfg,
            auth=self.auth,
            session=session,
            engine=engine,
            worktree=str(self.directory),
            reply=reply,
            get_session=lambda: self._active_session(),
            set_model=self._set_model,
            set_agent=self._set_agent,
            exit_app=self.exit,
            resume=self._resume_session,
            connect=self._open_connect,
            registry=self.command_registry,
        )
        handle_command(self.command_registry, ctx, line)
        self._update_header()

    def _preview_command(self, name: str) -> str:
        """Run a read-only command with a collecting reply and return its output."""
        from ..commands import CommandContext, handle_command

        collected: list[str] = []
        ctx = CommandContext(
            config=self.cfg,
            auth=self.auth,
            session=self._active_session(),
            engine=self._active_engine(),
            worktree=str(self.directory),
            reply=collected.append,
            get_session=lambda: self._active_session(),
            set_model=self._set_model,
            set_agent=self._set_agent,
            exit_app=self.exit,
            resume=self._resume_session,
            connect=self._open_connect,
            registry=self.command_registry,
            # preview pass: handlers must not mutate anything (/export used to
            # write the file here, then AGAIN when Run pressed it)
            preview_only=True,
        )
        handle_command(self.command_registry, ctx, f"/{name}")
        return "\n".join(collected)

    def on_command_selected(self, event: CommandSelected) -> None:
        # /models is the full-screen, live-updating provider model list
        if event.name == "models":
            self._open_model_picker()
            return
        if event.name == "sessions":
            self.action_sessions()
            return
        if event.name == "theme":
            self._open_theme_picker()
            return
        if event.name == "thinking":
            self._open_thinking_picker()
            return
        if event.name == "mcp":
            self._open_mcp_picker()
            return
        cmd = self.command_registry.get(event.name)
        content: str | None = None
        if cmd is not None and cmd.preview:
            content = self._preview_command(event.name).strip() or cmd.description

        def done(result: str | None) -> None:
            bar = self.query_one(InputBar)
            bar.input.focus()
            if result == "run":
                self._run_command(f"/{event.name}")
            elif result == "cancel":
                # Esc = back: put the command back in the input, cursor at the
                # end. ("close" leaves the input empty — the user READ the
                # output; stuffing "/name" back used to leave stray text.)
                bar.input.value = f"/{event.name}"
                bar.input.cursor_position = len(bar.input.value)

        from .command_popup import CommandPopup

        self.push_screen(
            CommandPopup(
                event.name,
                event.description,
                content=content,
                usage=_COMMAND_USAGE.get(event.name, ""),
            ),
            done,
        )

    def _open_model_picker(self) -> None:
        def on_picked(choice: str | None) -> None:
            try:
                self.query_one(InputBar).input.focus()
            except Exception:
                pass
            if not choice:
                return
            provider, _, model = choice.partition("/")
            if not provider or not model:
                return
            self.cfg.provider = provider
            self.cfg.model = model
            engine = self._active_engine()
            engine.provider_id = provider
            engine.model_id = model
            # Network-touching rebuild runs at the next turn start (engine
            # thread), never inline here on the UI thread.
            engine.mark_rotation_dirty()
            self.notify(f"Model set to {provider}/{model} (this session)")
            self._update_header()
            # Deliberately NOT save_config() here: picking a model to TRY must
            # not silently rewrite the user's default in opencode.json. The
            # Settings screen persists explicit choices.

        from .model_picker import ModelPicker

        self.push_screen(
            ModelPicker(current=self.cfg.model, cfg=self.cfg, auth=self.auth),
            on_picked,
        )

    def _open_theme_picker(self) -> None:
        """Arrow-navigable theme list (dark first); Enter applies it live."""
        from .theme import set_active_theme
        from .theme_picker import ThemePicker

        def done(choice: str | None) -> None:
            try:
                self.query_one(InputBar).input.focus()
            except Exception:
                pass
            if not choice:
                return
            self.cfg.theme = choice
            set_active_theme(choice)
            save_config(self.cfg)
            self.notify(f"Theme set to {choice}")

        self.push_screen(ThemePicker(current=self.cfg.theme), done)

    def _open_thinking_picker(self) -> None:
        """Thinking popup: current model + on/off + effort levels + show/hide.

        Enter applies the highlighted row immediately (effort persists and
        takes effect next turn; on/off flips bubble visibility live).
        """
        from .thinking_picker import ThinkingPicker

        engine = self._active_engine()
        try:
            model_id = getattr(engine, "model_id", "") or self.cfg.model
            provider_id = getattr(engine, "provider_id", "") or self.cfg.provider
        except Exception:
            model_id, provider_id = self.cfg.model, self.cfg.provider
        try:
            from ..providers.rotation import model_effort_levels
            levels = model_effort_levels(model_id, provider_id or "opencode")
        except Exception:
            levels = []
        try:
            chat = self._chat_for(self._current_session_id)
            n_thoughts = len(chat.reasoning_bubbles())
        except Exception:
            n_thoughts = 0

        def done(choice: str | None) -> None:
            try:
                self.query_one(InputBar).input.focus()
            except Exception:
                pass
            if not choice:
                return
            if choice == "__on__":
                self.cfg.show_thoughts = True
                save_config(self.cfg)
                self._set_thoughts_visible(True)
                self.notify("Thinking on — thought bubbles show.")
            elif choice == "__off__":
                self.cfg.show_thoughts = False
                save_config(self.cfg)
                self._set_thoughts_visible(False)
                self.notify("Thinking off — thoughts hidden (kept in history).")
            elif choice == "__show__":
                self._expand_all_thoughts()
            elif choice == "__hide__":
                self._collapse_all_thoughts()
            elif choice.startswith("__effort__"):
                self._set_reasoning_effort(choice[len("__effort__"):])

        self.push_screen(
            ThinkingPicker(
                model=f"{provider_id}/{str(model_id).split('/', 1)[-1]}",
                thinking_on=bool(getattr(self.cfg, "show_thoughts", True)),
                effort=str(getattr(self.cfg, "reasoning_effort", "") or ""),
                levels=levels,
                current_thoughts=n_thoughts,
            ),
            done,
        )

    def _set_thoughts_visible(self, visible: bool) -> None:
        """Apply the Thinking on/off switch to every mounted chat now."""
        try:
            for chat in list(getattr(self, "_chats", {}).values()):
                try:
                    chat.set_thoughts_visible(visible)
                except Exception:
                    continue
        except Exception:
            pass
        try:
            self._update_header()
        except Exception:
            pass

    def _expand_all_thoughts(self) -> None:
        try:
            chat = self._chat_for(self._current_session_id)
            n = chat.set_all_reasoning(True)
            self.notify(f"Expanded {n} thought(s).")
        except Exception:
            pass

    def _collapse_all_thoughts(self) -> None:
        try:
            chat = self._chat_for(self._current_session_id)
            n = chat.set_all_reasoning(False)
            self.notify(f"Collapsed {n} thought(s).")
        except Exception:
            pass

    def _set_reasoning_effort(self, level: str) -> None:
        """Validate an effort level against the CURRENT model and apply it."""
        level = str(level or "").strip().lower()
        try:
            engine = self._active_engine()
            model_id = getattr(engine, "model_id", "") or self.cfg.model
            provider_id = getattr(engine, "provider_id", "") or self.cfg.provider
            from ..providers.rotation import model_effort_levels
            levels = model_effort_levels(model_id, provider_id or "opencode")
        except Exception:
            levels, model_id = [], ""
        if not levels:
            self.notify("This model has no effort levels — fixed thinking.", severity="warning")
            return
        if level not in [str(v).lower() for v in levels]:
            self.notify(f"Effort must be one of: {', '.join(levels)}", severity="warning")
            return
        self.cfg.reasoning_effort = level
        save_config(self.cfg)
        try:
            self._apply_runtime_settings()
        except Exception:
            pass
        self.notify(f"Reasoning effort → {level} (next turn).")

    def _open_connect(self, provider: str = "") -> None:
        from .connect_screen import ConnectScreen

        self.app.push_screen(
            ConnectScreen(auth=self.auth, on_connected=self._on_connected, initial=provider),
            self._on_connect_dismissed,
        )

    def _on_connected(self, provider_id: str) -> None:
        self.notify(f"Saved API key for {provider_id}.")
        self._update_header()

    def _on_connect_dismissed(self, result: str | None) -> None:
        if result:
            self.notify(f"Connected {result}.")

    def _open_mcp_picker(self) -> None:
        """MCP manager popup: list/add/remove/test with arrows + Inputs."""
        from .. import mcp_manager as _mm
        from .mcp_picker import McpPicker

        servers = _mm.list_servers(self.cfg)

        def done(choice: str | None) -> None:
            try:
                from .input_bar import InputBar

                self.query_one(InputBar).input.focus()
            except Exception:
                pass
            if not choice:
                return
            chat = self._chat_for(self._current_session_id)
            if choice == "__test__":
                chat.append_meta("Testing MCP servers (up to ~15s each)…")
                self.run_worker(lambda: self._mcp_test_all(chat), thread=True)
                return
            if choice.startswith("__remove__"):
                name = choice[len("__remove__"):]
                path = _mm.remove_server(name, str(self.directory), False)
                raw = getattr(self.cfg, "raw", None)
                if isinstance(raw, dict) and isinstance(raw.get("mcpServers"), dict):
                    raw["mcpServers"].pop(name, None)
                if path is None:
                    self.notify(f"No MCP server '{name}'.", severity="warning")
                    chat.append_meta(f"No MCP server '{name}'.")
                    return
                try:
                    engine = self._active_engine()
                except Exception:
                    engine = None
                note = _mm.refresh_engine_mcp(engine, self.cfg)
                self.notify(f"Removed '{name}'. {note}")
                chat.append_meta(f"Removed MCP '{name}'. {note}")
                return
            if choice.startswith("__srv__"):
                name = choice[len("__srv__"):]
                spec = servers.get(name, {})
                cmd = spec.get("command", "?") if isinstance(spec, dict) else "?"
                chat.append_meta(
                    f"MCP '{name}': {cmd} — test: /mcp test {name} · remove: pick ⌧ in /mcp"
                )
                return
            if choice.startswith("__add__"):
                rest = choice[len("__add__"):]
                sep = rest.find("__")
                if sep < 0:
                    return
                name, cmdline = rest[:sep], rest[sep + 2:]
                import shlex as _shlex

                try:
                    parts = _shlex.split(cmdline)
                except ValueError as e:
                    self.notify(f"Bad run line: {e}", severity="error")
                    return
                if not parts:
                    return
                command, cmd_args = parts[0], parts[1:]
                warns = _mm.check_command(command, cmd_args)
                for w in warns:
                    self.notify(w, severity="warning")
                chat.append_meta(f"Adding MCP '{name}': {command} {' '.join(cmd_args)}…".rstrip())
                self.run_worker(
                    lambda: self._mcp_add(chat, name, command, cmd_args),
                    thread=True,
                )

        self.push_screen(McpPicker(servers=servers), done)

    def _mcp_add(self, chat: Any, name: str, command: str, cmd_args: list) -> None:
        from .. import mcp_manager as _mm

        line = _mm.test_server(name, command, list(cmd_args), timeout=15.0)
        try:
            path = _mm.save_server(name, command, list(cmd_args), str(self.directory), False)
        except OSError as e:
            self.call_from_thread(chat.append_meta, f"{line}\nSave FAILED: {e}")
            return
        raw = getattr(self.cfg, "raw", None)
        if isinstance(raw, dict):
            servers = raw.get("mcpServers")
            if not isinstance(servers, dict):
                servers = {}
                raw["mcpServers"] = servers
            servers[name] = {"command": command, "args": list(cmd_args)}
        try:
            engine = self._active_engine()
        except Exception:
            engine = None
        note = _mm.refresh_engine_mcp(engine, self.cfg)
        self.call_from_thread(
            chat.append_meta, f"{line}\nSaved '{name}' → {path}\n{note}"
        )
        self.call_from_thread(self.notify, f"MCP '{name}' added. {note}")

    def _mcp_test_all(self, chat: Any) -> None:
        from .. import mcp_manager as _mm

        servers = _mm.list_servers(self.cfg)
        if not servers:
            self.call_from_thread(chat.append_meta, "No MCP servers configured.")
            return
        lines = [
            _mm.test_server(n, s.get("command"), (s.get("args") or []), timeout=15.0)
            for n, s in servers.items()
            if isinstance(s, dict) and s.get("command")
        ]
        self.call_from_thread(chat.append_meta, "\n".join(lines) or "No testable servers.")

    # -- actions ---------------------------------------------------------
    def on_agent_toggle_requested(self, event: AgentToggleRequested) -> None:
        self.action_toggle_agent()

    def on_models_requested(self, event: Any) -> None:
        self.action_models()

    def _interrupt_requested(self, sid: str | None = None) -> bool:
        if sid is None:
            sid = self._current_session_id
        return bool(self._interrupt_flags.get(sid, False))

    def _interrupt_engines(self, sid: str | None = None) -> None:
        """Force-close the active provider stream for a specific session.

        Flipping the interrupt flag alone only stops the turn at the next
        per-chunk check; an idle "thinking" gap (no chunks arriving) would keep
        the stream blocked until the model emits. Closing the stream makes the
        blocked read surface the interrupt immediately."""
        if sid is None:
            sid = self._current_session_id
        engine = self._engines.get(sid) or (self.engine if self.engine and self.engine.session_id == sid else None)
        if engine is not None:
            try:
                engine.abort()
            except Exception:
                pass

    def action_interrupt(self) -> None:
        # Flipping the session's flag makes run_turn stop at its next iteration
        # check (loop.py). The worker thread's `finally` calls _turn_done,
        # which resets the flag and clears _busy — we must NOT call _turn_done
        # here, or the worker would finish concurrently and double-complete.
        sid = self._active_turn_session_id or self._current_session_id
        if self._busy_sessions and sid in self._busy_sessions:
            self.notify("Interrupting...")
            self._interrupt_flags[sid] = True
            self._interrupt_engines(sid)
            self._disarm_interrupt_escape()

    def action_interrupt_escape(self) -> None:
        """ESC arms on first press, force-stops everything on second.

        First press (busy): only arms the `esc again` hint in the footer —
        nothing is interrupted yet. Second press (still busy within 5s):
        force-stops ANYTHING running — all busy sessions' flags, every
        engine's streams/fetches/sub-agents, all background tasks, and any
        open modal. Nothing is ignored: the worker threads can't miss it
        (flags + closed sockets + dismissed dialogs + unblocked waits).
        Idle sessions just move focus back to the prompt.
        """
        self._cancel_esc_timer()
        if not self._busy_sessions:
            self._force_stop.clear()
            try:
                self.query_one(InputBar).focus()
            except Exception:
                pass
            return
        # opencode: `setStore("interrupt", store.interrupt + 1)` on every press
        self._esc_presses += 1
        self._esc_timer = self.set_timer(5.0, self._disarm_interrupt_escape)
        if self._esc_presses >= 2:
            self._force_stop_all()
            self._disarm_interrupt_escape()
            return
        self._arm_interrupt_escape(armed=True)

    def _force_stop_all(self) -> None:
        """Second ESC: stop ANYTHING still running, nothing ignored."""
        for sid in list(self._busy_sessions):
            self._interrupt_flags[sid] = True
            self._interrupt_engines(sid)
        # unblock workers stuck in a permission/question modal wait
        self._force_stop.set()
        # dismiss whatever modal is on top (a stuck dialog must not trap the
        # worker after the user demanded a stop)
        try:
            if not self._is_main_screen_active():
                self.pop_screen()
        except Exception:
            pass
        # background shell tasks are work too — stop them all
        stopped = 0
        try:
            from ..tools import background as _bg

            stopped = _bg.stop_all()
        except Exception:
            pass
        self.notify(f"Force-stopped{f' ({stopped} background task(s))' if stopped else ''}.")

    def _arm_interrupt_escape(self, armed: bool) -> None:
        if armed:
            self._esc_presses = 1
        else:
            self._esc_presses = 0
            self._cancel_esc_timer()
        try:
            self.query_one(StatusBar).set_interrupt_armed(armed)
        except Exception:
            pass

    def _cancel_esc_timer(self) -> None:
        if self._esc_timer is not None:
            self._esc_timer.stop()
            self._esc_timer = None

    # -- auto-refocus -------------------------------------------------------
    def _is_main_screen_active(self) -> bool:
        """True when the normal chat screen is on top (no modal / app exiting)."""
        try:
            return self._main_screen is not None and self.screen is self._main_screen
        except Exception:
            return False

    def on_descendant_focus(self, event: Any) -> None:
        """Drag the prompt cursor back ~1s after focus leaves the input box.

        Tapping a reasoning bubble (to expand it) or the chat area on a phone
        steals focus, so the blinking cursor disappears and typing stops. Any
        focus change outside the prompt arms a short timer; if the user touches
        nothing else in the meantime, the cursor comes back by itself."""
        if self._refocus_timer is not None:
            self._refocus_timer.stop()
            self._refocus_timer = None
        if not self._is_main_screen_active():
            # a picker / dialog on top, or the app is shutting down — never
            # steal focus or arm timers
            return
        try:
            if self.query_one(InputBar).input.has_focus:
                return
        except Exception:
            return
        self._refocus_timer = self.set_timer(1.0, self._refocus_prompt)

    def _refocus_prompt(self) -> None:
        self._refocus_timer = None
        if not self._is_main_screen_active():
            return
        try:
            self.query_one(InputBar).input.focus()
        except Exception:
            pass

    def _disarm_interrupt_escape(self) -> None:
        self._arm_interrupt_escape(False)

    # -- permission dialog (engine thread -> UI) --------------------------
    def _permission_ask(self, description: str, always_patterns: list[str]) -> str:
        """Bridge the engine thread's permission.ask to a modal dialog.

        Runs on the engine worker thread. Pushes the dialog on the UI thread via
        call_from_thread (which blocks until the push returns), then waits for the
        user's decision. Returns "once" / "always" / "reject".
        """
        outcome: dict[str, str] = {}
        decided = threading.Event()

        def on_decision(decision: str) -> None:
            outcome["decision"] = decision
            decided.set()

        with self._dialog_lock:
            try:
                self.call_from_thread(
                    self._show_permission_dialog, description, on_decision
                )
            except Exception:
                return "reject"
            # Wait for user decision with timeout; if timeout expires, default to "reject"
            # to avoid hanging the engine thread indefinitely if UI is unresponsive.
            # The window starts AFTER the dialog is shown (lock held), so queued
            # agents each get their full 30s instead of timing out behind others.
            # A force-stop (2nd ESC) breaks the wait immediately as "reject" —
            # the worker must never sit out the full timeout after STOP.
            start = time.monotonic()
            force = getattr(self, "_force_stop", None)
            while not self._exit_requested.is_set():
                if force is not None and force.is_set():
                    break
                remaining = _DIALOG_TIMEOUT - (time.monotonic() - start)
                if remaining <= 0:
                    break
                if decided.wait(timeout=min(0.5, remaining)):
                    break
        return outcome.get("decision", "reject")

    def _show_permission_dialog(
        self, description: str, on_decision: Any
    ) -> None:
        if not self.is_attached:
            on_decision("deny")
            return
        from .permission_dialog import PermissionDialog

        self.push_screen(PermissionDialog(description, on_decision=on_decision))

    # -- question dialog (engine thread -> UI) ----------------------------
    def _question_ask(self, questions: list[QuestionInfo]) -> list[list[str]]:
        """Bridge the engine thread's question.ask to a modal dialog.

        Runs on the engine worker thread, mirroring ``_permission_ask``.
        Returns the answers (list of list[str], one per question) or raises
        QuestionRejectedError when the user dismisses / the app is quitting.
        """
        from ..question import QuestionInfo, QuestionRejectedError

        result: dict[str, Any] = {}
        answered = threading.Event()

        def on_done(answers: list[list[str]] | None) -> None:
            result["answers"] = answers
            answered.set()

        with self._dialog_lock:
            try:
                self.call_from_thread(
                    self._show_question_dialog, questions, on_done
                )
            except Exception:
                raise QuestionRejectedError("no UI to ask the user") from None
            # Wait for user answers with timeout; if timeout expires, treat as dismissed.
            # A force-stop (2nd ESC) breaks the wait immediately as dismissed.
            start = time.monotonic()
            force = getattr(self, "_force_stop", None)
            while not self._exit_requested.is_set():
                if force is not None and force.is_set():
                    break
                remaining = _DIALOG_TIMEOUT - (time.monotonic() - start)
                if remaining <= 0:
                    break
                if answered.wait(timeout=min(0.5, remaining)):
                    break
        answers = result.get("answers")
        if answers is None:
            raise QuestionRejectedError("user dismissed the question")
        return answers

    def _show_question_dialog(
        self, questions: list[QuestionInfo], on_done: Any
    ) -> None:
        if not self.is_attached:
            on_done(None)
            return
        from .question_dialog import QuestionDialog

        self.push_screen(QuestionDialog(questions, on_done=on_done))

    def _save_all_live_sessions(self) -> None:
        """Final persist of every LIVE session that actually has a conversation.

        This is one of the ONLY durable-write paths (besides the in-flight
        crash-safety autosave tick). It runs when the user quits (ctrl+q /
        /exit), when the TUI tears down, and when Termux closes the app via
        SIGTERM/SIGHUP. Sessions with no conversation are skipped, so an
        untouched scratch session never leaves a file behind.
        """
        from ..session import save_session

        # Newer than any in-flight autosave: a worker that hasn't written yet
        # must not overwrite these final bodies with its (older) snapshot.
        self._autosave_generation += 1
        for sid, sess in list(self._sessions.items()):
            engine = self._engines.get(sid)
            history = None
            if engine is not None:
                try:
                    history = engine.get_history()
                except Exception:
                    history = None
            if not history:
                history = list(getattr(sess, "messages", None) or [])
            if not history:
                continue
            # Preserve child-session links stamped onto the in-memory bubbles
            # by the relinker: the engine's history copy predates them (it was
            # snapshotted at turn end), so a blind overwrite would ERASE the
            # clickability we just healed — the exact regression that wiped
            # this parent's links on exit.
            try:
                chat = self._chats.get(sid)
                if chat is not None:
                    live_links: dict[str, str] = {}
                    for b in chat.query("*"):
                        try:
                            if getattr(b, "role", "") != "tool":
                                continue
                            c = getattr(b, "content", None)
                            if not isinstance(c, dict) or c.get("tool") != "task":
                                continue
                            meta = c.get("metadata") or {}
                            lsid = meta.get("sessionId")
                            bid = c.get("id") or c.get("call_id")
                            if lsid and bid:
                                live_links[str(bid)] = str(lsid)
                        except Exception:
                            continue
                    if live_links:
                        for m in history:
                            try:
                                if isinstance(m, dict) and m.get("role") == "tool" and m.get("name") == "task":
                                    cid = str(m.get("tool_call_id") or "")
                                    if cid in live_links and not (m.get("session_id") or m.get("sessionId")):
                                        m["session_id"] = live_links[cid]
                                        m["sessionId"] = live_links[cid]
                            except Exception:
                                continue
            except Exception:
                pass
            sess.messages = history
            try:
                save_session(sess)
            except Exception:
                pass

    def on_exit_app(self) -> None:
        """Fired when the app quits (ctrl+q / /exit).

        Unblocks any engine thread waiting on a permission dialog and persists
        the final state of every live session that has a conversation.
        """
        self._exit_requested.set()
        try:
            self._save_all_live_sessions()
        except Exception:
            pass

    def on_unmount(self) -> None:
        """Final persist as the TUI tears down (same policy as on_exit_app)."""
        try:
            self._save_all_live_sessions()
        except Exception:
            pass

    def action_resume(self) -> None:
        """Ctrl+R: open the session picker to continue a past conversation."""
        self.action_sessions()

    def action_toggle_agent(self) -> None:
        engine = self._active_engine()
        self._set_agent("plan" if engine.agent != "plan" else "build")

    def action_settings(self) -> None:
        from .settings_screen import SettingsScreen

        self.push_screen(
            SettingsScreen(
                cfg=self.cfg,
                engine=self.engine,
                auth=self.auth,
                session=self.session,
                on_model_change=self._set_model,
                on_apply=self._apply_runtime_settings,
            ),
            self._on_settings_done,
        )

    def _apply_runtime_settings(self) -> None:
        """Push startup-captured settings into the LIVE components.

        Before this, a Settings change only touched the config FILE: every
        live engine kept its construction-time provider/model/agent snapshot,
        so old/resumed sessions answered with the OLD model until restart.
        Now model/provider/agent/rotation-lane-0 follow the new pick on the
        very next turn — for every live engine AND every resumed session.
        The bash tool snapshots max_lines/max_bytes/timeout when its registry
        is built, and each rotation lane bakes its httpx timeout into the
        provider instance — both are refreshed here too."""
        from ..tools import bash as bash_mod

        engines = list(dict.fromkeys(
            [e for e in self._engines.values() if e is not None]
            + ([self._main_engine] if self._main_engine is not None else [])
        ))
        new_provider = str(getattr(self.cfg, "provider", "") or "")
        new_model = str(getattr(self.cfg, "model", "") or "")
        new_agent = str(getattr(self.cfg, "default_agent", "") or "")
        mode = getattr(self.cfg, "permission_mode", "auto")
        for engine in engines:
            perm = getattr(engine, "permission", None)
            if perm is not None and hasattr(perm, "mode"):
                try:
                    raw = str(mode or "auto").lower()
                    perm.mode = raw if raw in ("ask", "deny", "fully_auto") else "auto"
                except Exception:
                    pass
            # keep the question tool in sync: entering fully_auto installs the
            # auto-answer hook, leaving it restores the dialog bridge.
            try:
                reg = getattr(engine, "registry", None)
                qs = getattr(engine, "question_service", None)
                if reg is not None and qs is not None:
                    if getattr(perm, "mode", "") == "fully_auto":
                        def _fully_auto_ask(questions: list) -> list[list[str]]:
                            out: list[list[str]] = []
                            for q in questions:
                                opts = getattr(q, "options", None) or []
                                first = getattr(opts[0], "label", "") if opts else ""
                                out.append([first] if first else [])
                            return out
                        reg.question_asker = _fully_auto_ask
                    else:
                        reg.question_asker = qs.ask
            except Exception:
                pass
        for engine in engines:
            # model/provider/agent follow the new pick NOW, not on restart.
            # The engine streams with its own snapshot (provider_id/model_id/
            # agent), so without this old sessions answer with the OLD model
            # forever. Rotation lanes rebuild from cfg at next-turn start and
            # put the new pick at lane 0.
            try:
                if new_provider:
                    engine.provider_id = new_provider
                if new_model:
                    engine.model_id = new_model
            except Exception:
                pass
            try:
                sess_id = getattr(engine, "session_id", "") or ""
                sess = self._sessions.get(sess_id)
                if sess is not None:
                    if new_provider:
                        sess.provider = new_provider
                    if new_model:
                        sess.model = new_model
                    if new_agent and not getattr(sess, "parent_id", None):
                        sess.agent = new_agent
                        engine.agent = new_agent
            except Exception:
                pass
            reg = getattr(engine, "registry", None)
            if reg is not None and hasattr(reg, "register"):
                try:
                    reg.register(
                        bash_mod.tool(
                            max_lines=self.cfg.tool_output_max_lines,
                            max_bytes=self.cfg.tool_output_max_bytes,
                            default_timeout=self.cfg.bash_default_timeout,
                            registry=reg,
                        )
                    )
                except Exception:
                    pass
            try:
                engine.mark_rotation_dirty()
            except Exception:
                pass
        try:
            self._update_header()
        except Exception:
            pass
        self.notify("Settings applied.")

    def _on_settings_done(self, result: Any) -> None:
        self.query_one(InputBar).focus()

    def _on_model_picked(self, model: str | None) -> None:
        if model:
            self._set_model(model)

    def action_focus_input(self) -> None:
        self.query_one(InputBar).focus()

    def action_toggle_thought(self) -> None:
        chat = self._chat_for(self._current_session_id)
        chat.toggle_last_reasoning()
        self.query_one(InputBar).focus()

    def _set_model(self, model: str) -> None:
        self.cfg.model = model
        # new pick follows everywhere NOW: every live engine + its session
        # record, so the next turn in ANY session (old or new) uses it.
        try:
            for engine in list(dict.fromkeys(
                [e for e in self._engines.values() if e is not None]
                + ([self._main_engine] if self._main_engine is not None else [])
            )):
                try:
                    engine.model_id = model
                    engine.mark_rotation_dirty()
                except Exception:
                    pass
                try:
                    sess_id = getattr(engine, "session_id", "") or ""
                    sess = self._sessions.get(sess_id)
                    if sess is not None:
                        sess.model = model
                except Exception:
                    pass
        except Exception:
            pass
        self.notify(f"Model set to opencode/{model}")
        self._update_header()

    def on_rotation_lock_toggled(self, event: RotationLockToggled) -> None:
        """Clicking the model dot in the meta row pins/unpins the selected model."""
        engine = self._active_engine()
        engine.rotation_locked = not engine.rotation_locked
        self.cfg.rotation_lock = engine.rotation_locked
        if engine.rotation_locked:
            self.notify(
                f"Rotation locked — staying on {engine.model_id} "
                "(rate limits/hard errors will surface, not switch)"
            )
        else:
            self.notify("Rotation unlocked — will fail over to backup lanes on errors")
        try:
            from ..config import save_config

            save_config(self.cfg)
        except Exception:
            pass
        self._update_header()

    def _set_agent(self, agent: str) -> None:
        engine = self._active_engine()
        if engine.agent == agent:
            self.notify(f"Agent: {agent}")
            self._update_header()
            return
        engine.agent = agent
        # The PermissionEngine is built from the agent's permission defaults at
        # construction time (build vs plan differ: plan force-denies
        # bash/write/edit/apply_patch). Switching agents mid-session must
        # rebuild it, otherwise those deny rules never take effect and a plan
        # agent can still execute a mutating tool call the model emits (e.g. a
        # bash command). Sub-agents share this same engine (spawn passes
        # permission_engine=self.permission), so one rebuild covers them too.
        from ..permission import PermissionEngine, merge_permissions

        engine.permission = PermissionEngine.from_config(
            merge_permissions(self.cfg.permission, agent),
            mode=getattr(engine.permission, "mode", "auto"),
        )
        engine.permission.ask_callback = self._permission_ask
        sess = self._sessions.get(self._current_session_id)
        if sess is not None:
            sess.agent = agent
        self.notify(f"Agent: {agent}")
        self._update_header()

    def _update_header(self) -> None:
        status = self.query_one(StatusBar)
        # The engine may not be warmed yet (first paint happens before the
        # background engine build finishes) — fall back to cfg values rather
        # than forcing the ~0.4s import on the UI thread mid-mount.
        engine = self._engines.get(self._current_session_id) or self._main_engine
        if engine is None:
            header = {
                "agent": self.cfg.default_agent or "build",
                "model": self.cfg.model,
                "provider": self.cfg.provider,
                "permission_mode": "auto",
                "rotation_locked": False,
                "reasoning_effort": getattr(self.cfg, "reasoning_effort", "") or "",
            }
        else:
            header = {
                "agent": engine.agent,
                # reflect the model/provider that actually answered: rotation can
                # fail over to a backup lane (e.g. deepseek -> nemotron) while
                # cfg.model keeps the user's configured base model.
                "model": getattr(engine, "model_id", "") or self.cfg.model,
                "provider": getattr(engine, "provider_id", "") or self.cfg.provider,
                "permission_mode": engine.permission.mode,
                "rotation_locked": getattr(engine, "rotation_locked", False),
                "reasoning_effort": getattr(self.cfg, "reasoning_effort", "") or "",
            }
        status.set_header(**header)
        try:
            bar = self.query_one(InputBar)
        except Exception:
            return
        if hasattr(bar, "set_header"):
            bar.set_header(**header)


def run_tui(cfg: Config | None = None, directory: Path | None = None) -> None:
    import os
    import signal

    # The engine chain (agent.loop, tools, commands) and the provider internals
    # (zen model list, OpenAI-compat SSE layer) are ~0.6s of lazy imports that
    # aren't needed until the first Enter. Warm them on a background thread NOW
    # so they overlap app.run()'s one-time compose/layout/first-paint work and
    # the first prompt responds immediately instead of waiting for the chain.
    # (This runs from run_tui, not at module import, so it can't race the app's
    # own imports on the main thread.)
    threading.Thread(target=_prewarm_heavy_deps, daemon=True).start()

    app = OpenCodeTUI(cfg=cfg, directory=directory)

    def _close_save_all(signum: int, frame: Any) -> None:
        """Termux-close / kill save (SIGTERM, SIGHUP).

        ``on_exit_app``/``on_unmount`` only run on a graceful exit. When the
        user closes Termux the process gets a signal instead, so save every
        live conversation synchronously here (best effort), then re-raise the
        default signal so shutdown stays immediate.
        """
        try:
            app._save_all_live_sessions()
        except Exception:
            pass
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    try:
        signal.signal(signal.SIGTERM, _close_save_all)
    except (AttributeError, ValueError):  # pragma: no cover - non-unix
        pass
    try:
        signal.signal(signal.SIGHUP, _close_save_all)
    except (AttributeError, ValueError):  # pragma: no cover - non-unix
        pass
    try:
        app.run()
    finally:
        # Release MCP server processes (and any other engine resources) so a
        # server started for this session isn't left dangling after exit.
        engine = getattr(app, "_main_engine", None)
        if engine is not None:
            close = getattr(engine, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # pragma: no cover - best effort on exit
                    pass


if __name__ == "__main__":
    run_tui()
