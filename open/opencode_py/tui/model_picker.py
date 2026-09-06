"""Model picker screen (/models + Settings): live, grouped, auto-refreshing.

Mirrors opencode's model picker (dialog-model.tsx): opencode provider first,
Free models first within each provider (sorted by title), deprecated filtered,
favorites/recents would sit on top. Only providers that actually respond are
shown — dead lanes (no key / fetch failed / health probe failed) are hidden
instead of showing placeholder rows.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListView, ListItem, Static

from ..providers import (
    FREE_PROVIDERS,
    PAID_PROVIDERS,
    fetch_zen_models,
    fetch_openrouter_models,
    fetch_live_models,
    sort_model_options,
)
from ..question import QuestionInfo
from .question_dialog import QuestionDialog

REFRESH_SECONDS = 60

# (provider id, display name) — free providers first, paid after.
FREE_SECTION: list[tuple[str, str]] = [
    ("opencode", "OpenCode Zen"),
    ("openrouter", "OpenRouter"),
    ("groq", "Groq"),
    ("cerebras", "Cerebras"),
    ("google", "Google AI Studio"),
    ("nvidia", "NVIDIA NIM"),
    ("mistral", "Mistral"),
    ("github", "GitHub Models"),
    ("sambanova", "SambaNova"),
    ("togetherai", "Together"),
    ("ollama", "Ollama (local)"),
]

PAID_SECTION: list[tuple[str, str]] = [
    ("anthropic", "Anthropic Claude"),
    ("openai", "OpenAI"),
    ("deepseek", "DeepSeek"),
    ("xai", "xAI"),
    ("deepinfra", "DeepInfra"),
]

SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("free", FREE_SECTION),
    ("paid", PAID_SECTION),
]

# curated fallback when a paid provider has no key / the live list is down.
DEFAULT_PAID_MODELS: dict[str, list[str]] = {
    "anthropic": ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"],
    "openai": ["gpt-4o", "gpt-4o-mini", "o3-mini"],
    "deepseek": ["deepseek-chat"],
    "xai": ["grok-2-latest"],
    "deepinfra": ["meta-llama/Meta-Llama-3.3-70B-Instruct"],
}

_MODEL_PICKER_CSS = """
ModelPicker {
    background: $background;
}
#model-picker {
    width: 100%;
    height: 100%;
    layout: vertical;
    padding: 1 2;
}
#models-header {
    height: auto;
    align-horizontal: right;
    margin-bottom: 1;
}
.screen-title {
    height: auto;
    width: 1fr;
    color: $text;
    text-style: bold;
}
.esc-hint {
    height: auto;
    color: $text-muted;
}
#models-search {
    height: 1;
    border: none;
    padding: 0 1;
    background: transparent;
    color: $text-muted;
    margin-bottom: 1;
}
#models-search:focus {
    border: none;
    background: $panel;
    background-tint: transparent;
}
#models-search > .input--cursor {
    background: $primary;
    color: $background;
    text-style: bold;
}
#models-search > .input--placeholder {
    color: $text-muted;
}
#models-status {
    height: auto;
    margin-bottom: 1;
    color: $text-muted;
}
#models-list {
    height: 1fr;
    border: none;
    background: $background;
}
.group-header {
    height: auto;
    padding: 1 0 0 1;
    color: $accent;
    text-style: bold;
}
.zen-sub-group {
    height: auto;
    padding: 0 0 0 2;
    color: $secondary;
    text-style: bold;
}
.model-item {
    height: auto;
    padding: 0 0 0 3;
    color: $text;
}
.model-item .free-tag {
    color: $success;
}
.model-item .current-mark {
    color: $secondary;
}
#models-actions {
    height: auto;
    padding-top: 1;
    align-horizontal: right;
}
#models-actions Button {
    margin-left: 1;
}
"""


class ModelsNav(Message):
    """The search input wants the list to move/select (mirrors opentui, where
    the filter input drives the selection while it keeps focus)."""

    def __init__(self, action: str) -> None:
        super().__init__()
        self.action = action


class _ModelsInput(Input):
    """Search box whose Up/Down/Enter/Escape drive the list instead of being
    consumed by the Input itself (Enter would otherwise just "submit")."""

    BINDINGS = [
        Binding("up", "nav_up", show=False),
        Binding("down", "nav_down", show=False),
        Binding("enter", "nav_select", "Select", show=False),
        Binding("escape", "nav_close", "Close", show=False),
    ]

    def action_nav_up(self) -> None:
        self.post_message(ModelsNav("up"))

    def action_nav_down(self) -> None:
        self.post_message(ModelsNav("down"))

    def action_nav_select(self) -> None:
        self.post_message(ModelsNav("select"))

    def action_nav_close(self) -> None:
        self.post_message(ModelsNav("close"))


class ModelPicker(ModalScreen[str | None]):
    """Full-screen model list; Enter selects, Esc dismisses, R refreshes."""

    CSS = _MODEL_PICKER_CSS

    BINDINGS = [
        Binding("r", "refresh_models", "Refresh"),
        Binding("f", "toggle_favorite", "Favorite"),
        Binding("escape", "dismiss_pop", "Close"),
    ]

    def __init__(
        self,
        current: str = "",
        on_select: Callable[[str], None] | None = None,
        cfg: Any = None,
        auth: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.current = current
        self.on_select = on_select
        self.cfg = cfg
        self.auth = auth
        self.models: dict[str, list[dict]] = {}
        self._item_lookup: list[dict] = []
        self._fetching = False
        self._timer: Any = None
        self._search_timer: Any = None
        # Custom providers registered under config `providers.<id>`: they get
        # their own section at the end of the list, fetched live from each
        # provider's /models endpoint when an API key is present.
        self.custom_providers: dict[str, dict] = dict((cfg.providers or {}).items()) if cfg else {}
        # Guard against re-entrant selection: programmatic `lv.index = …` during
        # a refresh/search fires a *spurious* ListView.Selected that must not be
        # treated as the user picking a model, and once dismissed, queued events
        # must not call dismiss() again (pop_screen would raise on an empty
        # stack).
        self._rebuilding = False
        self._dismissed = False

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker"):
            with Horizontal(id="models-header"):
                yield Label("Models", classes="screen-title")
                yield Label("esc", classes="esc-hint")
            yield _ModelsInput(placeholder="Search models…", id="models-search")
            yield Static("Loading models...", id="models-status")
            yield ListView(id="models-list")
            with Horizontal(id="models-actions"):
                yield Button("Refresh", id="models-refresh", variant="default")
                yield Button("Close", id="models-close", variant="primary")

    def on_mount(self) -> None:
        self.set_loading()
        self._start_worker()
        self._timer = self.set_interval(REFRESH_SECONDS, self._periodic_refresh)

    def on_unmount(self) -> None:
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:
                pass
            self._timer = None
        timer = getattr(self, "_search_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
            self._search_timer = None

    # -- fetching ----------------------------------------------------------
    def set_loading(self) -> None:
        if not self.is_attached:
            return
        try:
            self.query_one("#models-status", Static).update(
                f"Fetching model lists from providers... (auto-refresh every {REFRESH_SECONDS}s)"
            )
        except Exception:
            pass

    def _start_worker(self) -> None:
        if self._fetching:
            return
        self._fetching = True
        self.set_loading()
        self.run_worker(self._fetch_models, thread=True)

    def _periodic_refresh(self) -> None:
        self._start_worker()

    def _fetch_models(self) -> None:
        # This runs on a worker thread: pull the models.dev catalog
        # synchronously first (official fetchAndWrite parity) so upstream adds
        # (new free models) and removals land in THIS view, not a minute later.
        try:
            from ..providers import refresh_catalog_sync

            refresh_catalog_sync()
        except Exception:
            pass
        pids = [pid for _, providers in SECTIONS for pid, _ in providers]
        pids += list(self.custom_providers)
        per_provider: dict[str, list[dict]] = {}
        try:
            with ThreadPoolExecutor(max_workers=6) as ex:
                futures = {ex.submit(self._fetch_provider_models, pid): pid for pid in pids}
                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        per_provider[pid] = future.result() or []
                    except Exception:
                        per_provider[pid] = []
        finally:
            self._fetching = False
        self.app.call_from_thread(self.populate, per_provider)

    def _fetch_provider_models(self, pid: str) -> list[dict]:
        if pid == "opencode":
            # fetch_zen_models() lists every active catalog entry like the
            # official client (responding free first, none hidden).
            return fetch_zen_models()
        if pid == "openrouter":
            # Live free list only; [] when down so dead lanes stay hidden.
            return fetch_openrouter_models()
        if pid == "ollama":
            # Local, no key: show only when the daemon actually responds.
            try:
                live = fetch_live_models("ollama", "ollama", "http://localhost:11434/v1")
            except Exception:
                live = []
            if live:
                for m in live:
                    m["free"] = False
                    m.setdefault("release_date", "")
                return live
            return _probe_ollama_daemon()
        custom = self.custom_providers.get(pid)
        if custom:
            key = self.auth.get(pid) if self.auth else None
            models = fetch_live_models(pid, key, custom.get("base_url"))
            if models:
                return models
            # live fetch failed (no key / endpoint down): show the configured
            # model so the provider's section is never empty
            cur = (self.cfg.model or "").split("/")[-1] if self.cfg else ""
            return [{"id": cur, "name": cur, "context": 0, "free": False}] if cur else []
        meta = FREE_PROVIDERS.get(pid) or PAID_PROVIDERS.get(pid) or {}
        key = self.auth.get(pid) if self.auth else None
        models = (
            fetch_live_models(pid, key, meta.get("base_url"), meta.get("api_kind", "openai"))
            if meta
            else []
        )
        if models:
            # Official badges Free only for opencode (cost.input==0).
            # Other providers' models are paid/BYOK — no FREE tag.
            for m in models:
                m["free"] = False
                m.setdefault("release_date", "")
            return models
        # No live data / no key -> hide the provider entirely instead of
        # showing a placeholder that didn't respond (official shows only
        # available catalog entries).
        return []

    # -- display -----------------------------------------------------------
    def _query(self) -> str:
        if not self.is_attached:
            return ""
        try:
            return (self.query_one("#models-search", Input).value or "").strip().lower()
        except Exception:
            return ""

    def _set_status(self, text: str) -> None:
        if not self.is_attached:
            return
        try:
            self.query_one("#models-status", Static).update(text)
        except Exception:
            pass

    def populate(self, per_provider: dict[str, list[dict]]) -> None:
        # The fetch worker may complete after the screen was dismissed (Esc /
        # Close / model picked). Guard the widget lookups so a pruned screen
        # doesn't raise NoMatches and crash the whole app.
        if not self.is_attached:
            return
        self.models = per_provider
        self._populate_list()

    def _populate_list(self) -> None:
        if not self.is_attached or self._dismissed:
            return
        lv = self.query_one("#models-list", ListView)
        # Rebuilding fires spurious ListView.Selected events when lv.index is
        # set below; ignore them so refresh/search don't auto-dismiss.
        self._rebuilding = True
        try:
            lv.clear()
            self._item_lookup = []
            q = self._query()
            total_free, total_paid, shown, add_row = self._populate_rows(lv, q)
            if shown == 0 and not add_row:
                # Official shows "No results found" empty view; when fully
                # offline show popular providers to connect instead of nothing.
                if not q and self._maybe_show_popular(lv):
                    self._set_status("No live models — pick a provider to connect")
                    return
                self._set_status("No models match your search.")
                return
        finally:
            self._rebuilding = False

        # highlight the current model when present, else the first real row.
        # The `__custom__`/`__connect__` sentinel rows carry model == "" so
        # they must be excluded (an empty `current` must match nothing).
        current_hit = [
            e["row"]
            for e in self._item_lookup
            if e["provider"] not in ("__custom__", "__connect__")
            and self.current
            and (
                f"{e['provider']}/{e['model']}" == self.current
                or e["model"] == self.current.split("/")[-1]
            )
        ]
        self._rebuilding = True
        try:
            real_rows = [e["row"] for e in self._item_lookup if e["provider"] not in ("__custom__", "__connect__")]
            if current_hit:
                lv.index = current_hit[0]
            elif real_rows:
                lv.index = real_rows[0]
            elif self._item_lookup:
                lv.index = self._item_lookup[0]["row"]
            else:
                return
        finally:
            self._rebuilding = False

        def _fmt_count():
            if q:
                return f"{shown} model{'s' if shown != 1 else ''} — filtered by '{q}'"
            return f"{total_free} free, {total_paid} paid"

        self._set_status(
            f"{_fmt_count()} — updated {time.strftime('%H:%M:%S')} — Enter select · F favorite · R refresh"
        )

    def _populate_rows(self, lv: ListView, q: str) -> tuple[int, int, int, bool]:
        """Official dialog-model.tsx port.

        - Providers: opencode first, then by display name (not free/paid blocks).
        - Within provider: sort_model_options (Free first, release_date desc, title).
        - Favorites + Recent sections on top when no query.
        - Query: flat fuzzy list (no headers), footer shows provider name.
        Returns (total_free, total_paid, shown, add_row).
        """
        total_free = 0
        total_paid = 0
        shown = 0
        row = 0

        def _add_model_row(pid: str, m: dict, flat: bool = False) -> None:
            nonlocal row, shown, total_free, total_paid
            idx = f"{pid}/{m['id']}"
            self._item_lookup.append({"row": row, "provider": pid, "model": m["id"]})
            if pid == "opencode" and m.get("free"):
                total_free += 1
            else:
                total_paid += 1
            shown += 1
            lv.append(
                ListItem(
                    _model_row_label(
                        idx, m, self.current,
                        provider_name=self._provider_display(pid) if flat else "",
                        favorite=_is_favorite(pid, m["id"]),
                    )
                )
            )
            row += 1

        if q:
            # Flat fuzzy search across all live models (official: fuzzysort on
            # title+category, then sortModelOptions Free/newest/title).
            needle = q.lower()
            candidates: list[dict] = []
            for pid in self._ordered_provider_ids():
                for m in self.models.get(pid) or []:
                    title = str(m.get("name") or m["id"])
                    cat = self._provider_display(pid)
                    if _fuzzy_match(needle, title) or _fuzzy_match(needle, cat) or _fuzzy_match(needle, m["id"]):
                        candidates.append({"_pid": pid, **m, "_title": title, "_cat": cat})
            for pid in sorted(self.custom_providers):
                for m in self.models.get(pid) or []:
                    title = str(m.get("name") or m["id"])
                    if _fuzzy_match(needle, title) or _fuzzy_match(needle, m["id"]) or needle in pid:
                        candidates.append({"_pid": pid, **m, "_title": title, "_cat": pid})
            # Free first, newest release_date, then title (official).
            candidates = sort_model_options(
                [{**c, "name": c["_title"], "title": c["_title"]} for c in candidates]
            )
            for c in candidates:
                _add_model_row(c["_pid"], c, flat=True)
        else:
            # Favorites + Recent on top (official showExtra sections).
            favs = _load_prefs().get("favorite", [])
            recents = _load_prefs().get("recent", [])
            fav_keys = {(f.get("providerID"), f.get("modelID")) for f in favs}
            recent_keys = [(r.get("providerID"), r.get("modelID")) for r in recents]
            shown_keys: set[tuple[str, str]] = set()

            def _find_model(pid: str, mid: str) -> dict | None:
                for m in self.models.get(pid) or []:
                    if m["id"] == mid:
                        return m
                return None

            if favs:
                lv.append(ListItem(Label("  Favorites", classes="group-header")))
                row += 1
                for f in favs:
                    pid, mid = f.get("providerID"), f.get("modelID")
                    m = _find_model(pid, mid) if pid and mid else None
                    if m is None:
                        continue
                    shown_keys.add((pid, mid))
                    _add_model_row(pid, m)
            if recent_keys:
                lv.append(ListItem(Label("  Recent", classes="group-header")))
                row += 1
                for pid, mid in recent_keys:
                    if (pid, mid) in shown_keys or (pid, mid) in fav_keys:
                        continue
                    m = _find_model(pid, mid) if pid and mid else None
                    if m is None:
                        continue
                    shown_keys.add((pid, mid))
                    _add_model_row(pid, m)

            for pid in self._ordered_provider_ids():
                items = self.models.get(pid) or []
                if not items:
                    continue
                ordered = sort_model_options(items)
                # Skip models already shown in Favorites/Recent (official filters them).
                ordered = [m for m in ordered if (pid, m["id"]) not in shown_keys]
                if not ordered:
                    continue
                display = self._provider_display(pid)
                lv.append(ListItem(Label(f"  {display}", classes="group-header")))
                row += 1
                for m in ordered:
                    _add_model_row(pid, m)
        # custom providers from config `providers.<id>` — their own section at
        # the end, so a provider added from the "add custom" row shows up here.
        if not q:
            for pid in sorted(self.custom_providers):
                meta = self.custom_providers[pid]
                items = self.models.get(pid) or []
                if not items:
                    continue
                display = meta.get("name") or pid
                ordered = sort_model_options(items)
                lv.append(ListItem(Label(f"  {display}  (custom)", classes="group-header")))
                row += 1
                for m in ordered:
                    idx = f"{pid}/{m['id']}"
                    self._item_lookup.append({"row": row, "provider": pid, "model": m["id"]})
                    total_paid += 1
                    shown += 1
                    lv.append(ListItem(_model_row_label(idx, m, self.current)))
                    row += 1
        else:
            # custom already merged into flat candidates above; nothing extra.
            pass
        # the "add custom provider" entry lives at the very end of the list.
        add_row = False
        if not q or "custom" in q:
            add_row = True
            lv.append(ListItem(Label("  Custom", classes="group-header")))
            row += 1
            self._item_lookup.append({"row": row, "provider": "__custom__", "model": ""})
            lv.append(
                ListItem(
                    Label("   [bold]＋[/] Add custom provider (URL + API key + model)", classes="model-item")
                )
            )
            row += 1
        return total_free, total_paid, shown, add_row

    def _ordered_provider_ids(self) -> list[str]:
        """Official order: opencode first, then by display name."""
        ids = [pid for _, providers in SECTIONS for pid, _ in providers]
        def _name(pid: str) -> str:
            return self._provider_display(pid).lower()
        head = [pid for pid in ids if pid == "opencode"]
        rest = sorted([pid for pid in ids if pid != "opencode"], key=_name)
        return head + rest

    def _provider_display(self, pid: str) -> str:
        for _, providers in SECTIONS:
            for p, display in providers:
                if p == pid:
                    return display
        meta = self.custom_providers.get(pid)
        if meta:
            return str(meta.get("name") or pid)
        return pid

    def _maybe_show_popular(self, lv: ListView) -> bool:
        """When nothing responds (offline / no keys): show Popular providers to
        connect, like official's popularProviders when !connected."""
        try:
            from ..commands import KEY_HINTS
        except Exception:
            KEY_HINTS = {}
        popular = [pid for pid, _ in FREE_SECTION if pid != "opencode"][:6]
        if not popular:
            return False
        lv.append(ListItem(Label("  Popular providers", classes="group-header")))
        row = len(self._item_lookup) + 1
        for pid in popular:
            display = self._provider_display(pid)
            self._item_lookup.append({"row": row, "provider": "__connect__", "model": pid})
            lv.append(ListItem(Label(f"   {display}  — /connect {pid}", classes="model-item")))
            row += 1
        return True

    def _close(self, result: str | None = None) -> None:
        if self._dismissed:
            return
        self._dismissed = True
        self.dismiss(result)

    # -- events ------------------------------------------------------------
    def on_models_nav(self, event: Any) -> None:
        action = getattr(event, "action", "")
        if action == "up":
            self._move_selection(-1)
        elif action == "down":
            self._move_selection(1)
        elif action == "select":
            self._choose_current()
        elif action == "close":
            self._close(None)

    def _move_selection(self, direction: int) -> None:
        if not self.is_attached or self._dismissed:
            return
        # Every entry in _item_lookup is selectable (models + custom/connect
        # sentinels); headers are never in the lookup so they're skipped.
        rows = sorted({e["row"] for e in self._item_lookup})
        if not rows:
            return
        lv = self.query_one("#models-list", ListView)
        current = lv.index
        if current is None or current not in rows:
            # no selection yet: land on the first/last selectable row
            target = rows[0] if direction > 0 else rows[-1]
        else:
            target = current
            while True:
                target += direction
                if target in rows:
                    break
                if (direction > 0 and target > rows[-1]) or (direction < 0 and target < rows[0]):
                    return
        lv.index = target

    def _choose_current(self) -> None:
        if not self.is_attached or self._dismissed:
            return
        lv = self.query_one("#models-list", ListView)
        self._choose_row(lv.index)

    def _choose_row(self, index: int | None) -> None:
        # The refresh/search rebuild sets `lv.index` programmatically, which
        # fires a ListView.Selected that is NOT a user pick — ignore it. Also
        # once the picker is dismissed, later queued events must be no-ops.
        if self._rebuilding or self._dismissed or not self.is_attached:
            return
        for entry in self._item_lookup:
            if entry["row"] == index:
                if entry["provider"] == "__custom__":
                    self._add_custom_provider()
                    return
                if entry["provider"] == "__connect__":
                    self._connect_provider(entry["model"])
                    return
                choice = f"{entry['provider']}/{entry['model']}"
                _record_recent(entry["provider"], entry["model"])
                if self.on_select:
                    self.on_select(choice)
                self._close(choice)
                return

    def _connect_provider(self, pid: str) -> None:
        """Popular-provider row when offline: guide to /connect + key URL."""
        try:
            from ..commands import KEY_HINTS
            url = KEY_HINTS.get(pid, "")
        except Exception:
            url = ""
        msg = f"Run /connect {pid}" + (f" — key: {url}" if url else "")
        try:
            self.app.notify(msg)
        except Exception:
            pass

    def action_toggle_favorite(self) -> None:
        """F toggles favorite for the highlighted model (official ctrl+f)."""
        if self._dismissed or not self.is_attached:
            return
        try:
            lv = self.query_one("#models-list", ListView)
            idx = lv.index
        except Exception:
            return
        for entry in self._item_lookup:
            if entry["row"] == idx and entry["provider"] not in ("__custom__", "__connect__"):
                _toggle_favorite(entry["provider"], entry["model"])
                try:
                    self.app.notify(f"{'★' if _is_favorite(entry['provider'], entry['model']) else '☆'} {entry['provider']}/{entry['model']}")
                except Exception:
                    pass
                self._populate_list()
                return

    # -- add custom provider ---------------------------------------------
    def _add_custom_provider(self) -> None:
        """Ask for provider id / base URL / API key / model, save them, and
        switch to the new provider. Prompted via the existing QuestionDialog so
        the flow stays consistent with the rest of the TUI."""
        if self._dismissed or not self.is_attached:
            return
        questions = [
            QuestionInfo(
                question="Give this provider a short id (letters, numbers, _ and -), e.g. teamo.",
                header="Provider id",
            ),
            QuestionInfo(
                question="OpenAI-compatible base URL, e.g. https://api.teamorouter.com/v1",
                header="Base URL",
            ),
            QuestionInfo(
                question="API key (sk-...). Saved to auth.json (0600), never to the config file.",
                header="API key",
            ),
            QuestionInfo(
                question="Model id to use by default, e.g. x-preview-f-free",
                header="Model",
            ),
        ]
        self.app.push_screen(QuestionDialog(questions), self._on_custom_done)

    def _on_custom_done(self, result: Any) -> None:
        """Apply a completed add-custom-provider flow (or cancel)."""
        if not result or self._dismissed or not self.is_attached:
            return

        def _answer(i: int) -> str:
            return (result[i][0] if i < len(result) and result[i] else "").strip()

        import re

        pid = re.sub(r"[^a-z0-9_-]+", "", _answer(0).lower())
        base = _answer(1).rstrip("/")
        key = _answer(2)
        model = _answer(3)

        if not pid or not base or not model:
            self.app.notify("Custom provider needs an id, base URL and model.", severity="warning")
            return
        if not key:
            self.app.notify("Custom provider needs an API key.", severity="warning")
            return
        if not base.startswith("http://") and not base.startswith("https://"):
            base = "https://" + base
        from ..providers import FREE_PROVIDERS, PAID_PROVIDERS

        if pid in FREE_PROVIDERS or pid in PAID_PROVIDERS or pid in ("opencode", "ollama"):
            self.app.notify(f"'{pid}' is a built-in provider — pick a different id.", severity="warning")
            return

        if self.auth is not None:
            try:
                self.auth.set(pid, key)
            except Exception as e:
                self.app.notify(f"Failed to save API key: {e}", severity="error")
                return
        if self.cfg is not None:
            display_name = base.split("//")[-1] or pid
            self.cfg.providers[pid] = {"name": display_name, "base_url": base}
            self.custom_providers[pid] = self.cfg.providers[pid]
            self.cfg.provider = pid
            self.cfg.model = model
            try:
                from ..config import save_config

                save_config(self.cfg)
            except Exception as e:
                self.app.notify(f"Failed to save config: {e}", severity="error")
                return
        self.app.notify(f"Added custom provider {pid}/{model}")
        choice = f"{pid}/{model}"
        if self.on_select:
            self.on_select(choice)
        self._close(choice)

    def on_list_view_selected(self, event: Any) -> None:
        if self._rebuilding or self._dismissed:
            return
        index = event.index if event.index is not None else (getattr(event.item, "index", None) or 0)
        self._choose_row(index)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "models-search":
            if self.is_attached and not self._dismissed and self.models:
                # Debounce: rebuilding the whole ListView per keystroke
                # janked fast typists (full teardown + fuzzy re-sort per char).
                # 120ms after the LAST keystroke is indistinguishable.
                old = getattr(self, "_search_timer", None)
                if old is not None:
                    try:
                        old.stop()
                    except Exception:
                        pass
                try:
                    self._search_timer = self.set_timer(0.12, self._debounced_search)
                except Exception:
                    self._populate_list()

    def _debounced_search(self) -> None:
        self._search_timer = None
        if self.is_attached and not self._dismissed and self.models:
            self._populate_list()

    def action_refresh_models(self) -> None:
        if self._dismissed:
            return
        if self._query():
            # re-running the worker would clear the search input's siblings;
            # just re-render against the current data instead
            self._populate_list()
        else:
            self._start_worker()

    def action_dismiss_pop(self) -> None:
        self._close(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "models-close":
            self._close(None)
        elif bid == "models-refresh":
            self.action_refresh_models()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self._close(None)
            event.stop()


def _zen_family(model_id: str) -> str:
    """Upstream vendor behind an OpenCode Zen model, from its id prefix.

    Kept for compat; the official picker no longer clusters Zen by family
    (it sorts Free/newest/title instead).
    """
    import re

    match = re.match(r"^[a-z]+", model_id.lower())
    prefix = match.group(0) if match else model_id
    families = {
        "claude": "Anthropic",
        "gemini": "Google",
        "gpt": "OpenAI",
        "kimi": "Moonshot",
        "grok": "xAI",
        "deepseek": "DeepSeek",
        "glm": "Zhipu AI",
        "minimax": "MiniMax",
        "qwen": "Alibaba",
        "nemotron": "NVIDIA",
        "mimo": "Xiaomi",
        "laguna": "Poolside",
        "north": "Cohere",
        "big": "Other",
        "hy": "Other",
        "ling": "Other",
    }
    return families.get(prefix, "Other")


def _model_row_label(
    idx: str,
    m: dict,
    current: str,
    provider_name: str = "",
    favorite: bool = False,
) -> Label:
    """Build a model row matching official dialog-select Option.

    - current model marked with ●, favorites with ★
    - FREE footer only for opencode free models (official: cost.input==0)
    - flat search rows show the provider name as footer instead.
    """
    name = m.get("name") or m["id"]
    free = bool(m.get("free"))
    marked = idx == current or m["id"] == current
    mark = "[#fab283]●[/] " if marked else "   "
    star = "[#fab283]★[/] " if favorite and not marked else ""
    if provider_name:
        # flat search view: official shows category as footer.
        return Label(f"{mark}{star}{name} [#666666]{provider_name}[/]", classes="model-item")
    free_tag = " [#7fd88f]FREE[/]" if free else ""
    fav_tag = " [#fab283](Favorite)[/]" if favorite else ""
    return Label(f"{mark}{star}{name}{free_tag}{fav_tag}", classes="model-item")


def _fallback_models(pid: str, has_key: bool = False) -> list[dict]:
    """Kept for compat; the picker now hides dead lanes instead of showing
    bundled placeholders (official shows only available catalog entries)."""
    return []


def _probe_ollama_daemon() -> list[dict]:
    """Ollama has no placeholder list: [] when the daemon doesn't respond."""
    return []


def _fuzzy_match(needle: str, haystack: str) -> bool:
    """Tiny fuzzysort-ish match (official uses fuzzysort on title+category).

    Substring matches first; otherwise subsequence (chars in order).
    """
    n = (needle or "").lower()
    h = (haystack or "").lower()
    if not n:
        return True
    if n in h:
        return True
    return _subseq(n, h)


def _subseq(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(any(c == hc for hc in it) for c in needle)


def _prefs_path():
    from pathlib import Path

    from ..globals import Path as GPath

    try:
        GPath.init()
        base = Path(GPath.cache)
    except Exception:
        from pathlib import Path as _P

        base = _P.home() / ".cache" / "opencode_py"
    return base / "model-picker.json"


def _load_prefs() -> dict:
    import json

    try:
        data = json.loads(_prefs_path().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("recent", [])
            data.setdefault("favorite", [])
            return data
    except Exception:
        pass
    return {"recent": [], "favorite": []}


def _save_prefs(prefs: dict) -> None:
    import json

    try:
        p = _prefs_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(prefs), encoding="utf-8")
    except Exception:
        pass


def _record_recent(provider: str, model: str, limit: int = 10) -> None:
    prefs = _load_prefs()
    recents = [r for r in prefs.get("recent", []) if not (r.get("providerID") == provider and r.get("modelID") == model)]
    recents.insert(0, {"providerID": provider, "modelID": model})
    prefs["recent"] = recents[:limit]
    _save_prefs(prefs)


def _is_favorite(provider: str, model: str) -> bool:
    try:
        return any(
            f.get("providerID") == provider and f.get("modelID") == model
            for f in _load_prefs().get("favorite", [])
        )
    except Exception:
        return False


def _toggle_favorite(provider: str, model: str) -> None:
    prefs = _load_prefs()
    favs = list(prefs.get("favorite", []))
    if any(f.get("providerID") == provider and f.get("modelID") == model for f in favs):
        favs = [f for f in favs if not (f.get("providerID") == provider and f.get("modelID") == model)]
    else:
        favs.append({"providerID": provider, "modelID": model})
    prefs["favorite"] = favs
    _save_prefs(prefs)


def _format_context(value: Any) -> str:
    """Format a context size for display, tolerating "128k"/"1m" strings and junk."""
    if value is None:
        return "?"
    if isinstance(value, str):
        s = value.strip().lower()
        mult = 1
        if s.endswith("k"):
            mult, s = 1000, s[:-1]
        elif s.endswith("m"):
            mult, s = 1000000, s[:-1]
        try:
            return f"{int(float(s) * mult):,}"
        except (ValueError, TypeError):
            return value
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return "?"
