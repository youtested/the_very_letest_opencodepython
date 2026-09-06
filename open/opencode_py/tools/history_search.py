"""history_search tool: total recall over past session transcripts.

Sessions are already persisted as JSON under
``~/.local/share/opencode_py/sessions/<id>.json`` — but until now the model
could not see anything except the CURRENT conversation. This tool opens that
archive: "how did we fix the SSE timeout bug last month?" becomes one query
that returns the exact messages, files, and session ids.

Built directly on the session module's battle-tested plumbing (list_sessions
with its index + cache + crash-recovery), so no second source of truth.

Actions:
- search (default): all-terms AND match across every saved transcript,
  ranked by hit count then recency, with a snippet around each hit.
- list: recent sessions with ids/titles/dates.
- read: compact transcript of ONE session (id prefix match).
"""

from __future__ import annotations

import re
import time
from typing import Any

from .registry import Tool, schema_with

MAX_SESSIONS_SCANNED = 0  # 0 = scan the whole archive (the documented promise);
                          # index-backed listing keeps this cheap
MAX_BODY_BYTES = 8 * 1024 * 1024   # skip pathological transcripts
SNIPPET_RADIUS = 160               # chars of context either side of a hit
READ_MSG_CHARS = 400               # per-message cap for action=read
MAX_RESULTS = 10


def _msg_text(msg: dict[str, Any]) -> str:
    """Flatten one message to searchable text (str or OpenAI parts)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if "text" in p:
                    parts.append(str(p["text"]))
                elif "name" in p:
                    parts.append(f"{p.get('name', '')}({p.get('arguments', '')})")
        return "\n".join(parts)
    return "" if content is None else str(content)


def _transcript_text(messages: list[dict[str, Any]], role_filter: str = "") -> str:
    out = []
    for msg in messages or []:
        role = str(msg.get("role") or "")
        if role == "system":
            continue
        if role_filter and role != role_filter:
            continue
        text = _msg_text(msg)
        if text.strip():
            out.append(text)
    return "\n".join(out)


def _fmt_date(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    except (OSError, ValueError, OverflowError):
        return "?"


def _snippet(text: str, term: str) -> str:
    low, tlow = text.lower(), term.lower()
    idx = low.find(tlow)
    if idx < 0:
        return text[:SNIPPET_RADIUS * 2]
    start = max(0, idx - SNIPPET_RADIUS)
    end = min(len(text), idx + len(term) + SNIPPET_RADIUS)
    piece = text[start:end].replace("\n", " ")
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{piece}{suffix}"


def _parse_terms(query: str) -> list[str]:
    quoted = re.findall(r'"([^"]+)"', query)
    rest = re.sub(r'"[^"]+"', " ", query)
    terms = [q.strip() for q in quoted if q.strip()]
    terms += [w for w in rest.split() if w]
    seen, uniq = set(), []
    for t in terms:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq[:6]


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def _action_search(query: str, limit: int) -> dict:
    from ..session import load_session, list_sessions

    terms = _parse_terms(query)
    if not terms:
        return {"output": "Empty search. Give a word or \"a quoted phrase\".", "error": True}
    lowered = [t.lower() for t in terms]

    sessions = list_sessions()
    hits: list[dict[str, Any]] = []
    skipped_big = 0
    for meta in sessions:
        try:
            body_path = meta.path
            if body_path.stat().st_size > MAX_BODY_BYTES:
                skipped_big += 1
                continue
        except OSError:
            continue
        sess = load_session(meta.id)
        if sess is None:
            continue
        full = _transcript_text(sess.messages)
        low = full.lower()
        if not all(t in low for t in lowered):
            continue
        score = sum(low.count(t) for t in lowered)
        # first term's first hit drives the snippet
        snippet = _snippet(full, terms[0])
        title = sess.title or (sess.messages and next(
            (_msg_text(m)[:60] for m in sess.messages
             if m.get("role") == "user" and _msg_text(m).strip()), "")
        )
        hits.append({
            "id": sess.id,
            "title": title,
            "date": _fmt_date(sess.created),
            "created": sess.created,
            "score": score,
            "messages": len(sess.messages),
            "snippet": snippet,
        })
    if not hits:
        note = f" ({skipped_big} oversized transcripts skipped)" if skipped_big else ""
        return {"output": f"No past sessions match {terms}.{note}",
                "metadata": {"terms": terms, "matches": 0, "items": []}}

    # most relevant first; recency breaks ties
    hits.sort(key=lambda h: (-h["score"], -h["created"]))
    shown = hits[: max(1, int(limit or MAX_RESULTS))]
    lines = [f"{len(hits)} matching session(s):"]
    for i, h in enumerate(shown, 1):
        lines.append(
            f"\n{i}. [{h['date']}] {(h['title'] or '(untitled)')[:70]} "
            f"(hits: {h['score']}, id {h['id'][:8]}…)"
        )
        lines.append(f"   …{h['snippet']}…" if not h["snippet"].startswith("…") else f"   {h['snippet']}")
    lines.append("\nOpen one with action=read id=<id>.")
    return {"output": "\n".join(lines),
            "metadata": {"terms": terms, "matches": len(hits), "items": shown}}


def _resolve_session(prefix: str):
    from ..session import load_session, list_sessions

    prefix = (prefix or "").strip()
    sessions = list_sessions()
    exact = [s for s in sessions if s.id == prefix]
    if exact:
        return load_session(exact[0].id)
    part = [s for s in sessions if s.id.startswith(prefix)]
    if len(part) == 1:
        return load_session(part[0].id)
    if len(part) > 1:
        return None
    # fall back to title match
    tl = prefix.lower()
    titled = [s for s in sessions if tl and tl in (s.title or "").lower()]
    if len(titled) == 1:
        return load_session(titled[0].id)
    return None


def _action_read(prefix: str, limit_msgs: int) -> dict:
    from ..session import load_session, list_sessions

    sess = None
    sessions = list_sessions()
    p = (prefix or "").strip()
    exact = [s for s in sessions if s.id == p]
    part = [s for s in sessions if p and s.id.startswith(p)]
    if exact:
        sess = load_session(exact[0].id)
    elif len(part) == 1:
        sess = load_session(part[0].id)
    else:
        tl = p.lower()
        titled = [s for s in sessions if tl and tl in (s.title or "").lower()]
        if len(titled) == 1:
            sess = load_session(titled[0].id)
    if sess is None:
        known = ", ".join(s.id[:8] for s in sessions[:5]) or "(none)"
        return {"output": f"No unique session matches {prefix!r}. Recent ids: {known}…",
                "error": True}
    head = [
        f"[{_fmt_date(sess.created)}] {sess.title or '(untitled)'} "
        f"— {len(sess.messages)} messages, id {sess.id}"
    ]
    cap = max(10, min(int(limit_msgs or 80), 400))
    lines = list(head)
    count = 0
    for msg in sess.messages:
        role = str(msg.get("role") or "?")
        if role == "system":
            continue
        text = _msg_text(msg).strip()
        if not text:
            continue
        if len(text) > READ_MSG_CHARS:
            text = text[:READ_MSG_CHARS] + " …"
        lines.append(f"\n[{role}] {text}")
        count += 1
        if count >= cap:
            lines.append(f"\n(… truncated after {cap} messages; use action=read again with a larger message_limit)")
            break
    return {"output": "\n".join(lines),
            "metadata": {"session_id": sess.id, "shown_messages": count}}


def _action_list(limit: int) -> dict:
    from ..session import list_sessions

    sessions = list_sessions()
    if not sessions:
        return {"output": "No saved sessions yet."}
    shown = sessions[: max(1, min(int(limit or 15), 100))]
    lines = [f"{len(sessions)} saved session(s), most recent first:"]
    for s in shown:
        if getattr(s, "messages", None):
            n: Any = len(s.messages)
        elif getattr(s, "has_messages", False):
            n = "?"  # index-only listing: body not loaded, count unknown
        else:
            n = 0
        title = (s.title or "(untitled)")[:56]
        lines.append(f"  {_fmt_date(s.created)}  {s.id[:8]}…  {n:>4} msgs  {title}")
    return {"output": "\n".join(lines),
            "metadata": {"total": len(sessions), "items": [
                {"id": s.id, "title": s.title, "created": s.created} for s in shown]}}


_ACTIONS = ("search", "list", "read")


def tool() -> Tool:
    description = """Searches ALL past conversation transcripts (the session archive on this device).

The model today only sees the current chat; this is its long-term memory.
Use it before redoing work: "how did we fix X?", "what did we decide about Y?",
"which session touched file Z?".

Actions (via `action`):
- search (default): find sessions containing ALL given words (use quotes for
  phrases). Ranked by relevance, returns snippets + session ids.
- list: recent sessions with dates/ids/titles/message counts.
- read: dump one session as a compact transcript (match by id, id prefix, or
  unique title substring).

Tips:
- Search terms are plain substrings, case-insensitive; multiple words must
  ALL appear somewhere in the session (not necessarily together).
- After finding the right session, action=read gives you the full context."""

    def run(input: dict) -> dict:
        action = str(input.get("action") or "search").strip().lower()
        if action not in _ACTIONS:
            return {
                "output": f"Unknown action {action!r} (want one of {', '.join(_ACTIONS)}).",
                "error": True,
            }
        try:
            limit = int(input.get("limit") or 0)
        except (TypeError, ValueError):
            limit = 0
        if action == "list":
            return _action_list(limit or 15)
        if action == "read":
            try:
                msg_cap = int(input.get("message_limit") or 80)
            except (TypeError, ValueError):
                msg_cap = 80
            return _action_read(str(input.get("id") or ""), msg_cap)
        return _action_search(str(input.get("query") or ""), limit or MAX_RESULTS)

    return Tool(
        name="history_search",
        description=description,
        parameters=schema_with(
            {
                "action": {
                    "type": "string",
                    "enum": list(_ACTIONS),
                    "description": "What to do (default search).",
                    "optional": True,
                },
                "query": {
                    "type": "string",
                    "description": 'Words to find (all must appear); "quoted phrase" supported.',
                    "optional": True,
                },
                "id": {
                    "type": "string",
                    "description": "Session id / prefix / title substring (for action=read).",
                    "optional": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results/list rows.",
                    "optional": True,
                },
                "message_limit": {
                    "type": "integer",
                    "description": "Max messages dumped per read (default 80).",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="history_search",
    )