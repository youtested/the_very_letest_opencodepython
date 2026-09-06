"""find_symbols tool: real code navigation (definitions, callers, deps).

Like grep but with a brain: the repo is indexed once (Python via a true ``ast``
walk; other languages via per-language rules or universal-ctags when
installed), cached on disk, and refreshed incrementally. The model goes
straight to a definition and its call sites instead of grepping and re-reading
whole files.

Query verbs (first word decides the question):
    def NAME        -> where is NAME defined (+ signature, kind, container)
    callers NAME    -> every site that CALLS NAME
    refs NAME       -> every site that uses NAME
    deps FILE       -> what does this file import (local vs external)
    imports MOD     -> which files import this module/file
    symbols FILE    -> every definition inside a file
A bare name falls back to `def`.
"""

from __future__ import annotations

from pathlib import Path

from ..index.engine import MAX_RESULTS as _DEFAULT_LIMIT
from ..index.engine import query as _query
from .registry import Tool, schema_with


def tool() -> Tool:
    description = """- Real code navigation tool: indexes the repo and answers structural questions
- Answers "where is X defined?", "who calls X?", "what does this file depend on?"
- Python gets an exact AST index (definitions + call sites + imports); other languages get smart per-language rules, upgraded to exact definitions automatically when universal-ctags is installed
- The index is cached on disk and refreshed incrementally, so repeat queries are fast
- Query verbs (first word of `query` decides the question):
  - "def <name>" - where is <name> defined (signature, kind, enclosing class)
  - "callers <name>" - every place that calls/invokes <name>
  - "refs <name>" - every place that uses <name> (calls + reads)
  - "deps <path|module>" - what a file imports (local targets resolved)
  - "imports <module|name>" - which files import it
  - "symbols <path>" - all definitions in one file
- A bare word is treated as "def"
- Use BEFORE editing: jump to the definition and check the callers instead of re-reading whole files
- For plain text search (regex over contents) use grep; for filenames use glob"""

    def run(input: dict) -> dict:
        q = str(input.get("query") or "").strip()
        if not q:
            return {
                "output": (
                    "Empty query. Try: 'def run', 'callers _atomic_write', "
                    "'deps tools/edit.py', 'imports write', 'symbols main.py'."
                ),
                "error": True,
            }
        root = input.get("root")
        kind = str(input.get("kind") or "")
        try:
            limit = int(input.get("limit") or _DEFAULT_LIMIT)
        except (TypeError, ValueError):
            limit = _DEFAULT_LIMIT
        ignore = input.get("ignore")
        if isinstance(ignore, str):
            ignore = [ignore]
        force = bool(input.get("fresh", False))
        return _query(
            q,
            root=Path(root) if root else None,
            kind=kind,
            limit=max(1, min(limit, 200)),
            ignore_extra=[str(x) for x in (ignore or [])],
            force=force,
        )

    return Tool(
        name="find_symbols",
        description=description,
        parameters=schema_with(
            {
                "query": {
                    "type": "string",
                    "description": (
                        'What to find, with an optional leading verb: '
                        '"def run", "callers _atomic_write", "refs Registry", '
                        '"deps open/opencode_py/tools/edit.py", '
                        '"imports opencode_py.tools.write", "symbols bash.py".'
                    ),
                },
                "root": {
                    "type": "string",
                    "description": (
                        "Project root to index (default: the current worktree)."
                    ),
                    "optional": True,
                },
                "kind": {
                    "type": "string",
                    "description": (
                        'Optional definition-kind filter for "def" queries: '
                        "function, method, class, variable, constant."
                    ),
                    "optional": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to show (default 60).",
                    "optional": True,
                },
                "ignore": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra directory names to skip while indexing.",
                    "optional": True,
                },
                "fresh": {
                    "type": "boolean",
                    "description": "Force a full rebuild of the index (default false).",
                    "optional": True,
                },
            },
            ["query"],
        ),
        run=run,
        permission="find_symbols",
    )