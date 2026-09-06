"""read tool: read a file or directory with line numbers + offset/limit.

Smart-read behaviour (mobile-data friendly):
- mode="auto" (default): files over AUTO_OUTLINE_LINES lines return a compact
  structural OUTLINE first (classes/functions/headings + line numbers) so the
  model can target small offset windows instead of swallowing whole files.
- Every delivered window is recorded in tools/context_ledger.py; re-requesting
  the same content returns a short stub, and overlapping windows are trimmed
  to just the NEW lines. Nothing is sent twice unless force_full=true.
"""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path

from .context_ledger import mark_delivered, unseen_ranges
from .registry import Tool, schema_with

MAX_LINES = 2000
MAX_CHARS = 2000
MAX_OUTPUT = 50 * 1024  # 50 KB

# Files longer than this many lines return an outline under mode="auto".
AUTO_OUTLINE_LINES = 150

# Files up to this size are read whole in one syscall and sliced in memory;
# anything larger streams the window so memory stays bounded for huge files.
# Kept small-ish: on a large file a top-of-file read stops after the window,
# which is cheaper than slurping the whole payload just to slice a few lines.
FAST_READ_BYTES = 256 * 1024

IMAGE_EXTENSIONS = {".png", ".jpeg", ".jpg", ".gif", ".webp"}
PDF_EXTENSIONS = {".pdf"}


def _is_binary_sample(sample: bytes) -> bool:
    if not sample:
        return False
    sample = sample[:1024]
    # The fixed window can cut a multibyte UTF-8 char in half (e.g. the box-
    # drawing divider in README.md straddling byte 1024); that truncated tail
    # is "unexpected end of data", not invalid UTF-8. Retry after dropping up
    # to 3 trailing bytes (longest UTF-8 sequence) before declaring binary.
    for trim in range(4):
        try:
            sample[: len(sample) - trim].decode("utf-8")
            break
        except UnicodeDecodeError:
            continue
    else:
        return True
    nonprintable = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return nonprintable / len(sample) > 0.30


def _fuzzy_suggestion(path: Path) -> str | None:
    try:
        candidates = [p for p in path.parent.iterdir() if p.is_file()]
    except OSError:
        return None
    name = path.name
    matches = []
    for p in candidates:
        if p.stem == name or name in p.stem or p.stem in name:
            matches.append(p.name)
    return ", ".join(matches[:3]) or None


def _read_error(path: Path, error: BaseException) -> dict:
    suggestion = _fuzzy_suggestion(path)
    msg = f"Could not read file {path}: {error}"
    if suggestion:
        msg += f"\n\nDid you mean one of these?\n{suggestion}"
    return {"output": msg, "error": True}


def _read_window_lines(
    lines: list[str],
    offset: int,
    limit: int,
) -> tuple[list[str], int | None, bool, bool]:
    """Slice a line list into the requested window, capping the output size.

    Shared by the fast (whole-file) and streaming (large-file) read paths so
    both produce byte-identical results. Returns ``(numbered, total,
    reached_eof, truncated_out)`` where ``total`` is the last line index read
    (None if the window started past EOF) and ``numbered`` holds
    ``f"{lineno}: {content}"`` rows.
    """
    numbered: list[str] = []
    total: int | None = None
    truncated_out = False
    out_chars = 0
    start = max(0, offset - 1)
    limit = max(1, int(limit))
    end = start + limit
    last_existing = min(len(lines), end)
    if last_existing < start + 1:
        # window starts past EOF: mirror the streaming path, which reads every
        # line (total = line count) and reports end-of-file
        total = len(lines)
        return [], total, True, False
    total = last_existing
    for lineno in range(start + 1, last_existing + 1):
        content = lines[lineno - 1].rstrip("\n").rstrip("\r")
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + f"... (line truncated to {MAX_CHARS} chars)"
        numbered.append(f"{lineno}: {content}")
        capped_chars = len(content) + 16
        if out_chars + capped_chars > MAX_OUTPUT:
            truncated_out = True
            break
        out_chars += capped_chars
    # reached EOF when the window consumed the file's last line without the
    # output cap cutting it short
    reached_eof = total >= len(lines) and not truncated_out
    return numbered, total, reached_eof, truncated_out


# Outline reads are capped: definition lines live at the top of files, so
# scanning the head is enough — a 50 MB minified bundle no longer gets
# slurped + ast-parsed whole just to list its (nonexistent) structure.
OUTLINE_MAX_BYTES = 512 * 1024


def _outline(path: Path) -> dict:
    """Structural skeleton of a text file: definition/heading lines with their
    line numbers, NO bodies. A 500 KB file shrinks to a few KB so the model
    can target small offset windows instead of swallowing everything."""
    try:
        with open(path, "rb") as fh:
            data = fh.read(OUTLINE_MAX_BYTES + 1)
    except OSError as e:
        return _read_error(path, e)
    if _is_binary_sample(data):
        return {"output": f"File {path} is a binary file and cannot be read as text.", "error": True}
    head_truncated = len(data) > OUTLINE_MAX_BYTES
    if head_truncated:
        data = data[:OUTLINE_MAX_BYTES]
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    total = len(lines)
    total_suffix = "+" if head_truncated else ""
    entries: list[str] = []

    def _emit(lineno: int, indent: str) -> None:
        raw = lines[lineno - 1].strip()
        if raw:
            entries.append(f"{indent}{lineno}: {raw[:120]}")

    if path.suffix == ".py":
        import ast as _ast

        try:
            tree = _ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            def walk_body(body, indent: str) -> None:
                for node in body:
                    if isinstance(node, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
                        _emit(node.lineno, indent)
                        if isinstance(node, _ast.ClassDef):
                            walk_body(node.body, indent + "    ")
            walk_body(tree.body, "")
    if not entries:
        # fallback heuristics for non-Python text: headings + common defs
        import re as _re

        pat = _re.compile(r"^(#{1,6}\s+\S|\s{0,8}(?:def |class |func |fn |function |async def ))")
        for i, line in enumerate(lines, 1):
            if pat.match(line):
                entries.append(f"{i}: {line.strip()[:120]}")

    if not entries:
        return {
            "output": (
                f"<{path}>…</{path}>\n<type>outline</type>\n"
                f"({total}{total_suffix} lines; no recognizable structure — use mode=\"full\" "
                "or offset/limit windows to read it)"
            ),
            "metadata": {"loaded": [str(path)], "outline": True},
        }
    capped = False
    out_chars = 0
    kept: list[str] = []
    for entry in entries:
        cost = len(entry) + 1
        if out_chars + cost > MAX_OUTPUT:
            capped = True
            break
        kept.append(entry)
        out_chars += cost
    body = "\n".join(kept)
    footer = f"(outline: {len(kept)}/{len(entries)} definitions, {total}{total_suffix} lines total"
    if capped:
        footer += ", outline truncated"
    footer += ' — read regions with offset/limit, or mode="full")'
    return {
        "output": f"<{path}>…</{path}>\n<type>outline</type>\n<content>\n{body}\n</content>\n{footer}",
        "metadata": {"loaded": [str(path)], "outline": True},
    }


def _finalize_file_window(
    path: Path,
    st,
    numbered: list[str],
    start_line: int,
    force_full: bool,
    *,
    reached_eof: bool,
    total_all: int | None,
    truncated_out: bool,
) -> dict:
    """Shared tail for both read paths: dedup-gate the built window against
    the context ledger, trim already-delivered lines, mark what actually
    goes out, and assemble the legacy output format."""
    end_line = start_line + len(numbered) - 1
    by_lineno: dict[int, str] = {}
    for ln in numbered:
        try:
            by_lineno[int(ln.split(":", 1)[0])] = ln
        except ValueError:
            continue

    omitted = 0
    parts: list[str] = []
    new_ranges: list[tuple[int, int]] = []
    if not numbered:
        unseen: list[tuple[int, int]] = []
    elif force_full:
        unseen = [(start_line, end_line)]
    else:
        unseen = unseen_ranges(str(path), st.st_mtime_ns, st.st_size, start_line, end_line)

    if numbered and not unseen and not force_full:
        return {
            "output": (
                f"<{path}>…</{path}>\n<type>file</type>\n"
                f"(lines {start_line}-{end_line}: identical content was already delivered "
                "to you earlier this session — nothing new here, the file is unchanged on disk)\n"
                '(Re-request with force_full=true only if you genuinely need it again.)'
            ),
            "metadata": {"loaded": [str(path)], "dedup_stub": True},
        }

    cursor = start_line
    last_real = start_line - 1
    for a, b in unseen:
        if a > cursor:
            skipped = a - cursor
            parts.append(f"… lines {cursor}-{a - 1}: unchanged, shown earlier ({skipped} lines omitted)")
            omitted += skipped
        for lineno in range(a, min(b, end_line) + 1):
            ln = by_lineno.get(lineno)
            if ln is None:
                continue
            parts.append(ln)
            last_real = lineno
        new_ranges.append((a, b))
        cursor = b + 1
    if cursor <= end_line:
        skipped = end_line - cursor + 1
        parts.append(f"… lines {cursor}-{end_line}: unchanged, shown earlier ({skipped} lines omitted)")
        omitted += skipped

    # honour MAX_OUTPUT AFTER dedup-trim; drop any new_range whose lines were
    # cut so the ledger never records content that did not actually go out.
    body_parts: list[str] = []
    out_chars = 0
    kept_ranges: list[tuple[int, int]] = []
    cur: list[int] = []

    def flush_cur():
        if cur:
            kept_ranges.append((cur[0], cur[-1]))
            cur.clear()

    for part in parts:
        is_real = not part.startswith("… lines ")
        cost = len(part) + 1
        if out_chars + cost > MAX_OUTPUT:
            truncated_out = True
            break
        body_parts.append(part)
        out_chars += cost
        if is_real:
            try:
                lineno = int(part.split(":", 1)[0])
            except ValueError:
                continue
            if cur and lineno == cur[-1] + 1:
                cur.append(lineno)
            else:
                flush_cur()
                cur.append(lineno)
    flush_cur()

    for s, e in new_ranges:
        if (s, e) in kept_ranges or any(s >= ks and e <= ke for ks, ke in kept_ranges):
            mark_delivered(str(path), st.st_mtime_ns, st.st_size, s, e)
        else:
            # partial survival after the cap: mark only the kept prefix
            for ks, ke in kept_ranges:
                if ks <= s <= e <= ke:
                    continue
            overlap = max(0, min(e, max(ke for _, ke in kept_ranges)) - s + 1) if kept_ranges else 0
            if overlap:
                mark_delivered(str(path), st.st_mtime_ns, st.st_size, s, s + overlap - 1)

    body = "\n".join(body_parts)

    if reached_eof and total_all is not None:
        footer = f"(End of file - total {total_all} lines)"
    else:
        footer = f"(Showing line {start_line}-{last_real}. Use offset={last_real + 1} to continue.)"
    if truncated_out:
        footer += " (Output capped at 50 KB.)"
    if omitted:
        footer += f" Deduped: {omitted} previously-sent lines omitted."

    return {
        "output": f"<{path}>…</{path}>\n<type>file</type>\n<content>\n{body}\n</content>\n{footer}",
        "metadata": {"loaded": [str(path)]},
    }


def _read_fast(
    path: Path,
    offset: int,
    limit: int,
    st=None,
    mode: str = "auto",
    force_full: bool = False,
) -> dict:
    """Read a small file in one shot: fewer syscalls and no per-line IO loop.

    Used for files up to ~2 MB; the whole payload is decoded at once and the
    requested window is sliced from the line list. Universal-newline semantics
    (CRLF / lone-CR folding) are mirrored so line numbers match the streaming
    path exactly. Bigger files still use the streaming path so memory stays
    bounded.
    """
    try:
        with path.open("rb") as f:
            data = f.read()
    except OSError as e:
        return _read_error(path, e)
    if _is_binary_sample(data):
        return {"output": f"File {path} is a binary file and cannot be read as text.", "error": True}
    # newline=None (universal) in the streaming path folds \r\n and lone \r to
    # \n before splitting into lines — mirror that here. splitlines(keepends)
    # yields exactly the same items the TextIOWrapper iterator would (each line
    # with its terminator, or a final unterminated line; nothing for an empty
    # file), so line counts match the streaming path for CRLF too.
    text = data.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines(keepends=True)

    if (
        mode == "auto"
        and not force_full
        and int(offset) == 1
        and int(limit) >= AUTO_OUTLINE_LINES
        and len(lines) > AUTO_OUTLINE_LINES
    ):
        return _outline(path)

    numbered, total, reached_eof, truncated_out = _read_window_lines(lines, max(1, int(offset)), limit)

    return _finalize_file_window(
        path,
        st,
        numbered,
        max(1, int(offset)),
        force_full,
        reached_eof=reached_eof and not truncated_out,
        total_all=len(lines),
        truncated_out=truncated_out,
    )


def _read_file(
    path: Path,
    offset: int = 1,
    limit: int = MAX_LINES,
    mode: str = "auto",
    force_full: bool = False,
) -> dict:
    # image / pdf -> base64 file attachment
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS or suffix in PDF_EXTENSIONS:
        try:
            data = path.read_bytes()
        except OSError as e:
            return _read_error(path, e)
        mime, _ = mimetypes.guess_type(str(path))
        b64 = base64.b64encode(data).decode()
        return {
            "output": f"<{path.name}><type>{mime}</type><content>data:{mime};base64,{b64}</content>",
            "metadata": {"loaded": [str(path)], "mime": mime},
        }
    if mode == "outline":
        return _outline(path)

    # Whole-file fast path for ordinary-sized files; larger files stream so we
    # never hold a huge payload in memory just to read a window of it.
    try:
        st = path.stat()
    except OSError as e:
        return _read_error(path, e)
    if st.st_size <= FAST_READ_BYTES:
        return _read_fast(path, offset, limit, st=st, mode=mode, force_full=force_full)

    # binary detection + text window share ONE open: the old code opened the
    # file once for the binary sample and again for the text window, so a file
    # modified between the two opens could be torn (or the second open could
    # hit a different inode / fail on a deleted path). Seek back after the
    # sample and read the window from the same handle.
    start = max(0, offset - 1)
    limit = max(1, int(limit))
    end = start + limit

    # stream the window line-by-line, holding only the selected lines in memory
    numbered: list[str] = []
    total: int | None = None
    reached_eof = True
    out_chars = 0
    truncated_out = False
    try:
        with path.open("rb") as f:
            if _is_binary_sample(f.read(1024)):
                return {"output": f"File {path} is a binary file and cannot be read as text.", "error": True}
            f.seek(0)
            # newline=None so universal-newline splitting matches the old
            # text-mode open exactly (a CRLF / lone-CR file is not treated as
            # one giant line).
            stream = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline=None)
            try:
                for lineno, line in enumerate(stream, 1):
                    if lineno > end:
                        reached_eof = False
                        break
                    total = lineno
                    if lineno <= start:
                        continue
                    content = line.rstrip("\n").rstrip("\r")
                    if len(content) > MAX_CHARS:
                        content = content[:MAX_CHARS] + f"... (line truncated to {MAX_CHARS} chars)"
                    numbered.append(f"{lineno}: {content}")
                    capped_chars = (len(content) + 16)
                    if out_chars + capped_chars > MAX_OUTPUT:
                        truncated_out = True
                        reached_eof = False
                        break
                    out_chars += capped_chars
            finally:
                # keep the wrapper from closing/stealing the raw buffer when it
                # goes out of scope (the `with` on `f` manages fd lifetime).
                stream.detach()
    except OSError as e:
        return _read_error(path, e)

    return _finalize_file_window(
        path,
        st,
        numbered,
        start + 1,
        force_full,
        reached_eof=reached_eof and not truncated_out,
        total_all=total if (reached_eof and total is not None) else None,
        truncated_out=truncated_out,
    )


def _read_directory(path: Path, offset: int = 1, limit: int = MAX_LINES) -> dict:
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        return {"output": f"Could not read directory {path}: {e}", "error": True}
    total = len(entries)
    start = max(0, offset - 1)
    selected = entries[start : start + limit]
    lines = [(f"{p.name}/" if p.is_dir() else p.name) for p in selected]
    body = "\n".join(lines)
    if start == 0 and total <= limit:
        footer = f"(Showing {total} entries)"
    else:
        end = min(start + limit, total)
        footer = f"(Showing {start + 1}-{end} of {total} entries. Use offset={end + 1} to continue.)"
    return {
        "output": f"<{path}>…</{path}>\n<type>directory</type>\n<content>\n{body}\n</content>\n{footer}",
        "metadata": {"loaded": [str(path)]},
    }


def _read(
    filePath: str,
    offset: int = 1,
    limit: int = MAX_LINES,
    mode: str = "auto",
    force_full: bool = False,
) -> dict:
    path = Path(filePath)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        suggestion = _fuzzy_suggestion(path)
        msg = f"Path {path} does not exist."
        if suggestion:
            msg += f"\n\nDid you mean one of these?\n{suggestion}"
        return {"output": msg, "error": True}
    if path.is_dir():
        return _read_directory(path, offset=offset, limit=limit)
    return _read_file(path, offset=offset, limit=limit, mode=mode, force_full=force_full)


def tool() -> Tool:
    description = """Read a file or directory from the local filesystem. If the path does not exist, an error is returned.

Usage:
- The filePath parameter should be an absolute path.
- By default, this tool returns up to 2000 lines from the start of the file.
- The offset parameter is the line number to start from (1-indexed).
- To read later sections, call this tool again with a larger offset.
- mode="auto" (default): files over ~150 lines return a compact OUTLINE first
  (definitions/headings with line numbers). Use those line numbers with
  offset/limit to fetch just the regions you need — much cheaper than reading
  everything. Pass mode="full" to read raw content directly.
- Deduplication: content already delivered to you earlier in this session is
  NOT sent again. Re-requesting identical lines returns a short stub; partial
  overlaps return only the new lines. If a stub blocks you after compaction,
  re-request with force_full=true to get the real body again.
- Use the grep tool to find specific content in large files or files with long lines.
- If you are unsure of the correct file path, use the glob tool to look up filenames by glob pattern.
- Contents are returned with each line prefixed by its line number as `<line>: <content>`. For example, if a file has contents "foo\\n", you will receive "1: foo\\n". For directories, entries are returned one per line (without line numbers) with a trailing `/` for subdirectories.
- Any line longer than 2000 characters is truncated.
- Call this tool in parallel when you know there are multiple files you want to read.
- Avoid tiny repeated slices (30 line chunks). If you need more context, read a larger window.
- This tool can read image files and PDFs and return them as file attachments."""

    def run(input: dict) -> dict:
        return _read(
            input["filePath"],
            offset=int(input.get("offset") or 1),
            limit=int(input.get("limit") or MAX_LINES),
            mode=input.get("mode") or "auto",
            force_full=bool(input.get("force_full")),
        )

    return Tool(
        name="read",
        description=description,
        parameters=schema_with(
            {
                "filePath": {"type": "string", "description": "The absolute path to the file or directory to read"},
                "offset": {"type": "integer", "description": "The line number to start from (1-indexed)", "optional": True},
                "limit": {"type": "integer", "description": "The number of lines to read", "optional": True},
                "mode": {
                    "type": "string",
                    "enum": ["auto", "full", "outline"],
                    "description": 'auto (default): big files give an outline first; full: raw lines; outline: structure only.',
                    "optional": True,
                },
                "force_full": {
                    "type": "boolean",
                    "description": "Resend content even if it was already delivered earlier this session.",
                    "optional": True,
                },
            },
            ["filePath"],
        ),
        run=run,
        permission="read",
    )
