"""write tool: create/overwrite a file."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .registry import Tool, schema_with


def _atomic_write(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
            try:
                fh.flush()
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
        try:
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            except OSError:
                pass
            finally:
                try:
                    os.close(dfd)
                except OSError:
                    pass
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write(filePath: str, content: str) -> dict:
    path = Path(filePath)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, content)
    except (OSError, UnicodeError) as e:
        return {"output": f"Error writing file {path}: {e}", "error": True}
    # feed the verify tool's homework checker
    from .verify import track

    track(path, "write")
    return {
        "output": "Wrote file successfully.",
        "metadata": {"filePath": str(path), "content": content},
    }


def tool() -> Tool:
    description = """Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked."""

    def run(input: dict) -> dict:
        return _write(input["filePath"], input["content"])

    return Tool(
        name="write",
        description=description,
        parameters=schema_with(
            {
                "content": {"type": "string", "description": "The content to write to the file"},
                "filePath": {"type": "string", "description": "The absolute path to the file to write"},
            },
            ["content", "filePath"],
        ),
        run=run,
        permission="edit",
    )
