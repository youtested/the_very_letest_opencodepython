from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import Tool, schema_with

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME = 64
MAX_DESC = 1024
MAX_BODY = 50 * 1024

SKILL_DIRS = (".opencode/skills", ".claude/skills", ".agents/skills")

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, list["Skill"]]] = {}
_CACHE_TTL = 5.0


@dataclass
class Skill:
    name: str
    description: str
    body: str
    source: str = ""


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            end = i
            break
    if end < 0:
        return {}, text
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line or line[:1] in (" ", "\t"):
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        if key in ("name", "description"):
            meta[key] = val.strip().strip("\"'")
    return meta, "\n".join(lines[end + 1:])


def _valid(name: str, desc: str, dirname: str) -> bool:
    return (
        bool(name)
        and len(name) <= MAX_NAME
        and bool(NAME_RE.match(name))
        and name == dirname
        and bool(desc)
        and len(desc) <= MAX_DESC
    )


def _read_skill_file(path: Path) -> Skill | None:
    try:
        if path.stat().st_size > MAX_BODY + 4096:
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    meta, body = _parse_frontmatter(text)
    name = meta.get("name", "")
    desc = meta.get("description", "")
    if not _valid(name, desc, path.parent.name):
        return None
    body = body.strip()
    if not body:
        return None
    if len(body.encode("utf-8")) > MAX_BODY:
        body = body.encode("utf-8")[:MAX_BODY].decode("utf-8", "replace")
    return Skill(name=name, description=desc, body=body, source=str(path))


def _worktree() -> Path:
    try:
        from ..globals import resolve_worktree

        return resolve_worktree(Path.cwd())
    except OSError:
        return Path.cwd()


def _scan() -> list[Skill]:
    found: dict[str, Skill] = {}
    wt = _worktree().resolve()
    chain: list[Path] = [wt]
    for parent in wt.parents:
        chain.append(parent)
        if (parent / ".git").exists():
            break
    for base in chain:
        for sub in SKILL_DIRS:
            root = base / sub
            try:
                kids = sorted(p for p in root.iterdir() if p.is_dir())
            except OSError:
                continue
            for kid in kids:
                if kid.name in found:
                    continue
                skill = _read_skill_file(kid / "SKILL.md")
                if skill is not None:
                    found[kid.name] = skill
    try:
        from ..globals import Path as GPath

        for sub in ("skills",):
            root = GPath.config / "opencode" / sub
            alt = GPath.config / sub
            for bucket in (root, alt):
                try:
                    kids = sorted(p for p in bucket.iterdir() if p.is_dir())
                except OSError:
                    continue
                for kid in kids:
                    if kid.name in found:
                        continue
                    skill = _read_skill_file(kid / "SKILL.md")
                    if skill is not None:
                        found[kid.name] = skill
    except Exception:
        pass
    try:
        home = Path.home()
        for sub in (".claude/skills", ".agents/skills"):
            root = home / sub
            try:
                kids = sorted(p for p in root.iterdir() if p.is_dir())
            except OSError:
                continue
            for kid in kids:
                if kid.name in found:
                    continue
                skill = _read_skill_file(kid / "SKILL.md")
                if skill is not None:
                    found[kid.name] = skill
    except Exception:
        pass
    return sorted(found.values(), key=lambda s: s.name)


def list_skills(*, fresh: bool = False) -> list[Skill]:
    key = str(_worktree())
    now = __import__("time").monotonic()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and not fresh:
            ts, skills = hit
            if now - ts < _CACHE_TTL:
                return list(skills)
    skills = _scan()
    with _LOCK:
        _CACHE[key] = (now, skills)
    return list(skills)


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def visible_skills(permission=None) -> list[Skill]:
    skills = list_skills()
    if permission is None:
        return skills
    try:
        from ..permission import PermissionEngine

        match = PermissionEngine.match
    except Exception:
        return skills
    out: list[Skill] = []
    for s in skills:
        action = "allow"
        try:
            action = permission.evaluate("skill", s.name)
        except Exception:
            action = "allow"
        if action == "deny":
            continue
        if action == "ask" and getattr(permission, "mode", "auto") == "deny":
            continue
        out.append(s)
    return out


def skills_block(permission=None, limit: int = 40) -> str:
    skills = visible_skills(permission)[: max(0, limit)]
    if not skills:
        return ""
    lines = ["<available_skills>"]
    for s in skills:
        lines.append("<skill>")
        lines.append(f"<name>{s.name}</name>")
        lines.append(f"<description>{s.description}</description>")
        lines.append("</skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _load(name: str, permission=None) -> Skill | None:
    want = (name or "").strip()
    if not want or not NAME_RE.match(want):
        return None
    for s in list_skills():
        if s.name == want:
            return s
    return None


def tool(registry=None) -> Tool:
    description = (
        "Load a skill: reusable instructions from SKILL.md files. "
        "The system prompt lists <available_skills> with name + description; "
        "call skill({name}) to load the full content when the task matches, "
        "then follow it."
    )

    def run(arguments: dict) -> dict:
        name = str(arguments.get("name", "")).strip()
        if not name:
            skills = visible_skills(getattr(registry, "_skill_permission", None))
            if not skills:
                return {"output": "No skills installed. Create .opencode/skills/<name>/SKILL.md."}
            lines = ["Available skills:"]
            for s in skills[:40]:
                lines.append(f"- {s.name}: {s.description}")
            lines.append('Load one with skill({"name": "<name>"}).')
            return {"output": "\n".join(lines)}
        skill = _load(name)
        if skill is None:
            skills = visible_skills(getattr(registry, "_skill_permission", None))
            names = ", ".join(s.name for s in skills[:20]) or "(none installed)"
            return {"output": f'Unknown skill "{name}". Available: {names}', "error": True}
        return {
            "output": f"# Skill: {skill.name}\n{skill.description}\n\n{skill.body}",
            "metadata": {"skill": skill.name, "source": skill.source},
        }

    return Tool(
        name="skill",
        description=description,
        parameters=schema_with(
            {
                "name": {
                    "type": "string",
                    "description": "Skill name to load (folder name in skills/). Empty lists available skills.",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="skill",
    )
