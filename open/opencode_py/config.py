"""Config loading / deep-merge for opencode_py.

Mirrors opencode's config model (ConfigV1.Info) without pydantic:
  - Files: opencode.json / opencode.jsonc (JSONC = JSON with comments/trailing commas)
    discovered from the project dir (walking up to worktree) then the user-level
    config dir. Later sources override earlier (deep merge).
  - Env overrides: OPENCODE_CONFIG (file), OPENCODE_CONFIG_CONTENT (raw JSON),
    OPENCODE_PERMISSION (JSON merged into permission).
  - Config variable substitution: {env:VAR}, {file:path}.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .globals import Path as GPath

_VAR_RE = re.compile(r"\{env:([A-Za-z0-9_]+)\}|\{file:([^}]+)\}")
_ALLOWED_ENV_VARS = {"OPENCODE_API_KEY", "OPENCODE_BASE_URL", "OPENCODE_MODEL", "OPENCODE_PROVIDER", "OPENCODE_SMALL_MODEL"}
_FILE_VAR_MAX_BYTES = 4096


def _load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(_strip_jsonc(text))


def _strip_jsonc(text: str) -> str:
    """JSONC -> JSON: strip // and /* */ comments (keeps strings intact).

    Handles nested block comments and treats string escapes (including \\uXXXX
    unicode escapes) as opaque so a comment marker can never appear from inside
    an escape sequence.
    """
    out = []
    i = 0
    in_str = False
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                nxt = text[i + 1]
                out.append(nxt)
                i += 2
                # skip the rest of a \uXXXX unicode escape atomically so its
                # hex digits can never be confused with comment markers
                if nxt == "u" and i + 4 <= n:
                    for _ in range(4):
                        out.append(text[i])
                        i += 1
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                while i < n and text[i] != "\n":
                    i += 1
                continue
            if nxt == "*":
                depth = 1
                i += 2
                while depth > 0 and i < n:
                    if text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                        depth -= 1
                        i += 2
                    elif text[i] == "/" and i + 1 < n and text[i + 1] == "*":
                        depth += 1
                        i += 2
                    else:
                        i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base (returns a new dict). Lists are replaced."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def substitute_vars(text: str, base_dir: Path) -> str:
    def repl(m: re.Match) -> str:
        env = m.group(1)
        if env:
            if env not in _ALLOWED_ENV_VARS:
                return ""
            return os.environ.get(env, "")[:_FILE_VAR_MAX_BYTES]
        fpath = m.group(2)
        p = Path(fpath).expanduser()
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        else:
            p = p.resolve()
        try:
            base_res = base_dir.resolve()
            p.relative_to(base_res)
        except ValueError:
            return ""
        try:
            data = p.read_text(encoding="utf-8")
        except OSError:
            return ""
        if len(data) > _FILE_VAR_MAX_BYTES:
            data = data[:_FILE_VAR_MAX_BYTES]
        return data
    return _VAR_RE.sub(repl, text)


def _apply_vars(value: Any, base_dir: Path) -> Any:
    if isinstance(value, str):
        return substitute_vars(value, base_dir)
    if isinstance(value, list):
        return [_apply_vars(v, base_dir) for v in value]
    if isinstance(value, dict):
        return {k: _apply_vars(v, base_dir) for k, v in value.items()}
    return value


DEFAULT_MODEL = "muse-spark-1.3-contributor-free"
DEFAULT_SMALL_MODEL = "mimo-v2.5-free"


def _usage_path() -> Path:
    return GPath.data / "model-usage.json"


def record_model_use(provider: str, model: str) -> None:
    """Count one answered turn for (provider, model).

    Best-effort, never raises: powers the most-used default (see
    most_used_model). Corrupt cache degrades to a fresh count.
    """
    try:
        provider = str(provider or "").strip() or "opencode"
        model = str(model or "").strip()
        if not model:
            return
        try:
            data = json.loads(_usage_path().read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        counts = data.get("counts")
        if not isinstance(counts, dict):
            counts = {}
        key = f"{provider}/{model.split('/', 1)[-1]}"
        counts[key] = int(counts.get(key, 0) or 0) + 1
        data["counts"] = counts
        _usage_path().parent.mkdir(parents=True, exist_ok=True)
        tmp = _usage_path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, _usage_path())
    except Exception:
        pass


def most_used_model() -> str | None:
    """The user's most-answered (provider/model), or None when unknown.

    Counts only models that still exist upstream (a removed model can never
    win, even if it was once #1). Never raises; None on any failure.
    """
    try:
        data = json.loads(_usage_path().read_text(encoding="utf-8"))
        counts = data.get("counts")
        if not isinstance(counts, dict) or not counts:
            return None
    except (OSError, ValueError):
        return None
    try:
        from .providers.rotation import fetch_catalog
        catalog = (fetch_catalog().get("opencode") or {}).get("models", {})
        live = {mid for mid, m in catalog.items()
                if m.get("status", "active") == "active"}
    except Exception:
        live = set()
    if not live or not catalog:
        # catalog unreachable (offline / first run): can't prove removal —
        # count everything rather than wrongly dropping the real winner.
        live = set()
    try:
        from .providers.rotation import check_zen_model_health
        health = check_zen_model_health(
            [str(k).split("/", 1)[-1] if "/" in str(k) else str(k) for k in counts])
    except Exception:
        health = {}
    best: str | None = None
    best_n = 0
    for key, n in counts.items():
        try:
            n = int(n or 0)
        except (TypeError, ValueError):
            continue
        if n <= best_n:
            continue
        bare = str(key).split("/", 1)[-1] if "/" in str(key) else str(key)
        if live and bare not in live and str(key) not in live:
            continue  # removed upstream — can never win
        if health.get(bare) is False:
            continue  # not answering right now — skip, don't crown it
        best, best_n = str(key), n
    if best is None:
        # every counted model is dead/removed: fall back to the
        # most-answering LIVE model rather than None (which would start on
        # the hardcoded default while a healthy winner exists).
        for key, n in sorted(counts.items(), key=lambda kv: int(kv[1] or 0), reverse=True):
            bare = str(key).split("/", 1)[-1] if "/" in str(key) else str(key)
            if live and bare not in live and str(key) not in live:
                continue
            if health.get(bare) is False:
                continue
            return str(key)
    return best


@dataclass
class Config:
    provider: str = "opencode"
    model: str = DEFAULT_MODEL
    small_model: str = DEFAULT_SMALL_MODEL
    default_agent: str = "build"
    subagent_depth: int = 1
    username: str = field(default_factory=lambda: os.environ.get("USER", "user"))
    instructions: list[str] = field(default_factory=list)
    permission: dict[str, Any] = field(default_factory=dict)
    agents: dict[str, Any] = field(default_factory=dict)
    providers: dict[str, Any] = field(default_factory=dict)
    commands: dict[str, Any] = field(default_factory=dict)
    rotation: list[dict[str, str]] = field(default_factory=list)
    system_prompt: str = ""
    theme: str = "opencode"
    diff_style: str = "split"  # "split" (auto side-by-side >120 cols) | "stacked" (unified always)
    diff_wrap_mode: str = "word"  # "word" | "none"
    suppress_backgrounds: bool = False
    # the `Build (2 of 4) 12,345 (2%) Prev/Next` bar under a sub-agent chat.
    # OFF by default: arrow-key session navigation works without it, and the
    # bar ate a screen line on small phone displays.
    subagent_footer: bool = False
    tool_output_max_lines: int = 2000
    tool_output_max_bytes: int = 51200
    compaction_enabled: bool = True
    compaction_tail_turns: int = 2
    context_budget: int = 120000
    bash_default_timeout: int = 120
    model_read_timeout: float = 300.0
    auto_retry: bool = True
    auto_retry_count: int = 5
    auto_approve: bool = False
    rotation_lock: bool = False
    # "auto" = never ask (allow-all; explicit deny rules still apply),
    # "ask"  = popup allow/deny for every action not covered by allow rules,
    # "deny" = block everything not explicitly allowed,
    # "fully_auto" = auto + zero popups of ANY kind: suppresses the question
    #   tool dialog too, so the model runs to the goal uninterrupted.
    permission_mode: str = "auto"
    # Reasoning effort (official variant parity): e.g. "high" for muse-spark.
    # "" = gateway default. Only sent when valid for the CURRENT model per the
    # live catalog — future models with new levels work with zero code changes.
    reasoning_effort: str = ""
    # Whether thought bubbles (`+ Thought` rows) render at all. Off hides
    # them without dropping the underlying reasoning (re-enabling shows them
    # again, /thinking last still prints in headless).
    show_thoughts: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> "Config":
        raw = data
        cfg = cls()
        cfg.raw = raw
        base = base_dir or Path.cwd()
        if "provider" in data:
            cfg.provider = data["provider"]
        if "model" in data:
            cfg.model = _parse_model(data["model"], cfg.provider)
        elif "model" not in data:
            # no saved pick: start on the user's most-used live model
            # (falls back to DEFAULT_MODEL when usage is unknown).
            try:
                fav = most_used_model()
                if fav:
                    cfg.model = _parse_model(fav, cfg.provider)
            except Exception:
                pass
        if "small_model" in data:
            cfg.small_model = _parse_model(data["small_model"], cfg.provider)
        if "default_agent" in data:
            cfg.default_agent = data["default_agent"]
        if "subagent_depth" in data:
            try:
                cfg.subagent_depth = int(data["subagent_depth"])
            except (ValueError, TypeError):
                pass
        if "username" in data:
            cfg.username = data["username"]
        if "instructions" in data:
            cfg.instructions = [str(x) for x in data["instructions"]]
        if "permission" in data:
            perm = _apply_vars(data["permission"], base)
            # opencode allows `permission: "ask"` as a global default action;
            # normalize it to `{"*": <action>}` so consumer code (permission
            # engine, /config validate) always sees a dict and a malformed
            # value can never crash the engine at startup.
            if isinstance(perm, str):
                perm = {"*": perm}
            cfg.permission = perm if isinstance(perm, dict) else {}
        if "agent" in data:
            cfg.agents = _apply_vars(data["agent"], base)
        if "agents" in data:
            # user wrote the plural field name: honor it (overrides singular)
            cfg.agents = _apply_vars(data["agents"], base)
        if "provider" in data and isinstance(data.get("provider"), dict):
            # 'provider' key is the default provider id; custom providers live under
            # 'providers' to avoid ambiguity with the top-level model selection.
            pass
        if "providers" in data:
            cfg.providers = _apply_vars(data["providers"], base)
        if "commands" in data:
            cfg.commands = data["commands"]
        if "rotation" in data:
            # Sanitize so a malformed `rotation` (string, list of non-dicts /
            # entries without a provider/model) can never crash build_rotation
            # or Rotation.stream at startup. Non-dict lanes are dropped; a
            # wholly-invalid value just falls back to the default lane.
            rot = data["rotation"]
            if isinstance(rot, list):
                cfg.rotation = [
                    {k: v for k, v in lane.items() if k in ("provider", "model")}
                    for lane in rot
                    if isinstance(lane, dict) and lane.get("provider") and lane.get("model")
                ]
            else:
                cfg.rotation = []
        if "system_prompt" in data:
            cfg.system_prompt = data["system_prompt"]
        if "theme" in data:
            cfg.theme = data["theme"]
        if "diff" in data:
            d = data["diff"]
            if isinstance(d, dict):
                cfg.diff_style = str(d.get("style", cfg.diff_style))
                cfg.diff_wrap_mode = str(d.get("wrap", cfg.diff_wrap_mode))
                cfg.suppress_backgrounds = bool(d.get("suppress_backgrounds", cfg.suppress_backgrounds))
        if "subagent_footer" in data:
            cfg.subagent_footer = bool(data["subagent_footer"])
        if "tool_output" in data:
            to = data["tool_output"]
            try:
                cfg.tool_output_max_lines = int(to.get("max_lines", 2000))
            except (ValueError, TypeError):
                pass
            try:
                cfg.tool_output_max_bytes = int(to.get("max_bytes", 51200))
            except (ValueError, TypeError):
                pass
        if "context_budget" in data:
            try:
                cfg.context_budget = int(data["context_budget"])
            except (ValueError, TypeError):
                pass
        if "bash_default_timeout" in data:
            try:
                cfg.bash_default_timeout = int(data["bash_default_timeout"])
            except (ValueError, TypeError):
                pass
        if "model_read_timeout" in data:
            try:
                cfg.model_read_timeout = float(data["model_read_timeout"])
            except (ValueError, TypeError):
                pass
        if "auto_retry" in data:
            cfg.auto_retry = bool(data["auto_retry"])
        if "auto_retry_count" in data:
            try:
                cfg.auto_retry_count = int(data["auto_retry_count"])
            except (ValueError, TypeError):
                pass
        if "permission_mode" in data:
            pm = str(data["permission_mode"]).lower()
            if pm in ("auto", "ask", "deny", "fully_auto"):
                cfg.permission_mode = pm
        if "reasoning_effort" in data:
            cfg.reasoning_effort = str(data["reasoning_effort"] or "").strip().lower()
        if "show_thoughts" in data:
            cfg.show_thoughts = bool(data["show_thoughts"])
        if "rotation_lock" in data:
            cfg.rotation_lock = bool(data["rotation_lock"])
        if "compaction" in data:
            c = data["compaction"]
            cfg.compaction_enabled = bool(c.get("auto", True))
            try:
                cfg.compaction_tail_turns = int(c.get("tail_turns", 2))
            except (ValueError, TypeError):
                pass
        # OPENCODE_PERMISSION env overrides permission
        if os.environ.get("OPENCODE_PERMISSION"):
            try:
                override = json.loads(os.environ["OPENCODE_PERMISSION"])
                cfg.permission = deep_merge(cfg.permission, override)
            except json.JSONDecodeError:
                pass
        return cfg

    def as_dict(self, *, show_secrets: bool = False) -> dict:
        providers = self.providers
        if not show_secrets and isinstance(providers, dict):
            redacted: dict[str, Any] = {}
            for pid, pconf in providers.items():
                if isinstance(pconf, dict) and "api_key" in pconf:
                    p = dict(pconf)
                    p["api_key"] = "*** (hidden, use --show-secrets to display)"
                    redacted[pid] = p
                else:
                    redacted[pid] = pconf
            providers = redacted
        return {
            "provider": self.provider,
            "model": self.model,
            "small_model": self.small_model,
            "default_agent": self.default_agent,
            "subagent_depth": self.subagent_depth,
            "username": self.username,
            "instructions": self.instructions,
            "permission": self.permission,
            "agent": self.agents,
            "providers": providers,
            "commands": self.commands,
            "rotation": self.rotation,
            "system_prompt": self.system_prompt,
            "theme": self.theme,
            "diff": {
                "style": self.diff_style,
                "wrap": self.diff_wrap_mode,
                "suppress_backgrounds": self.suppress_backgrounds,
            },
            "subagent_footer": self.subagent_footer,
            "tool_output": {
                "max_lines": self.tool_output_max_lines,
                "max_bytes": self.tool_output_max_bytes,
            },
            "context_budget": self.context_budget,
            "bash_default_timeout": self.bash_default_timeout,
            "model_read_timeout": self.model_read_timeout,
            "auto_retry": self.auto_retry,
            "auto_retry_count": self.auto_retry_count,
            "rotation_lock": self.rotation_lock,
            "permission_mode": self.permission_mode,
            "reasoning_effort": self.reasoning_effort,
            "show_thoughts": self.show_thoughts,
            "compaction": {
                "auto": self.compaction_enabled,
                "tail_turns": self.compaction_tail_turns,
            },
        }


def _parse_model(model: str, default_provider: str) -> str:
    if "/" in model:
        parts = model.split("/")
        if len(parts) == 2 and parts[0] and parts[1]:
            return model
        return f"{default_provider}/{model.replace('/', '-')}"
    return f"{default_provider}/{model}"


def _project_config_files(directory: Path) -> list[Path]:
    """Find opencode.json/opencode.jsonc walking up from directory to worktree root."""
    from .globals import resolve_worktree

    worktree = resolve_worktree(directory)
    files: list[Path] = []
    d = directory.resolve()
    while True:
        for name in ("opencode.jsonc", "opencode.json"):
            p = d / name
            if p.exists():
                files.append(p)
        if d == worktree:
            break
        if d.parent == d:
            break
        d = d.parent
    return files


def _user_config_file() -> Path | None:
    for name in ("opencode.jsonc", "opencode.json"):
        p = GPath.config / name
        if p.exists():
            return p
    return None


def load_config(directory: Path | None = None) -> Config:
    """Load config following opencode's merge order (later wins)."""
    directory = directory or Path.cwd()
    merged: dict[str, Any] = {}

    sources: list[tuple[Path, dict]] = []

    def add(path: Path) -> None:
        try:
            sources.append((path, _load_json(path)))
        except (OSError, json.JSONDecodeError):
            pass

    # user-level global config first (lowest priority), then project configs
    user = _user_config_file()
    if user:
        add(user)
    for p in _project_config_files(directory):
        add(p)

    # OPENCODE_CONFIG env points to an explicit file (overrides discovered)
    if os.environ.get("OPENCODE_CONFIG"):
        add(Path(os.environ["OPENCODE_CONFIG"]).expanduser())

    # OPENCODE_CONFIG_CONTENT inline JSON (highest)
    if os.environ.get("OPENCODE_CONFIG_CONTENT"):
        try:
            sources.append((Path.cwd(), json.loads(os.environ["OPENCODE_CONFIG_CONTENT"])))
        except json.JSONDecodeError:
            pass

    for path, data in sources:
        data = _apply_vars(data, path.parent)
        merged = deep_merge(merged, data)

    return Config.from_dict(merged, directory)


def save_config(cfg: Config, path: Path | None = None) -> None:
    path = path or (GPath.config / "opencode.json")
    out = cfg.as_dict()
    for key, value in (cfg.raw or {}).items():
        if key not in out:
            out[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    from .session import _write_durable
    _write_durable(path, json.dumps(out, indent=2, ensure_ascii=False), file_sync=True, dir_sync=True)
