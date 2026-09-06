"""Provider factory + registry + failover rotation.

rotation list: [{"provider": "zen", "model": "..."}, {"provider": "groq", "model": "..."}, ...]
On a real rate limit (429 / "limit reached") the engine tries the next lane.
Transient hiccups (timeout, 5xx, overload, empty reply) keep the current model.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from ..config import Config
from .base import ContextOverflowError, ProviderError, ProviderEvent, RateLimitError

if TYPE_CHECKING:  # pragma: no cover - annotations only, httpx stays lazy at runtime
    import httpx

# free-tier direct providers (OpenAI-compatible). Env var names match auth.py.
FREE_PROVIDERS: dict[str, dict[str, Any]] = {
    "groq": {"name": "Groq", "base_url": "https://api.groq.com/openai/v1", "env": ("GROQ_API_KEY",)},
    "cerebras": {"name": "Cerebras", "base_url": "https://api.cerebras.ai/v1", "env": ("CEREBRAS_API_KEY",)},
    "google": {
        "name": "Google AI Studio",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "env": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "env": ("OPENROUTER_API_KEY",),
        "headers": {"HTTP-Referer": "https://opencode.ai/", "X-Title": "opencode"},
    },
    "nvidia": {
        "name": "NVIDIA NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env": ("NVIDIA_API_KEY",),
        "headers": {"HTTP-Referer": "https://opencode.ai/", "X-Title": "opencode"},
    },
    "mistral": {"name": "Mistral", "base_url": "https://api.mistral.ai/v1", "env": ("MISTRAL_API_KEY",)},
    "github": {"name": "GitHub Models", "base_url": "https://models.github.ai/inference", "env": ("GITHUB_TOKEN",)},
    "sambanova": {"name": "SambaNova", "base_url": "https://api.sambanova.ai/v1", "env": ("SAMBANOVA_API_KEY",)},
    "togetherai": {"name": "Together", "base_url": "https://api.together.xyz/v1", "env": ("TOGETHER_API_KEY",)},
}

# default free models per provider (Phase 3 preload).
# "zen" is the most-used free lane: muse-spark-1.3 (1M ctx, multimodal,
# reasoning) — the model this user runs almost everything on.
FREE_DEFAULT_MODELS: dict[str, str] = {
    "zen": "muse-spark-1.3-contributor-free",
    "groq": "llama-3.3-70b-versatile",
    "cerebras": "llama-3.3-70b",
    "google": "gemini-2.5-flash",
    "openrouter": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia": "nemotron-3-ultra-free",
    "mistral": "codestral-latest",
    "github": "gpt-4o-mini",
    "sambanova": "Meta-Llama-3.3-70B-Instruct",
    "togetherai": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}

# paid (bring-your-own-key) providers with their own /models endpoints.
PAID_PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "env": ("ANTHROPIC_API_KEY",),
        "api_kind": "anthropic",
    },
    "openai": {"name": "OpenAI", "base_url": "https://api.openai.com/v1", "env": ("OPENAI_API_KEY",)},
    "deepseek": {"name": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "env": ("DEEPSEEK_API_KEY",)},
    "xai": {"name": "xAI", "base_url": "https://api.x.ai/v1", "env": ("XAI_API_KEY",)},
    "deepinfra": {"name": "DeepInfra", "base_url": "https://api.deepinfra.com/v1/openai", "env": ("DEEPINFRA_API_KEY",)},
}


# Keep the historical local name (used across the rotation code) but delegate
# to the shared classifier so detection is identical everywhere.
from .classify import is_context_overflow as _is_context_overflow_message  # noqa: E402


def _is_rate_limit_message(message: str) -> bool:
    from .classify import is_rate_limit

    return is_rate_limit(message)


def _fail_message(provider_id: str, error: Exception) -> str:
    """Message for a surfaced failure; keeps prior lane failures for context."""
    return f"{provider_id}: {error}"


def build_read_timeout(read_seconds: float | None = None, model: str = "") -> httpx.Timeout:
    """httpx timeout with a long configurable read window (streaming).

    The read timeout bounds the gap *between* SSE chunks. Mirrors the
    official client's 300s inter-chunk window (provider.ts chunkTimeout):
    free-tier models routinely think for minutes before emitting a token, so
    a short read timeout kills mid-conversation turns. Users can
    raise/lower it via `model_read_timeout` in settings.

    Reasoning models (deepseek, o1/o3, claude-sonnet-4, gemini-2.5, grok-3,
    ...) stream `reasoning_content` and can sit silently for many minutes
    between tokens mid-thought. A fixed gap timeout trips them and the auto
    retry's "keep going" nudge interrupts their chain of thought — the model
    looks like it "stopped responding". Give those lanes a much longer gap
    (never less than 900s) so a long silent think isn't killed; an explicit
    higher user setting still wins.
    """
    read = float(read_seconds) if read_seconds else 300.0
    if is_reasoning_model(model):
        read = max(read, 900.0)
    import httpx

    return httpx.Timeout(connect=10.0, read=read, write=30.0, pool=10.0)


_REASONING_MODEL_PATTERN = re.compile(
    r"deepseek|reasoner|thinking|(?:^|[/\-_])(?:o1|o3|r1)(?:$|[^0-9a-z])|"
    r"gpt-5|claude-sonnet-4|claude-opus|gemini-2\.[5-9]|grok-3|qwen|"
    r"x-preview|big-pickle|mimo-v|nemotron|-free(?:$|[^0-9a-z])",
    re.IGNORECASE,
)


def is_reasoning_model(model_id: str) -> bool:
    """True for model ids that stream a reasoning/thinking phase (long silent
    gaps between tokens) so they get a more generous streaming read timeout."""
    return bool(model_id and _REASONING_MODEL_PATTERN.search(model_id))


class Rotation:
    """Try lanes in order on rate limits / dead lanes.

    `session_id` is the stable conversation id sent to Zen as
    `x-opencode-session` (and reused for x-opencode-request). It must stay
    constant across turns so the Zen gateway keeps provider affinity and serves
    the real free quota for a trusted client. Mutable: the engine rebinds it
    when a session is created/switched, and each newly-built provider picks it
    up from here.
    """

    def __init__(self, lanes: list[dict[str, str]], make_provider: Callable[[str, str], Any], session_id: str | None = None):
        self.lanes = lanes
        self.make_provider = make_provider
        self.session_id = session_id or uuid4().hex
        # the provider currently reading a stream — the interrupt path closes it
        # so a blocked read wakes up immediately (idle "thinking" gap)
        self._active_provider: Any | None = None

    def abort(self) -> None:
        """Force-close the active provider stream, if one is being read.

        Lets the interrupt path abort instantly even when the model is silent
        (no chunk arrives to trigger the per-chunk interrupt check)."""
        provider = self._active_provider
        if provider is not None:
            try:
                provider.abort_stream()
            except Exception:
                pass

    def new_turn(self) -> None:
        """Start the next turn on a fresh Zen upstream lane.

        Zen pins each session id to ONE lane; that lane can die server-side
        between turns, so reusing it makes the first attempt of every new
        turn fail (empty reply / network_error) and forces a visible retry.
        Bumping the epoch up front draws a fresh lane for attempt #1 — the
        official client keeps no per-conversation lane at all, so this is the
        closest parity. Never raises.
        """
        try:
            from .zen import _LANE_EPOCH, _LANE_LOCK

            with _LANE_LOCK:
                _LANE_EPOCH[self.session_id] = _LANE_EPOCH.get(self.session_id, 0) + 1
        except Exception:
            pass

    @staticmethod
    def _rotate_lane(provider: Any) -> None:
        """Ask the provider to drop its current upstream lane assignment.

        Zen pins each session id to one lane; when a stream dies SERVER-SIDE
        (in-band error / empty reply with finish_reason='network_error'),
        retrying with the same identity re-hits the same corpse. Providers
        that expose rotate_session() (ZenProvider) get a fresh identity so
        the next attempt lands on a different lane. No-op elsewhere."""
        fn = getattr(provider, "rotate_session", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    @property
    def first(self) -> Any | None:
        if not self.lanes:
            return None
        l = self.lanes[0]
        return self.make_provider(l.get("provider", "zen"), l.get("model", FREE_DEFAULT_MODELS["zen"]))

    def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        on_event: Callable[[ProviderEvent], None],
        on_notice: Callable[[str, str, str], None] | None = None,
        locked: bool = False,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Stream across lanes; returns (provider_id, model_id) that succeeded.

        `on_notice(provider_id, model, reason)` is called when a lane other
        than the first succeeds (a failover happened), with a short reason for
        the switch so the UI can announce it accurately.

        `locked=True` pins the stream to the FIRST lane only (the model the
        user selected): a rate limit or hard failure surfaces as an error
        instead of failing over to another model — the TUI's red lock dot.
        Retries (transient failures) still run on the same locked lane.

        A lane fails over when it genuinely can't serve: a real rate limit
        (429 or an in-band "limit reached" error), a hard error (bad model
        id, bad key, dead endpoint), or an empty reply (dead handshake —
        the official client never retries empty turns either). Transient
        failures on the user's chosen lane (timeout, 5xx, overload) raise so
        the agent loop's own backoff retry (5 attempts, official parity) can
        wait it out on the same model instead of silently routing elsewhere.
        Temporarily-down backup lanes are skipped so a dead free model never
        blocks the chain.
        """
        errors: list[str] = []
        saw_rate_limit = False
        saw_other = False
        last_reason = ""
        lanes = self.lanes[:1] if locked else self.lanes
        for index, lane in enumerate(lanes):
            provider_id = lane.get("provider", "zen")
            model = lane.get("model", FREE_DEFAULT_MODELS.get(provider_id, FREE_DEFAULT_MODELS["zen"]))
            had_output = False
            had_text = False
            had_error = False
            error_message = ""
            live_streamed = False
            buffered: list[ProviderEvent] = []

            def wrapped(evt: ProviderEvent) -> None:
                nonlocal had_output, had_text, had_error, error_message, live_streamed
                # text, reasoning, or a real tool call = usable output.
                # Reasoning counts too: a reasoning model cut off before any
                # content must not be treated as "empty" and silently dropped.
                if evt.kind in ("text_delta", "reasoning_delta", "tool_call"):
                    had_output = True
                    if evt.kind == "text_delta":
                        had_text = True
                elif evt.kind == "error":
                    had_error = True
                    error_message = evt.error or error_message
                    if live_streamed and had_text:
                        # Real ANSWER TEXT already reached the screen, then the
                        # lane hit a hard in-band error (e.g. free-quota
                        # exhaustion mid-reply). Commit the partial answer and
                        # stop the turn clearly — do NOT replay a generic error,
                        # retry, or fail over (which would glue a backup's
                        # answer onto the partial text). Marked `partial` so the
                        # handler below doesn't re-wrap the message.
                        err = ProviderError(
                            f"{provider_id}: {evt.error or 'error response'}"
                            " — the reply was cut off by an error",
                            retryable=False,
                        )
                        err.partial = True
                        raise err
                    # Reasoning-only partials fall through to the post-stream
                    # classification below: no answer text is on screen yet, so
                    # nothing user-visible is lost by failing this lane and
                    # retrying / rotating (fixes "stops mid-thinking").
                if evt.kind in ("text_delta", "reasoning_delta"):
                    # Live-stream visible tokens so the UI isn't a frozen
                    # spinner until the whole response finishes. Unknown
                    # success = later events (tool calls, usage, done) replay,
                    # but visible content has already reached the user.
                    live_streamed = True
                    on_event(evt)
                    return
                # Buffer non-visible events (tool calls, usage, done, errors)
                # and only replay them once this lane is known to have
                # completed successfully — a failed lane must not leak them.
                buffered.append(evt)

            try:
                provider = self.make_provider(provider_id, model)
                self._active_provider = provider
                try:
                    provider.stream_chat(messages, tools, wrapped, **kwargs)
                finally:
                    self._active_provider = None
                if had_error and not had_text:
                    # the lane errored before producing any answer TEXT — pure
                    # thinking included, since committing it would end the turn
                    # mid-reasoning ("stops in the thinking part") when the
                    # lane could simply be retried or rotated instead
                    self._rotate_lane(provider)
                    if _is_context_overflow_message(error_message):
                        raise ContextOverflowError(f"{provider_id}: {error_message or 'context overflow'}")
                    if _is_rate_limit_message(error_message):
                        raise RateLimitError(f"{provider_id}: {error_message or 'rate limited'}")
                    raise ProviderError(
                        f"{provider_id}: {error_message or 'error response'}",
                        retryable=True,
                    )
                if not had_output:
                    # The lane "succeeded" but produced nothing usable — Zen's
                    # flaky upstream dying mid-handshake looks exactly like
                    # this (finish_reason='network_error', zero events).
                    # Like the official client (which never retries empty
                    # turns), don't burn a visible same-lane retry here: move
                    # on to the next lane silently. Only the last lane raises,
                    # so a single-lane setup still surfaces instead of hanging.
                    self._rotate_lane(provider)
                    if not locked and index < len(lanes) - 1:
                        errors.append(f"{provider_id}: empty response")
                        last_reason = "empty response"
                        saw_other = True
                        continue
                    raise ProviderError(f"{provider_id}: empty response", retryable=True)
                for evt in buffered:
                    on_event(evt)
                if provider_id == "opencode":
                    # Only lanes that actually respond stay pinned/visible:
                    # a real answer outweighs any probe result.
                    try:
                        pin_zen_model_alive(model)
                    except Exception:
                        pass
                if index > 0 and on_notice:
                    on_notice(provider_id, model, last_reason or "provider error")
                return provider_id, model
            except ContextOverflowError as e:
                # History overflowed the window — every lane shares the same
                # oversized history, so rotating would just hit the same wall.
                # Propagate so the caller (agent loop) trims history and retries.
                raise
            except RateLimitError as e:
                if had_text:
                    # Visible ANSWER TEXT already reached the screen, then the
                    # lane hit a limit. Failover would glue the backup's answer
                    # onto the partial text (duplicate response), so commit and
                    # surface. Reasoning-only partials fall through and rotate.
                    raise ProviderError(
                        f"{provider_id}: {e}\n\n(partial answer already shown —"
                        " the reply was cut off by a rate limit)",
                        retryable=False,
                    ) from e
                errors.append(f"{provider_id}: rate limited ({e})")
                saw_rate_limit = True
                last_reason = "rate limited"
                if locked:
                    # locked: never leave the user's selected model
                    raise
                continue
            except ProviderError as e:
                if getattr(e, "partial", False):
                    # already a clear "reply cut off after partial output" — the
                    # live-streamed text stays; never re-wrap or rotate
                    raise
                # Only permanently broken lanes (bad model id / bad key / dead
                # endpoint) rotate. Transient failures (timeout, 5xx, overload,
                # empty reply) must NOT silently move the user off the model
                # they picked, especially the primary lane.
                if had_text:
                    # Visible ANSWER TEXT already reached the screen, then the
                    # lane failed. Retrying or rotating would duplicate the
                    # partial answer, so commit: keep the partial text and
                    # surface the real cause. A lane that only streamed THINKING
                    # falls through — the agent loop's auto-retry ("keep going")
                    # or the next lane takes over instead of the turn dying
                    # silently mid-thought (the reported "stops in the thinking
                    # part" on x-preview-f-free).
                    raise ProviderError(
                        f"{provider_id}: {e}\n\n(partial answer already shown —"
                        " the reply was cut off)",
                        retryable=False,
                    ) from e
                if e.retryable:
                    if index == 0:
                        # the user's chosen model hiccuped — surface the real
                        # cause instead of routing them elsewhere. Keep
                        # retryable=True so the agent loop's own backoff retry
                        # (auto_retry_count) can wait it out on the same model.
                        raise ProviderError(
                            f"{provider_id}: {e}\n\nHint: this looks like a"
                            " temporary issue — add another provider to your"
                            " 'rotation' list to fail over, or wait and retry.",
                            retryable=True,
                        )
                    # a backup lane that's temporarily down: skip it, the next
                    # one may still answer
                    errors.append(f"{provider_id}: {e}")
                    last_reason = (e.message or str(e))[:120]
                    saw_other = True
                    continue
                # hard (non-retryable) error: model id gone, bad key, dead
                # endpoint — this lane can never answer, rotate on
                errors.append(f"{provider_id}: {e}")
                last_reason = (e.message or str(e))[:120]
                if locked:
                    # locked: never leave the user's selected model
                    raise
                continue
        message = (
            "all providers failed:\n" + "\n".join(errors)
            + "\n\nHint: add another provider to your 'rotation' list to fail over,"
            + " or wait and retry."
        )
        if saw_rate_limit and not saw_other:
            # every failure was a rate limit -> surface as a retryable rate-limit error
            raise RateLimitError(message)
        raise ProviderError(message)


def build_provider(
    cfg: Config,
    provider_id: str | None = None,
    model: str | None = None,
    auth=None,
    session_id: str | None = None,
) -> Any:
    """Build a provider instance from config + auth."""
    explicit_provider = provider_id is not None
    provider_id = provider_id or cfg.provider
    model = model or cfg.model
    if not explicit_provider and "/" in model:
        # 'provider/model' shorthand only when the provider isn't already known;
        # avoids breaking model ids that legitimately contain '/' (e.g. OpenRouter).
        provider_id, model = model.split("/", 1)
    elif explicit_provider and model.startswith(provider_id + "/"):
        # Strip the 'provider/' prefix already folded in by _parse_model; it must
        # not be sent to the API (e.g. Zen rejects "opencode/deepseek-v4-flash-free").
        model = model.split("/", 1)[1]
    key = auth.get(provider_id) if auth else None

    timeout = build_read_timeout(getattr(cfg, "model_read_timeout", None), model)
    # Per-model reasoning effort (official variant parity): only sent when the
    # level is valid for THIS model per the live catalog — future models with
    # new levels work with zero code changes; fixed thinkers get nothing.
    effort: str | None = None
    try:
        want = str(getattr(cfg, "reasoning_effort", "") or "").strip().lower()
        if want and want in model_effort_levels(model, provider_id):
            effort = want
    except Exception:
        effort = None

    if provider_id == "opencode":
        from .zen import ZenProvider

        return ZenProvider(api_key=key, model=model, timeout=timeout, session_id=session_id,
                         reasoning_effort=effort) if effort else ZenProvider(api_key=key, model=model, timeout=timeout, session_id=session_id)
    if provider_id == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(model=model, timeout=timeout)
    if provider_id == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(api_key=key, model=model, timeout=timeout,
                                reasoning_effort=effort) if effort else AnthropicProvider(api_key=key, model=model, timeout=timeout)
    if provider_id == "openai":
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            id="openai",
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key=key,
            model=model,
            is_free=False,
            timeout=timeout,
            reasoning_effort=effort,
        )
    if provider_id in FREE_PROVIDERS:
        from .openai_compat import OpenAICompatProvider

        info = FREE_PROVIDERS[provider_id]
        return OpenAICompatProvider(
            id=provider_id,
            name=info["name"],
            base_url=info["base_url"],
            api_key=key,
            model=model,
            is_free=True,
            extra_headers=info.get("headers", {}),
            timeout=timeout,
            reasoning_effort=effort,
        )
    if provider_id in PAID_PROVIDERS:
        from .openai_compat import OpenAICompatProvider

        info = PAID_PROVIDERS[provider_id]
        return OpenAICompatProvider(
            id=provider_id,
            name=info["name"],
            base_url=info["base_url"],
            api_key=key,
            model=model,
            is_free=False,
            extra_headers=info.get("headers", {}),
            timeout=timeout,
            reasoning_effort=effort,
        )
    # custom provider from config providers.<id>
    custom = cfg.providers.get(provider_id)
    if custom and isinstance(custom, dict):
        from .openai_compat import OpenAICompatProvider

        base_url = custom.get("base_url") or custom.get("api")
        api_key = custom.get("api_key") or key
        if not base_url:
            raise ProviderError(f"provider {provider_id}: no base_url configured")
        return OpenAICompatProvider(
            id=provider_id,
            name=custom.get("name", provider_id),
            base_url=base_url,
            api_key=api_key,
            model=model,
            extra_headers=custom.get("headers", {}) or {},
            timeout=timeout,
            reasoning_effort=effort,
        )
    raise ProviderError(f"unknown provider: {provider_id}")


def _has_openrouter_key(auth) -> bool:
    """True if the user has an OpenRouter API key (env or auth.json)."""
    import os

    if os.environ.get("OPENROUTER_API_KEY"):
        return True
    return auth is not None and bool(auth.get("openrouter"))


def model_effort_levels(model_id: str, provider_id: str = "opencode") -> list[str]:
    """Reasoning-effort levels a model accepts, from the live catalog.

    Reads the model's ``reasoning_options`` (``{type: effort, values: [...]}``)
    so present AND future models work with zero code changes: a new model
    advertising ``[low, high, max]`` is honored the day it lands. Models with
    no effort option (fixed thinkers like big-pickle/nemotron, toggles,
    budget-tokens) return []. Paid models work identically — options come
    from the same catalog record. Never raises.
    """
    try:
        bare = str(model_id).split("/", 1)[-1]
    except Exception:
        return []
    try:
        catalog = (fetch_catalog().get(provider_id) or {}).get("models", {})
        entry = catalog.get(bare) or catalog.get(str(model_id)) or {}
        for opt in entry.get("reasoning_options") or []:
            if isinstance(opt, dict) and opt.get("type") == "effort":
                vals = [str(v) for v in (opt.get("values") or []) if str(v)]
                seen: list[str] = []
                for v in vals:
                    if v not in seen:
                        seen.append(v)
                return seen
    except Exception:
        pass
    return []


def effort_payload_fragment(model_id: str, effort: str, provider_id: str = "opencode") -> dict[str, Any]:
    """Provider-native request fragment for an effort level.

    Mirrors official opencode's variant mapping (provider/transform.ts):
    OpenAI-family (Responses + chat) takes ``reasoning: {effort}`` /
    ``reasoningEffort``; Anthropic takes adaptive ``thinking`` + effort;
    Google takes ``thinkingLevel``; Zen's gateway translates the OpenAI shape
    to whatever the upstream needs. Unknown models fall back to the OpenAI
    shape (harmless: gateways ignore unknown fields, strict ones get it only
    when the catalog advertised the level). Never raises.
    """
    effort = str(effort or "").strip().lower()
    if not effort:
        return {}
    try:
        bare = str(model_id).split("/", 1)[-1].lower()
    except Exception:
        bare = ""
    if "claude" in bare or provider_id == "anthropic":
        return {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": effort}
    if "gemini" in bare or "gemma" in bare or provider_id == "google":
        return {"thinkingLevel": effort, "includeThoughts": True}
    # Responses API shape (official zen path sends reasoning.effort)
    return {"reasoning": {"effort": effort}, "reasoningEffort": effort}


def _capability_score(model: dict) -> tuple:
    """Power ranking for free Zen models, most capable first.

    Sort key (tuple = lexicographic priority):
      1. alive first (a dead giant helps nobody)
      2. tool_call (agentic work needs tools)
      3. reasoning (coding models reason)
      4. attachment (images/pdf for real tasks)
      5. structured_output (reliable tool args)
      6. modalities breadth (text+image+audio+video > text-only)
      7. context + 4x output (raw window power)
      8. release_date (newer wins ties)
    Higher tuple wins; caller sorts reverse=True. Never raises.
    """
    raw = model.get("_raw") or {}
    limit = raw.get("limit") or {}
    modalities = raw.get("modalities") or {}
    in_mod = modalities.get("input") or []

    def _num(*vals: Any) -> float:
        for v in vals:
            try:
                return float(v or 0)
            except (TypeError, ValueError):
                continue
        return 0.0

    ctx = _num(limit.get("context"), model.get("context"))
    out = _num(limit.get("output"), model.get("output"))
    return (
        bool(model.get("alive", True)),
        bool(raw.get("tool_call", True)),
        bool(raw.get("reasoning", False)),
        bool(raw.get("attachment", False)),
        bool(raw.get("structured_output", False)),
        len(in_mod) if isinstance(in_mod, list) else 0,
        ctx + out * 4.0,
        str(raw.get("release_date") or model.get("release_date") or ""),
    )


def live_opencode_lanes(provider: str = "opencode") -> list[dict]:
    """Failover lanes from the LIVE models list, most powerful first.

    Takes every FREE opencodezen model (same source the picker uses:
    catalog -> endpoint -> bundle), asks the DEFAULT catalog record how
    capable it is (tools, reasoning, attachments, modalities, window), and
    sorts most-powerful first. Dead (probe-red) models are dropped; removed
    upstream vanishes; new upstream appears. Your picked model is NOT
    special-cased here — build_rotation keeps it lane 0 and appends these
    behind it. Never raises: falls back to the bundled list on any failure.
    """
    try:
        models = fetch_zen_models()
    except Exception:
        models = []
    live = [
        m for m in (models or [])
        if m.get("free") and m.get("status", "active") == "active"
        and m.get("alive", True)
    ]
    if not live:
        try:
            from .zen import FREE_MODELS

            live = [dict(m, free=True, alive=True) for m in FREE_MODELS]
        except Exception:
            live = []
    live = sorted(live, key=_capability_score, reverse=True)
    seen: set[str] = set()
    lanes: list[dict] = []
    for m in live:
        mid = str(m.get("id") or "")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        lanes.append({"provider": provider, "model": mid})
    return lanes


def build_rotation(cfg: Config, auth=None, session_id: str | None = None) -> Rotation:
    """Build a rotation with the picked model as the primary lane.

    - An explicit `rotation` list in config is failover order, not a veto:
      the picked model is always lane 0.
    - Stale lanes (model removed upstream) are dropped, so rotation never
      gets stuck on ghosts.
    - Failover behind the pick comes from the LIVE models list, best-capable
      first — new models appear automatically, removed ones vanish.
    - If the user has an OpenRouter API key, the OpenRouter default free model
      is added as a final failover lane.
    """
    lanes = list(cfg.rotation)

    # The model the user PICKED is always lane 0 — an explicit `rotation` list
    # in config is failover order, not a veto over the picker selection.
    selection = {"provider": cfg.provider, "model": cfg.model}

    def _bare(model_id: str) -> str:
        return str(model_id).split("/", 1)[-1]

    def _same(lane: dict, sel: dict) -> bool:
        return (
            lane.get("provider") == sel["provider"]
            and _bare(lane.get("model", "")) == _bare(sel["model"])
        )

    if lanes and not any(_same(l, selection) for l in lanes):
        lanes.insert(0, dict(selection))

    # Drop lanes whose model no longer exists upstream (catalog source of
    # truth): a removed id would just burn a turn on a guaranteed error.
    # Covers BOTH "opencode" and legacy "zen" provider keys.
    try:
        catalog_ids = {
            mid
            for mid, m in (fetch_catalog().get("opencode") or {}).get("models", {}).items()
            if m.get("status", "active") == "active"
        }
        if catalog_ids:
            lanes = [
                l
                for l in lanes
                if l.get("provider") not in ("opencode", "zen")
                or _bare(l.get("model", "")) in catalog_ids
                or "/" in str(l.get("model", ""))
            ]
    except Exception:
        pass

    if not lanes:
        lanes = [dict(selection)]
    else:
        lanes = [l for l in lanes if l]

    if not any(_same(l, selection) for l in lanes):
        lanes.insert(0, dict(selection))

    if _has_openrouter_key(auth) and cfg.provider != "openrouter" and not cfg.rotation:
        lanes.append({"provider": "openrouter", "model": FREE_DEFAULT_MODELS["openrouter"]})

    if len(lanes) == 1 and cfg.provider in ("opencode", "zen"):
        # Failover from the LIVE list, best-capable first — never the frozen
        # bundle, never stale config ghosts.
        current = cfg.model.split("/", 1)[-1]
        lanes += [
            lane for lane in live_opencode_lanes(provider=cfg.provider)
            if _bare(lane["model"]) != current and not _same(lane, selection)
        ]
    rotation = Rotation(
        lanes,
        lambda pid, m: build_provider(cfg, pid, m, auth, rotation.session_id),
        session_id=session_id,
    )
    return rotation


# -- models.dev catalog (mirrors official opencode's ModelsDev service) ------
# The official client lists models from the models.dev catalog
# (https://models.opencode.ai/api.json), NOT from a provider /models endpoint:
# names, context limits, pricing and add/removes all come from it. It caches
# the raw api.json on disk (atomic write), treats it as fresh for 5 minutes,
# and re-fetches hourly in the background so models added or removed upstream
# appear/disappear automatically. Env overrides match opencode:
#   OPENCODE_MODELS_URL          alternate catalog source
#   OPENCODE_MODELS_PATH         read the catalog from this file instead
#   OPENCODE_DISABLE_MODELS_FETCH=1  never fetch; use cache/bundled only

_MODELS_SOURCE = "https://models.opencode.ai"
_MODELS_FALLBACK_SOURCE = "https://models.dev/api.json"
_CATALOG_FRESH_SECONDS = 5 * 60
_CATALOG_REFRESH_SECONDS = 60 * 60

_catalog_refresher_lock = threading.Lock()
_catalog_refresher_started = False
# Wake event for the refresher: fetch_catalog() sets it instead of fetching
# synchronously, so a stale catalog never blocks the calling thread (which
# used to be the TUI's UI thread at end of turn — a hard screen freeze).
_catalog_kick = threading.Event()


def _catalog_cache_file():
    from ..globals import Path as GPath

    source = os.environ.get("OPENCODE_MODELS_URL")
    if source and source != _MODELS_SOURCE:
        import hashlib

        return GPath.cache / f"models-catalog-{hashlib.sha1(source.encode()).hexdigest()[:8]}.json"
    override = os.environ.get("OPENCODE_MODELS_PATH")
    if override:
        from pathlib import Path as _P

        return _P(override)
    return GPath.catalog_file()


def _catalog_fresh(path) -> bool:
    import time

    try:
        return time.time() - path.stat().st_mtime < _CATALOG_FRESH_SECONDS
    except OSError:
        return False


def _fetch_catalog_text() -> str | None:
    urls = []
    override = os.environ.get("OPENCODE_MODELS_URL")
    if override:
        urls.append(override.rstrip("/") + "/api.json")
    else:
        urls.extend([f"{_MODELS_SOURCE}/api.json", _MODELS_FALLBACK_SOURCE])
    for url in urls:
        try:
            import httpx

            resp = httpx.get(
                url,
                headers={"User-Agent": "opencode/latest/0.1.0/cli"},
                timeout=10,
                follow_redirects=True,
            )
            if resp.status_code == 200 and resp.text.strip():
                return resp.text
        except Exception:
            continue
    return None


def _write_catalog_atomic(path, text: str) -> None:
    """tmp-file + rename, like the official client's fetchAndWrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{id(text)}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _start_catalog_refresher() -> None:
    """Hourly background refresh, mirroring opencode's Schedule.spaced(60m)."""
    global _catalog_refresher_started

    if os.environ.get("OPENCODE_DISABLE_MODELS_FETCH"):
        return
    with _catalog_refresher_lock:
        if _catalog_refresher_started:
            return
        _catalog_refresher_started = True

    def _loop() -> None:
        while True:
            _catalog_kick.clear()
            if not _catalog_readonly():
                try:
                    path = _catalog_cache_file()
                    text = _fetch_catalog_text()
                    if text is not None and text != _read_catalog_text(path):
                        _write_catalog_atomic(path, text)
                except Exception:
                    pass
            _catalog_kick.wait(_CATALOG_REFRESH_SECONDS)

    threading.Thread(target=_loop, name="opencode_py-models-refresh", daemon=True).start()


def _read_catalog_text(path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _catalog_readonly() -> bool:
    """True when the catalog source is a user-pinned local file
    (OPENCODE_MODELS_PATH): serve it read-only — the hourly background
    refresher used to OVERWRITE the pinned file with network data."""
    return bool(os.environ.get("OPENCODE_MODELS_PATH"))


def fetch_catalog() -> dict:
    """The models.dev provider catalog {provider_id: {..., models: {...}}}.

    Cache-first with a 5-minute freshness TTL (stale data is still served when
    the network is down — listing never blocks or fails); a daemon thread
    re-fetches hourly so live changes propagate while the app runs.
    """
    path = _catalog_cache_file()

    if not os.environ.get("OPENCODE_DISABLE_MODELS_FETCH") and not _catalog_readonly():
        # NEVER top up synchronously: a stale cache used to trigger blocking
        # httpx calls right here, freezing whatever thread called this — the
        # TUI's UI thread at turn end (rebuild_rotation) being the worst case
        # (full connect timeout x2 URLs). Serve stale immediately and kick the
        # background refresher instead; fresh data lands seconds later with
        # nobody blocking.
        if path.exists() and not _catalog_fresh(path):
            _catalog_kick.set()
        _start_catalog_refresher()

    text = _read_catalog_text(path)
    if text:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


def refresh_catalog_sync() -> bool:
    """Synchronously pull the models.dev catalog and cache it (official parity).

    Like the official client's fetchAndWrite: new models appear, removed ones
    vanish. Blocking by design — call only from a worker thread (the model
    picker's fetch worker), never the UI/engine thread. Respects
    OPENCODE_DISABLE_MODELS_FETCH / OPENCODE_MODELS_PATH. Returns True when
    the cache was updated.
    """
    if os.environ.get("OPENCODE_DISABLE_MODELS_FETCH") or _catalog_readonly():
        return False
    try:
        path = _catalog_cache_file()
        text = _fetch_catalog_text()
        if text is not None and text != _read_catalog_text(path):
            try:
                data = json.loads(text)
                if not isinstance(data, dict):
                    return False
            except json.JSONDecodeError:
                return False
            _write_catalog_atomic(path, text)
            return True
    except Exception:
        pass
    return False


# -- zen model health (so /models only lists models that actually answer) ----
# The catalog lists every free-tier model, but several need a Zen account key
# or are currently down upstream — picking them just yields 400/401/503.
# A tiny "are you alive" probe per model (1 token, parallel) filters the list;
# results are cached for HEALTH_TTL_SECONDS so refreshes stay cheap.

HEALTH_TTL_SECONDS = 30 * 60


def _probe_headers(session: str | None = None) -> dict:
    import os

    # Per-model session: Zen pins a session id to ONE upstream lane, so all
    # probes sharing "health" would hammer the same lane and misreport every
    # other model. A stable per-model id keeps probes independent.
    sid = session or "health"
    headers = {
        "User-Agent": "opencode/latest/0.1.0/cli",
        "x-opencode-client": "cli",
        "x-opencode-project": "opencode_py",
        "x-opencode-request": sid,
    }
    # Authenticate exactly like ZenProvider does for keyless free lanes:
    # without this the gateway could 401/403 the anonymous ping and mark a
    # model dead that answers fine in real chat. Env key first, then the
    # stored auth.json key (contributor-free models need a real key — probing
    # anonymous "public" alone false-negatives them).
    key = os.environ.get("OPENCODE_API_KEY") or _stored_opencode_key()
    headers["Authorization"] = f"Bearer {key or 'public'}"
    headers["x-opencode-session"] = sid
    return headers


def _stored_opencode_key() -> str | None:
    """Best-effort read of the stored opencode key (auth.json), if any."""
    try:
        from ..auth import Auth
        from ..globals import Path as GPath

        return Auth(auth_file=GPath.auth_file()).get("opencode")
    except Exception:
        return None


def pin_zen_model_alive(model_id: str) -> None:
    """Pin a model as working because it actually responded in real chat.

    Probes can false-negative (anonymous key, dead probe lane, transient 500)
    for models that answer fine with the real session/key — so a lane that
    produces real output is pinned here and stays visible in /models forever
    (until removed from the catalog). Never raises.
    """
    if not model_id:
        return
    try:
        import time
        from pathlib import Path

        from ..globals import Path as GPath

        cache_path = Path(GPath.cache) / "model-health.json"
        data = _load_health_cache(cache_path)
        pinned: dict = dict(data.get("pinned") or {})
        health: dict = dict(data.get("health") or {})
        bare = str(model_id).split("/", 1)[-1]
        pinned[bare] = time.time()
        health[bare] = True
        fails = data.get("fails") or {}
        if isinstance(fails, dict):
            fails.pop(bare, None)
            data["fails"] = fails
        data["pinned"] = pinned
        data["health"] = health
        data["ts"] = data.get("ts") or time.time()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def _probe_zen_model(model_id: str) -> bool:
    """True when the gateway actually serves this model right now.

    Probes the model's preferred transport first (Responses by default, like
    the official client — some models only answer there, others only on Chat
    Completions), falling back to the other API before calling it dead. 200 =
    alive; 429 = alive but throttled (don't hide a working model during a
    burst); anything else (401 no-key / 400 / 503 dead upstream) = not usable
    on either transport. Transport errors propagate so the health layer can
    fail open on a total outage.
    """
    from .responses import (
        TransportIncompatible,
        get_preferred_endpoint,
        probe_responses,
        set_preferred_endpoint,
    )

    pref = get_preferred_endpoint(model_id)
    order = [pref, "chat" if pref == "responses" else "responses"]
    seen: set[str] = set()
    for ep in order:
        if ep in seen:
            continue
        seen.add(ep)
        try:
            if ep == "responses":
                ok = probe_responses(model_id, _probe_headers(session=f"health-{model_id}"))
            else:
                ok = _probe_chat_model(model_id)
        except TransportIncompatible:
            continue
        if ok:
            set_preferred_endpoint(model_id, ep)
            return True
    return False


def _probe_chat_model(model_id: str) -> bool:
    """Legacy Chat Completions probe (one leg of _probe_zen_model)."""
    try:
        import httpx

        resp = httpx.post(
            f"{ZEN_CHAT_URL}/chat/completions",
            headers=_probe_headers(session=f"health-{model_id}"),
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=12,
        )
        return resp.status_code in (200, 429)
    except Exception:
        raise


def _load_health_cache(cache_path) -> dict:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def check_zen_model_health(model_ids: list[str], ttl_seconds: int = HEALTH_TTL_SECONDS) -> dict[str, bool]:
    """{model_id: alive} with an on-disk cache so /models stays fast.

    Like the official client the catalog is the source of truth for which
    models exist (added upstream → appear after a catalog refresh; removed or
    deprecated → dropped). On top, free models must actually respond:

    - a model that answered (probe or real chat) is ``pinned`` and reported
      alive without probing while the cache is fresh;
    - once the window goes stale every listed model is re-probed, pins
      included — a pin only survives actual answers;
    - consecutive probe failures are counted: the first one is tolerated (a
      transient blip must not hide a working model), the second in a row
      unpins and hides it. Any real answer re-pins immediately via
      :func:`pin_zen_model_alive`;
    - a pin disappears as soon as the model vanishes from the passed-in list
      (the Zen catalog removed it).

    Fail-open: if every probe errored (network down, gateway unreachable),
    report everything alive rather than emptying the picker.
    """
    import time
    from pathlib import Path

    from ..globals import Path as GPath

    cache_path = Path(GPath.cache) / "model-health.json"
    data = _load_health_cache(cache_path)
    pinned: dict[str, float] = {
        k: v for k, v in (data.get("pinned") or {}).items() if isinstance(v, (int, float))
    }
    fails: dict[str, int] = {
        k: int(v) for k, v in (data.get("fails") or {}).items() if isinstance(v, int) and v > 0
    }
    cached: dict = data.get("health", {}) or {}
    fresh = time.time() - data.get("ts", 0) < ttl_seconds

    known = set(model_ids)
    for mid in [m for m in pinned if m not in known]:
        del pinned[mid]  # the provider removed it from the catalog — unpin
    for mid in [m for m in fails if m not in known]:
        del fails[mid]

    results: dict[str, bool] = {}
    to_probe: list[str] = []
    for mid in model_ids:
        if fresh and pinned.get(mid):
            results[mid] = True  # answered before and window still fresh
        elif not fresh or mid not in cached:
            to_probe.append(mid)  # unknown, stale window, or stale pin: probe now
        else:
            results[mid] = bool(cached.get(mid))

    probed: dict[str, bool] = {}
    if to_probe:
        try:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = {ex.submit(_probe_zen_model, mid): mid for mid in to_probe}
                for fut, mid in futures.items():
                    try:
                        probed[mid] = bool(fut.result())
                    except Exception:
                        # transport error (DNS/TLS/connection) — not a verdict
                        # on the model; the fail-open below keeps it visible
                        probed[mid] = False
        except Exception:
            probed = {mid: True for mid in to_probe}

        # fail-open: nothing proven alive (network down, gateway unreachable)
        # -> don't trust the failures, keep everything visible WITHOUT counting
        # strikes or pinning (an outage must neither hide nor pin anything).
        if to_probe and not any(probed.values()):
            for mid in to_probe:
                results[mid] = True
        else:
            for mid, ok in probed.items():
                if ok:
                    results[mid] = True
                    fails.pop(mid, None)
                    if mid not in pinned:
                        pinned[mid] = time.time()  # responded -> pinned
                else:
                    strikes = fails.get(mid, 0) + 1
                    if strikes >= 2 or mid not in pinned:
                        # dead: never-pinned models hide at once; pinned ones get
                        # one tolerated blip, then unpin on the second strike.
                        results[mid] = False
                        fails.pop(mid, None)
                        pinned.pop(mid, None)
                    else:
                        fails[mid] = strikes
                        results[mid] = True  # first strike: keep showing

    merged = dict(results)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"ts": time.time(), "health": merged, "pinned": pinned, "fails": fails}),
            encoding="utf-8",
        )
    except OSError:
        pass
    return merged


ZEN_CHAT_URL = "https://opencode.ai/zen/v1"


def _zen_models_from_catalog(catalog: dict) -> list[dict]:
    """Flatten the catalog's `opencode` provider entry into model dicts.

    Keeps the full capability record in ``_raw`` so power ranking can see
    tools/reasoning/attachments/modalities (not just ctx/out).
    """
    models = (catalog.get("opencode") or {}).get("models") or {}
    out: list[dict] = []
    for mid, m in models.items():
        limit = m.get("limit") or {}
        cost = m.get("cost") or {}
        free = cost.get("input", -1) == 0 and cost.get("output", -1) == 0
        out.append(
            {
                "id": mid,
                "name": m.get("name") or mid,
                "context": int(limit.get("context") or 0),
                "output": int(limit.get("output") or 0),
                "free": bool(free),
                "status": m.get("status", "active"),
                "release_date": m.get("release_date") or "",
                "_raw": {
                    "limit": dict(limit),
                    "modalities": dict(m.get("modalities") or {}),
                    "tool_call": m.get("tool_call"),
                    "reasoning": m.get("reasoning"),
                    "attachment": m.get("attachment"),
                    "structured_output": m.get("structured_output"),
                    "release_date": m.get("release_date") or "",
                },
            }
        )
    return out


def fetch_zen_models(cache_file=None, ttl_hours: int = 1) -> list[dict]:
    """OpenCode Zen model list, exactly like the official client.

    Primary source: the models.dev catalog (names like "Big Pickle", real
    context limits, $0 pricing → FREE badge). Every active catalog entry is
    listed — none skipped. Models added or removed upstream show up/vanish on
    their own via the cached catalog + sync refresh on picker open + hourly
    background refresh. Within the list, free models that actually respond
    (probe/pin health) sort first. Fallbacks: Zen's own /models endpoint,
    then the bundled free-model list.
    """
    models = _zen_models_from_catalog(fetch_catalog())
    if models:
        # Official dialog-model.tsx: filter deprecated, disable (hide) -nano.
        # Everything else active is listed — even free models whose probe is
        # currently red (they sort below the responding ones, never hidden).
        models = [m for m in models if m.get("status", "active") != "deprecated"]
        models = [m for m in models if "-nano" not in m.get("id", "")]
        # Liveness only orders the list (responding free first); paid ones
        # just need a key, so they aren't probed.
        alive: set[str] = set()
        free_ids = [m["id"] for m in models if m["free"]]
        if free_ids:
            try:
                health = check_zen_model_health(free_ids)
                alive = {mid for mid, ok in health.items() if ok}
            except Exception:
                alive = set()
        for m in models:
            m["_alive"] = (not m["free"]) or (m["id"] in alive) or (not alive and not free_ids)
            # _raw already carries the full catalog capability record from
            # _zen_models_from_catalog — leave it intact for power ranking.
        return _normalize_models(models)

    models = _fetch_zen_endpoint_models(cache_file, ttl_hours)
    if models:
        return models

    from .zen import FREE_MODELS

    return list(FREE_MODELS)


def _fetch_zen_endpoint_models(cache_file=None, ttl_hours: int = 1) -> list[dict]:
    """Fallback: https://opencode.ai/zen/v1/models with a cached fallback."""
    import time
    from pathlib import Path

    from ..globals import Path as GPath

    cache_file = cache_file or GPath.models_file()
    cache_path = Path(cache_file)
    models: list[dict] = []

    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            age = time.time() - data.get("ts", 0)
            models = data.get("models", [])
            if age < ttl_hours * 3600 and models:
                return _normalize_models(models)
        except (OSError, json.JSONDecodeError):
            pass

    try:
        import httpx

        from .zen import ZEN_BASE_URL

        resp = httpx.get(f"{ZEN_BASE_URL}/models", timeout=10, follow_redirects=True)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", []) if isinstance(data, dict) else data
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"ts": time.time(), "models": models}), encoding="utf-8")
            return _normalize_models(models)
    except Exception:
        pass

    return []


def fetch_openrouter_models() -> list[dict]:
    """Live-fetch OpenRouter's public model list filtered to free models.

    Returns [{id, name, context, free, provider='openrouter'}] sorted by id.
    The chosen default free model is always included even if the live list is
    unavailable or it happens to be missing.
    """
    import time

    out: list[dict] = []
    default = FREE_DEFAULT_MODELS["openrouter"]
    seen: set[str] = set()
    live_ok = False

    try:
        import httpx

        resp = httpx.get("https://openrouter.ai/api/v1/models", timeout=10, follow_redirects=True)
        if resp.status_code == 200:
            live_ok = True
            data = resp.json()
            raw = data.get("data", []) if isinstance(data, dict) else data
            for m in raw:
                mid = m.get("id", "")
                if not mid.endswith(":free"):
                    continue
                seen.add(mid)
                out.append(
                    {
                        "id": mid,
                        "name": m.get("name") or mid,
                        "context": (m.get("context_length") or 0) // 1000,
                        "free": True,
                        "provider": "openrouter",
                    }
                )
    except Exception:
        pass

    # Only show the bundled default when the live list actually responded —
    # otherwise we'd display a model that didn't respond (hide dead lanes).
    if live_ok and default not in seen:
        out.append(
            {
                "id": default,
                "name": default,
                "context": (1000 if "550b" in default else 0),
                "free": True,
                "provider": "openrouter",
            }
        )

    return sorted(out, key=lambda d: d["id"])


def fetch_live_models(
    provider_id: str,
    api_key: str | None = None,
    base_url: str | None = None,
    api_kind: str = "openai",
) -> list[dict]:
    """Live-fetch a provider's `GET /models` list.

    Handles OpenAI-compatible endpoints (Bearer auth) and Anthropic
    (x-api-key + anthropic-version). Returns
    ``[{id, name, context, free, provider}]`` (``free=False``) or ``[]`` when
    there is no key, the endpoint fails, or nothing usable comes back. The
    free-focused fetchers for opencode/openrouter are handled separately.
    """
    if not api_key:
        return []
    base_url = (
        base_url
        or FREE_PROVIDERS.get(provider_id, {}).get("base_url")
        or PAID_PROVIDERS.get(provider_id, {}).get("base_url")
    )
    if not base_url:
        return []
    headers = (
        {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        if api_kind == "anthropic"
        else {"Authorization": f"Bearer {api_key}"}
    )
    try:
        import httpx

        resp = httpx.get(base_url.rstrip("/") + "/models", headers=headers, timeout=8, follow_redirects=True)
        if resp.status_code != 200:
            return []
        data = resp.json()
        raw = data.get("data", []) if isinstance(data, dict) else (data or [])
        out: list[dict] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            mid = m.get("id") or ""
            if not mid:
                continue
            if provider_id == "openai" and not (
                mid.startswith("gpt-") or mid.startswith("o") or mid.startswith("chatgpt-")
            ):
                continue
            if provider_id == "anthropic" and not mid.startswith("claude-"):
                continue
            if provider_id == "deepseek" and mid not in ("deepseek-chat", "deepseek-reasoner"):
                continue
            ctx = m.get("context_length") or m.get("context") or 0
            try:
                ctx = int(ctx)
            except (TypeError, ValueError):
                ctx = 0
            out.append(
                {
                    "id": mid,
                    "name": m.get("display_name") or m.get("name") or mid,
                    "context": ctx,
                    "free": False,
                    "provider": provider_id,
                }
            )
        return out
    except Exception:
        return []


_context_cache: dict[tuple[str, str], int] = {}

# Per-provider model list (id + real context), fetched once so the context-window
# lookup hits the network at most once per provider per process.
_model_list_cache: dict[str, list[dict]] = {}

# Known context windows for the default free-provider models (failover lanes).
# Used when a provider's live /models list isn't available.
KNOWN_CONTEXT: dict[str, int] = {
    "opencode": 131072,
    "groq": 131072,
    "cerebras": 131072,
    "google": 1048576,
    "openrouter": 1000000,
    "nvidia": 1000000,
    "mistral": 256000,
    "github": 128000,
    "sambanova": 131072,
    "togetherai": 131072,
    "anthropic": 200000,
    "openai": 128000,
    "deepseek": 128000,
    "xai": 256000,
    "deepinfra": 128000,
    "ollama": 131072,
}

# Exact context windows for specific paid models (documented by the vendor).
# `OPencode /models` and the paid `/models` endpoints rarely report a context
# window, so these prevent the TUI from showing a 0% or a per-provider guess.
# Explicitly documented sizes only — no invented numbers.
_KNOWN_CONTEXTS_BY_MODEL: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "o1": 200000,
    "o1-mini": 128000,
    "o3-mini": 200000,
    "o3": 200000,
    "claude-3-7-sonnet-20250219": 200000,
    "claude-3-7-sonnet-latest": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-sonnet-4-20250514": 200000,
    "deepseek-chat": 65536,
    "deepseek-reasoner": 65536,
    "deepseek-v3": 65536,
    "deepseek-v3.2": 65536,
    "deepseek-coder": 65536,
    "grok-2": 131072,
    "grok-2-1212": 131072,
    "grok-beta": 131072,
    "grok-3": 256000,
    "grok-3-mini": 256000,
    "nemotron-3-ultra-550b-a55b:free": 1000000,
}


def _known_context(provider_id: str, model_id: str) -> int:
    """Context size from the bundled Zen free-model list (opencode only)."""
    if provider_id == "opencode":
        from .zen import FREE_MODELS

        for m in FREE_MODELS:
            if m["id"] == _bare_model_id(provider_id, model_id):
                return int(m.get("context") or 0)
    # A hardcoded per-provider guess is never the *selected model's* real
    # window, so it must NOT short-circuit the live provider lookup below.
    return 0


def _provider_model_list(provider_id: str, auth=None) -> list[dict]:
    """Real model list (id + context) for a provider, fetched once and cached.

    Returns the provider's live `/models` data so the status-bar percentage is
    the REAL context window of the selected model (mirrors opencode, which
    derives the percentage from `model.limit.context`). opencode/openrouter get
    their dedicated fetchers; other providers (openai, anthropic, the free
    BYOK providers, custom ones) use `fetch_live_models` with the configured
    API key. Falls back to the bundled free-model list when no key exists.
    """
    cached = _model_list_cache.get(provider_id)
    if cached is not None:
        return cached
    models: list[dict] = []
    if provider_id == "opencode":
        # Context-size lookups only need ids/windows, NOT liveness — skip
        # fetch_zen_models()'s health probes here (12s timeout per model,
        # previously run on the ENGINE thread at every usage event, stalling
        # the moment the stream finished). Raw catalog + bundled fallback.
        models = _normalize_models(_zen_models_from_catalog(fetch_catalog()))
        if not models:
            from .zen import FREE_MODELS

            models = list(FREE_MODELS)
    elif provider_id == "openrouter":
        # fetch_openrouter_models reports context in thousands; normalize to tokens
        models = [
            {**m, "context": int(m.get("context") or 0) * 1000}
            for m in fetch_openrouter_models()
        ]
    else:
        meta = FREE_PROVIDERS.get(provider_id) or PAID_PROVIDERS.get(provider_id) or {}
        key = auth.get(provider_id) if auth else None
        if meta:
            models = fetch_live_models(
                provider_id,
                key,
                meta.get("base_url"),
                meta.get("api_kind", "openai"),
            )
    _model_list_cache[provider_id] = models or []
    return models or []


def _bare_model_id(provider_id: str, model_id: str) -> str:
    """Strip a folded `provider/` prefix so lookup matches the bare model id."""
    if model_id.startswith(provider_id + "/"):
        return model_id.split("/", 1)[1]
    return model_id


def model_context_size(provider_id: str, model_id: str, auth=None) -> int:
    """Real context-window size (in tokens) for the selected provider/model.

    Resolution order: bundled known sizes (no network, opencode free models)
    -> the provider's live model list (uses the configured API key, caches the
    list once) -> 0 when genuinely unknown (the UI then omits the percentage).
    Returns the selected model's actual context, not a hardcoded per-provider
    guess, so the TUI's `12,345 (6%)` is the truth for openai/openrouter/etc.
    """
    key = (provider_id, model_id)
    if key in _context_cache:
        return _context_cache[key]
    bare = _bare_model_id(provider_id, model_id)
    # opencode bundled free models are exact and need no network
    size = _known_context(provider_id, model_id)
    if not size:
        # real per-model window from the provider's model list
        for m in _provider_model_list(provider_id, auth):
            if m.get("id") == bare and m.get("context"):
                size = int(m["context"])
                break
    if not size:
        # documented per-model windows (vendor-published, not guesses) for
        # providers whose /models endpoint doesn't report a context window
        size = _KNOWN_CONTEXTS_BY_MODEL.get(bare, 0)
    if not size:
        # last resort: a provider-wide default (only for the bundled free
        # providers whose whole catalogue shares one documented window)
        size = KNOWN_CONTEXT.get(provider_id, 0)
    _context_cache[key] = size
    return size


def _known_output(provider_id: str, model_id: str) -> int:
    """Output-token limit from the bundled Zen free-model list."""
    if provider_id == "opencode":
        from .zen import FREE_MODELS

        for m in FREE_MODELS:
            if m["id"] == _bare_model_id(provider_id, model_id):
                return int(m.get("output") or 0)
    return 0


def model_output_limit(provider_id: str, model_id: str) -> int:
    """Best-effort max output tokens for a provider/model lane (0 when unknown)."""
    if not model_id:
        return 0
    return _known_output(provider_id, model_id)


def _normalize_models(raw: list[dict]) -> list[dict]:
    from .zen import FREE_MODELS

    fallback = {f["id"]: f for f in FREE_MODELS}
    out = []
    for m in raw:
        cost = m.get("cost") or {}
        limit = m.get("limit") or {}
        fb = fallback.get(m.get("id"), {})
        is_free = (
            (isinstance(cost.get("input"), (int, float)) and cost["input"] == 0)
            or m.get("id") in fallback
            or m.get("free") is True
            or str(m.get("id", "")).endswith("-free")
        )
        out.append(
            {
                "id": m.get("id", ""),
                "name": m.get("name") or fb.get("name") or m.get("id", ""),
                "context": limit.get("context", 0) or m.get("context", 0) or fb.get("context", 0),
                "output": limit.get("output", 0) or m.get("output", 0) or fb.get("output", 0),
                "free": bool(is_free),
                "status": m.get("status", "active"),
                "release_date": m.get("release_date") or fb.get("release_date") or "",
                "alive": bool(m.get("_alive", True)),
                "provider": "opencode",
                "_raw": dict(m.get("_raw") or {}),
            }
        )
    # Official sortModelOptions: Free first, then release_date desc, then title.
    # Responding free models (_alive) sort above the rest of the free tier so
    # the working ones are on top but none are ever hidden.
    # Stable multi-pass sort (last key first) to mix asc/desc correctly.
    out.sort(key=lambda x: str(x.get("name") or x["id"]).lower())
    out.sort(key=lambda x: str(x.get("release_date") or ""), reverse=True)
    out.sort(key=lambda x: (not x.get("alive", True)))
    out.sort(key=lambda x: (not x["free"]))
    return out


def sort_model_options(options: list[dict], newest_first: bool = False) -> list[dict]:
    """Python port of official sortModelOptions (dialog-model.tsx).

    Default: Free first, then release_date desc, then title.
    newest_first (single-provider view): release_date desc, then title.
    Options are dicts with optional free/release_date/name/id keys.
    """
    opts = list(options)
    opts.sort(key=lambda o: str(o.get("name") or o.get("title") or o.get("id") or "").lower())
    opts.sort(key=lambda o: str(o.get("release_date") or ""), reverse=True)
    if not newest_first:
        opts.sort(key=lambda o: (not bool(o.get("free") or o.get("footer") == "Free")))
    return opts


def check_provider(cfg: Config, auth) -> dict[str, Any]:
    """Ping a provider; used by --check. Returns status dict."""
    result: dict[str, Any] = {}
    lanes = cfg.rotation or [{"provider": cfg.provider, "model": cfg.model}]
    for lane in lanes:
        pid = lane.get("provider", "zen")
        model = lane.get("model", FREE_DEFAULT_MODELS.get(pid, FREE_DEFAULT_MODELS["zen"]))
        try:
            provider = build_provider(cfg, pid, model, auth)
            if not provider.api_key and pid != "opencode" and pid != "ollama":
                result[pid] = {"ok": False, "model": model, "error": "no API key (see /connect or env)"}
                continue
            # lightweight models list ping (GET /models) for openai-compat
            try:
                import httpx

                meta = FREE_PROVIDERS.get(pid) or PAID_PROVIDERS.get(pid) or {}
                api_kind = meta.get("api_kind", "openai")
                if api_kind == "anthropic":
                    # Anthropic's /models endpoint needs x-api-key +
                    # anthropic-version, not a Bearer token (Bearer => 401).
                    headers = {
                        "x-api-key": provider.api_key or "",
                        "anthropic-version": "2023-06-01",
                    }
                else:
                    headers = {"Authorization": f"Bearer {provider.api_key}"}
                resp = httpx.get(f"{provider.base_url}/models", headers=headers, timeout=10, follow_redirects=True)
                ok = resp.status_code == 200
                result[pid] = {"ok": ok, "model": model, "status": resp.status_code}
            except Exception as e:
                result[pid] = {"ok": False, "model": model, "error": str(e)}
        except Exception as e:
            result[pid] = {"ok": False, "model": model, "error": str(e)}
    return result
