"""Review provider backed by khonliang-dispatcher (MS-GPU-GATEWAY,
fr_reviewer_50a5b842).

Replaces direct HTTP access to the box's "internal" LLM stack —
TabbyAPI (the resident GPU engine) and Ollama (local models + Ollama
Cloud, all served through our own ollama daemon) — with
``dispatcher_lib.DispatcherClient``, so the dispatcher's VRAM lease
ledger / admission gate / drain rule finally see reviewer's traffic
instead of it competing for the shared GPU outside the gateway's
view. ``claude_cli`` (external Anthropic API, no local GPU/VRAM
footprint) is explicitly OUT of scope — confirmed with the
maintainer this session that external-LLM gatewaying is a reasonable
future FR, not required now.

Model resolution mirrors the existing ``_resolve_model`` convention
in ``reviewer/providers/ollama.py``/``tabbyapi.py`` EXACTLY, just
swapping the fallback: ``request.metadata["model"]`` is still the one
channel for "I know exactly which model I want" (an operator's A/B
comparison pin, or the rule table's explicit ``kimi-k2.5:cloud``
escalation in ``reviewer/rules/policy.py``) — when present, this
provider calls the dispatcher with ``model=``. When absent (today's
``FAST_TIER_MODEL`` empty-sentinel case — "whatever's resident on
this box"), it calls with ``skill=request.kind`` instead: the
dispatcher's skill/model/engine resolver (fr_dispatcher_94a483f6) is
exactly the box-specific-model-name problem the sentinel existed to
paper over, now solved server-side via the operator's
``skill_policy.yaml`` instead of reviewer's local config.

One class, constructed twice (once per legacy backend name it
replaces) so `ReviewResult.backend` / `UsageEvent.backend` keep
reporting `"tabbyapi"` / `"ollama"` — existing usage/cost dashboards
and the pricing table key on those strings and don't need to change.
Both instances may share one underlying `DispatcherClient` (stateless
HTTP client, no reason to duplicate the connection pool).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from dispatcher_lib import (
    Busy,
    Cancelled,
    ChatTask,
    DeadlineExceeded,
    DispatcherClient,
    ModelUnavailable,
    TaskInvalid,
)

from khonliang_reviewer import (
    ReviewFinding,
    ReviewProvider,
    ReviewRequest,
    ReviewResult,
    UsageEvent,
)

from reviewer.providers._prompt import (
    binary_questions_active,
    build_review_prompt,
    parse_verdicts,
    validate_verdict_coverage,
)
from reviewer.providers.ollama import _suggest_num_ctx

logger = logging.getLogger(__name__)

#: Maps this repo's legacy backend names to the dispatcher's own
#: engine-kind string (``dispatcher/agent.py``'s ``kind: ollama`` /
#: ``kind: tabby`` config values) -- used by
#: :meth:`DispatcherProvider.is_available` to find the right entry in
#: ``GET /v1/engines``'s inventory.
_NAME_TO_ENGINE_KIND = {"tabbyapi": "tabby", "ollama": "ollama"}


@dataclass
class DispatcherProviderConfig:
    """Construction-time configuration for :class:`DispatcherProvider`."""

    #: khonliang-dispatcher's HTTP base (its own default is 8790).
    base_url: str = "http://localhost:8790"
    #: Wall-clock budget handed to ``DispatcherClient.run(deadline_s=...)``.
    #: Deliberately more generous than the old providers' per-call
    #: ``timeout_seconds`` (TabbyAPI 180s / Ollama 300s): the dispatcher
    #: may queue this request behind other tenants or swap a model in
    #: before dispatching, budget the old direct-HTTP calls never had
    #: to account for.
    deadline_s: float = 240.0
    #: This INSTANCE's own pinned model (per-backend: the "ollama"
    #: DispatcherProvider and the "tabbyapi" one each get their own
    #: config with their own value here) -- restores exact backend-pin
    #: semantics from the old OllamaProviderConfig/TabbyAPIProviderConfig
    #: (codex review finding, round 2). Every existing call site that
    #: reaches this provider WITHOUT a ``request.metadata["model"]``
    #: override has already made a load-bearing choice of WHICH
    #: registry entry to invoke -- the rule table's default backend,
    #: the tabby-unavailable degrade-to-ollama reroute, or an explicit
    #: caller ``backend=`` pin with no model. None of those mean "any
    #: internal engine, don't care" -- they mean "this specific
    #: backend, whatever model it's pinned to." Sending bare
    #: ``skill=request.kind`` in that case would let the dispatcher's
    #: skill_policy.yaml resolve to EITHER internal engine, silently
    #: overriding the caller's backend choice. ``skill=`` is only used
    #: as the last-resort fallback when this is also unset.
    default_model: str = ""
    #: ``ollama`` instance only: operator-pinned ``num_ctx``, mirroring
    #: ``OllamaProviderConfig.num_ctx`` (codex review finding, round 3
    #: -- forwarded generation options were dropped entirely). ``None``
    #: falls through to the auto-bump heuristic (:func:`_suggest_num_ctx`,
    #: reused from ``reviewer.providers.ollama`` rather than
    #: reimplemented) the same way the direct provider does.
    num_ctx: int | None = None
    #: ``tabbyapi`` instance only: forwarded as OpenAI ``max_tokens``,
    #: mirroring ``TabbyAPIProviderConfig.max_tokens``.
    max_tokens: int = 4096
    #: ``tabbyapi`` instance only: send ``chat_template_kwargs:
    #: {"enable_thinking": false}`` (mirrors
    #: ``TabbyAPIProviderConfig.disable_thinking`` -- the switch that
    #: actually suppresses the Qwen3 ``<think>`` preamble; see
    #: ``tabbyapi.py``'s module docstring).
    disable_thinking: bool = True


def _build_engine_options(
    name: str,
    config: DispatcherProviderConfig,
    request: ReviewRequest,
    prompt: str,
) -> dict[str, Any]:
    """Forwarded generation options -- codex review finding (round 3):
    the initial cut built ``ChatTask`` with no ``options`` at all,
    silently dropping every backend-specific tuning knob the direct
    providers used to send.

    ``ChatTask.options`` reaches ``dispatcher_lib``'s wire body as the
    ``options`` submit field, which the dispatcher's executors consume
    differently per engine kind (``dispatcher/executors/ollama.py``
    nests it under ollama's own native ``options`` object;
    ``dispatcher/executors/tabby.py`` spreads it at the payload's top
    level, matching tabby's OpenAI-compatible surface) -- both are
    preserved exactly here.

    ONE known, accepted gap: Ollama's ``format="json"`` grammar
    constraint (``OllamaProviderConfig.format``, the switch that
    actually enforces JSON output on the native endpoint) has NO
    equivalent in the dispatcher's ``OllamaRequest`` -- it's a
    top-level sibling of ``options`` in ollama's own API, and the
    dispatcher wire contract only forwards ``options``. Gatewayed
    Ollama reviews lose that enforcement until the dispatcher's own
    contract grows a passthrough for it -- filed as
    fr_dispatcher_ba059d43, a dispatcher-side follow-on; not
    something reviewer can work around from its own request options.
    """
    if name == "ollama":
        caller_num_ctx = request.metadata.get("num_ctx")
        num_ctx = (
            caller_num_ctx
            if isinstance(caller_num_ctx, int)
            and not isinstance(caller_num_ctx, bool)
            and caller_num_ctx > 0
            else (config.num_ctx or _suggest_num_ctx(prompt))
        )
        return {"num_ctx": num_ctx} if num_ctx is not None else {}
    if name == "tabbyapi":
        options: dict[str, Any] = {"max_tokens": config.max_tokens}
        if config.disable_thinking:
            options["chat_template_kwargs"] = {"enable_thinking": False}
        return options
    return {}


class DispatcherProvider(ReviewProvider):
    """Review provider backed by ``dispatcher_lib.DispatcherClient``.

    ``name`` is an instance attribute (not the class-level default) so
    the same class can stand in for both the legacy ``"tabbyapi"`` and
    ``"ollama"`` registry keys without reporting the wrong backend
    label in usage records.
    """

    def __init__(
        self,
        name: str,
        config: DispatcherProviderConfig | None = None,
        *,
        client: DispatcherClient | None = None,
    ) -> None:
        self.name = name
        self.config = config or DispatcherProviderConfig()
        #: Injectable for tests; production builds a real client.
        self._client = client or DispatcherClient(
            base_url=self.config.base_url, caller="khonliang-reviewer",
        )

    async def is_available(self, timeout_s: float = 2.0) -> bool:
        """Cheap liveness probe -- restores the duck-typed hook
        ``TabbyAPIProvider``/``OllamaProvider`` used to expose (codex
        review finding: two existing call sites depend on it via
        ``hasattr``/``callable`` rather than an abstract method, so
        silently dropping it broke them without a type error).
        ``reviewer.agent``'s rule-table degrade-path calls this to
        reroute a ``tabbyapi`` pick to ``ollama`` when the resident
        engine can't serve; ``ProviderRegistry._check_availability``
        would call the sync analogue (``is_provisioned``) for the
        cheap ``list_models`` probe -- NOT implemented here, since
        there's no local, network-free signal to check anymore (the
        api-key-presence check it used to run doesn't apply once the
        dispatcher owns credentials). That cheap probe intentionally
        degrades to "assume available" for ``tabbyapi`` now, same
        posture ``ollama`` already has in that function.

        Queries the dispatcher's ``GET /v1/engines`` and reports True
        iff at least one engine of this provider's mapped kind
        (``tabbyapi`` -> dispatcher engine kind ``"tabby"``; ``ollama``
        -> ``"ollama"``) is currently available. Any transport/parse
        failure (dispatcher unreachable, malformed body) reports
        False -- same "don't route here" posture the old providers'
        probes used.
        """
        engine_kind = _NAME_TO_ENGINE_KIND.get(self.name, self.name)
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as http:
                response = await http.get(
                    f"{self.config.base_url.rstrip('/')}/v1/engines"
                )
                response.raise_for_status()
                body = response.json()
        except Exception as exc:  # noqa: BLE001 — any probe failure means "don't route here"
            logger.debug(
                "dispatcher availability probe failed for %s: %s",
                self.name, exc,
            )
            return False
        engines = body.get("engines") if isinstance(body, dict) else None
        if not isinstance(engines, list):
            return False
        return any(
            isinstance(e, dict)
            and e.get("engine") == engine_kind
            and e.get("available")
            for e in engines
        )

    async def review(self, request: ReviewRequest) -> ReviewResult:
        repo_prompts = request.metadata.get("_khonliang_repo_prompts")
        example_format = request.metadata.get("_khonliang_example_format")
        region_sweep = request.metadata.get("_khonliang_region_sweep") is True
        binary_questions = (
            request.metadata.get("_khonliang_binary_questions") is True
        )
        prompt = build_review_prompt(
            request,
            include_schema=True,
            repo_prompts=repo_prompts,
            example_format=example_format if isinstance(example_format, str) else None,
            region_sweep=region_sweep,
            binary_questions=binary_questions,
        )
        started_wall = time.time()
        started_mono = time.monotonic()

        # Same resolution convention as the providers this replaces
        # (ollama.py/tabbyapi.py's ``_resolve_model``): an explicit
        # ``request.metadata["model"]`` -- an operator's A/B-comparison
        # pin, or the rule table's explicit kimi-k2.5:cloud escalation
        # -- always wins. Absent, fall back to THIS instance's own
        # ``config.default_model`` (codex review finding, round 2):
        # every call reaching this provider without an override has
        # already made a load-bearing choice of which backend to use
        # (rule-table default, the tabby-unavailable degrade-to-ollama
        # reroute, an explicit ``backend=`` pin) -- that choice must
        # stay pinned to THIS engine, not be handed to the dispatcher's
        # skill resolver to redecide. ``skill=request.kind`` is only
        # the last-resort fallback when even the per-instance default
        # is unset (an operator relying entirely on skill_policy.yaml).
        override = request.metadata.get("model")
        model_override = (
            override.strip()
            if isinstance(override, str) and override.strip()
            else (self.config.default_model or None)
        )

        run_kwargs: dict[str, Any] = {
            "task": ChatTask(
                messages=[{"role": "user", "content": prompt}],
                options=_build_engine_options(self.name, self.config, request, prompt),
            ),
            "deadline_s": self.config.deadline_s,
        }
        if model_override is not None:
            run_kwargs["model"] = model_override
            run_kwargs["role"] = request.kind
        else:
            run_kwargs["skill"] = request.kind

        # DispatcherClient.run() is SYNCHRONOUS and blocking -- it holds
        # the caller's deadline_s via time.sleep() between retries
        # (dispatcher_lib.client, PR4 v0). Calling it directly here
        # would block this coroutine's entire event loop for up to
        # deadline_s seconds, freezing every other concurrent review /
        # bus handler this agent is running. Run it in a thread so only
        # this one review's coroutine waits.
        handle = await asyncio.to_thread(lambda: self._client.run(**run_kwargs))
        try:
            result = handle.result()
        except TaskInvalid as exc:
            return _errored(
                request,
                error=f"dispatcher rejected the request as malformed: {exc}",
                error_category="malformed_envelope",
                model=model_override or "",
                backend=self.name,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )
        except ModelUnavailable as exc:
            return _errored(
                request,
                error=f"dispatcher: no engine serves this request: {exc}",
                error_category="backend_error",
                model=model_override or "",
                backend=self.name,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )
        except DeadlineExceeded as exc:
            return _errored(
                request,
                error=f"dispatcher deadline exceeded: {exc}",
                error_category="backend_timeout",
                model=model_override or "",
                backend=self.name,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )
        except Busy as exc:
            return _errored(
                request,
                error=f"dispatcher shed the request (busy): {exc}",
                error_category="backend_error",
                model=model_override or "",
                backend=self.name,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )
        except Cancelled as exc:
            return _errored(
                request,
                error=f"dispatcher: request cancelled by another caller: {exc}",
                error_category="backend_error",
                model=model_override or "",
                backend=self.name,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )

        result_out = _parse_result(
            result.text,
            request=request,
            model=result.envelope.get("model") or model_override or "",
            backend=self.name,
            started_wall=started_wall,
            duration_ms=_elapsed_ms(started_mono),
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
        )
        logger.debug(
            "dispatcher review done: backend=%s disposition=%s category=%s "
            "model=%s tokens_in=%s tokens_out=%s duration_ms=%s",
            self.name,
            result_out.disposition,
            result_out.error_category or "-",
            result_out.model,
            result_out.usage.input_tokens if result_out.usage else 0,
            result_out.usage.output_tokens if result_out.usage else 0,
            result_out.usage.duration_ms if result_out.usage else 0,
        )
        return result_out


#: Same think-block + markdown-fence cleanup as
#: ``reviewer/providers/tabbyapi.py`` -- the resident model behind
#: the dispatcher is the same Qwen3-family reasoning model, with the
#: same quirks (leading ``<think>...</think>`` preamble, JSON fenced
#: in ```` ```json ```` blocks even with a schema in the prompt).
_THINK_BLOCK_RE = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE_RE = re.compile(r"\A\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*\Z", re.DOTALL)


def _clean_content(content: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", content, count=1)
    fence = _FENCE_RE.match(cleaned)
    if fence:
        cleaned = fence.group(1)
    return cleaned.strip()


def _parse_result(
    text: str,
    *,
    request: ReviewRequest,
    model: str,
    backend: str,
    started_wall: float,
    duration_ms: int,
    tokens_in: int | None,
    tokens_out: int | None,
) -> ReviewResult:
    """Translate the dispatcher's plain-text result into a
    :class:`ReviewResult` -- the dispatcher already did the
    HTTP/transport/JSON-envelope work (that's `ChatResult.text`);
    this only re-does the model-response-shape parsing the old
    providers did on their own ``choices[0].message.content``."""
    if not text:
        return _errored(
            request,
            error="dispatcher result contained no text",
            error_category="malformed_envelope",
            model=model,
            backend=backend,
            started_wall=started_wall,
            duration_ms=duration_ms,
        )

    try:
        payload = json.loads(_clean_content(text))
    except json.JSONDecodeError as exc:
        return _errored(
            request,
            error=f"dispatcher result content is not JSON: {exc}",
            error_category="malformed_envelope",
            model=model,
            backend=backend,
            started_wall=started_wall,
            duration_ms=duration_ms,
        )

    if not isinstance(payload, dict):
        return _errored(
            request,
            error=(
                "dispatcher result content is not a JSON object "
                f"(type={type(payload).__name__})"
            ),
            error_category="malformed_envelope",
            model=model,
            backend=backend,
            started_wall=started_wall,
            duration_ms=duration_ms,
        )

    summary = str(payload.get("summary", ""))
    raw_findings = payload.get("findings") or []
    if not isinstance(raw_findings, list):
        raw_findings = []
    findings = [
        ReviewFinding(
            severity=item.get("severity", "comment"),
            title=str(item.get("title", "")),
            body=str(item.get("body", "")),
            category=str(item.get("category", "")),
            path=item.get("path"),
            line=_int_or_none(item.get("line")),
            suggestion=item.get("suggestion"),
            section=item.get("section") if isinstance(item.get("section"), str) else None,
        )
        for item in raw_findings
        if isinstance(item, dict)
    ]

    verdicts = (
        parse_verdicts(payload) if binary_questions_active(request) else []
    )
    if binary_questions_active(request):
        coverage_error = validate_verdict_coverage(verdicts)
        if coverage_error:
            return _errored(
                request,
                error=(
                    "dispatcher result failed binary-questions contract: "
                    f"{coverage_error}"
                ),
                error_category="malformed_envelope",
                model=model,
                backend=backend,
                started_wall=started_wall,
                duration_ms=duration_ms,
            )

    usage = UsageEvent(
        timestamp=started_wall,
        backend=backend,
        model=model,
        input_tokens=_safe_int(tokens_in),
        output_tokens=_safe_int(tokens_out),
        cache_read_tokens=0,
        cache_creation_tokens=0,
        duration_ms=duration_ms,
        disposition="posted",
        request_id=request.request_id,
        repo=str(request.metadata.get("repo", "")),
        pr_number=_int_or_none(request.metadata.get("pr_number")),
        # Local resident engine (gatewayed) -- zero marginal cost, same
        # as the direct-HTTP providers this replaces.
        estimated_api_cost_usd=0.0,
    )

    return ReviewResult(
        request_id=request.request_id,
        summary=summary,
        findings=findings,
        verdicts=verdicts,
        disposition="posted",
        usage=usage,
        backend=backend,
        model=model,
        created_at=started_wall,
    )


def _errored(
    request: ReviewRequest,
    *,
    error: str,
    error_category: str,
    model: str,
    backend: str,
    started_wall: float,
    duration_ms: int,
) -> ReviewResult:
    usage = UsageEvent(
        timestamp=started_wall,
        backend=backend,
        model=model,
        duration_ms=duration_ms,
        disposition="errored",
        request_id=request.request_id,
        repo=str(request.metadata.get("repo", "")),
        pr_number=_int_or_none(request.metadata.get("pr_number")),
        estimated_api_cost_usd=0.0,
        error=error,
        error_category=error_category,
    )
    return ReviewResult(
        request_id=request.request_id,
        summary="",
        findings=[],
        disposition="errored",
        error=error,
        error_category=error_category,
        usage=usage,
        backend=backend,
        model=model,
        created_at=started_wall,
    )


def _elapsed_ms(started_mono: float) -> int:
    return int((time.monotonic() - started_mono) * 1000)


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


__all__ = ["DispatcherProvider", "DispatcherProviderConfig"]
