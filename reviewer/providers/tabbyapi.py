"""TabbyAPI review provider via the OpenAI-compatible ``/v1/chat/completions``.

TabbyAPI (ExLlamaV3) is the box's resident GPU inference engine — a
systemd service holding the model weights in VRAM permanently. This
provider consolidates the hot-tier local review onto that engine
(fr_khonliang-reviewer_0e7ccff1): with TabbyAPI resident, the ollama
models no longer fit in VRAM and run CPU-bound into timeouts
(dog_d6895752), so the reviewer talks to the engine that actually owns
the GPU instead of competing with it.

Differences from :mod:`reviewer.providers.ollama`, all deliberate:

- **No ``num_ctx`` dance.** TabbyAPI's context window is fixed
  server-side (``max_seq_len`` in its config); a prompt that exceeds it
  fails loudly with an HTTP error rather than silently truncating, so
  the whole auto-bump heuristic is unnecessary here.
- **No cold-start retry.** The model is resident by definition — there
  is no eviction/cold-load path — and retry-on-timeout is exactly what
  compounded the timeout cascade in dog_d6895752. One attempt; a
  timeout returns ``backend_timeout`` immediately so the caller's
  degrade-to-skip path can act inside its own budget.
- **Thinking-model hygiene.** The resident model (Qwen3 family) is a
  reasoning model that prefixes responses with a ``<think>…</think>``
  block. Verified live 2026-07-02: ``chat_template_kwargs:
  {"enable_thinking": false}`` suppresses it cleanly; ``/no_think``
  leaves an empty think block; ``response_format`` is accepted but does
  NOT grammar-constrain output (the model still fences JSON in
  markdown). So the provider sends ``enable_thinking: false`` by
  default AND the parser defensively strips think blocks and markdown
  fences before ``json.loads``.
- **``usage`` may be ``null``.** The current TabbyAPI build returns
  ``"usage": null`` on non-streaming completions (verified live). The
  usage record tolerates that and falls back to zero token counts —
  honest zeros rather than invented estimates; the pricing layer treats
  the local engine as zero-cost anyway.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

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


logger = logging.getLogger(__name__)


class TabbyAPIHealthcheckError(RuntimeError):
    """Base for :meth:`TabbyAPIProvider.healthcheck` failures.

    Mirrors :class:`reviewer.providers.ollama.OllamaHealthcheckError` so
    agent boot can treat every backend's healthcheck failure uniformly
    while still telling auth problems apart via the subclass.
    """


class TabbyAPIAuthError(TabbyAPIHealthcheckError):
    """Raised when the endpoint rejects credentials during healthcheck."""


@dataclass
class TabbyAPIProviderConfig:
    """Construction-time configuration for :class:`TabbyAPIProvider`."""

    #: OpenAI-compat base. TabbyAPI serves the real OpenAI surface here —
    #: unlike ollama there is no native/compat split to work around.
    base_url: str = "http://localhost:5000/v1"
    #: Sent as ``Authorization: Bearer <api_key>``. TabbyAPI *requires*
    #: a key (401 without one). An explicit value here is an operator
    #: pin; when empty the provider consults its ``api_key_provider``
    #: callable (the agent wires ``credentials.get_tabbyapi_key``) on
    #: EVERY request, so engine-side key rotation is picked up without a
    #: restart. With neither, requests surface ``auth_not_provisioned``
    #: rather than crashing at boot.
    api_key: str = ""
    #: The model id TabbyAPI reports for its loaded model (e.g.
    #: ``Qwen3-14B-exl3-6bpw``). Box-specific — the deployed value lives
    #: in the local ``config.yaml``, not in code. TabbyAPI serves ONE
    #: resident model, so this is largely a label, but the OpenAI surface
    #: requires it and sending the wrong id errors loudly.
    default_model: str = ""
    #: GPU-resident inference is fast (the whole point of consolidation);
    #: 180s is generous for a 32k-context review and keeps the worst case
    #: well inside the caller's degrade-to-skip budget (dog_d6895752).
    timeout_seconds: float = 180.0
    #: Response-length cap forwarded as OpenAI ``max_tokens``. A
    #: structured review (summary + findings + optional verdicts) fits
    #: comfortably; the cap stops a runaway generation from burning the
    #: whole timeout budget.
    max_tokens: int = 4096
    #: Send ``chat_template_kwargs: {"enable_thinking": false}`` so the
    #: Qwen3-family resident model skips its ``<think>`` preamble
    #: (verified live: this is the switch that works; ``/no_think`` and
    #: ``response_format`` don't). Disable only for a non-thinking model
    #: whose chat template rejects the kwarg.
    disable_thinking: bool = True


class TabbyAPIProvider(ReviewProvider):
    """Review provider backed by TabbyAPI's OpenAI-compatible endpoint."""

    name = "tabbyapi"

    def __init__(
        self,
        config: TabbyAPIProviderConfig | None = None,
        *,
        http_client: Any | None = None,
        api_key_provider: Callable[[], str | None] | None = None,
    ):
        self.config = config or TabbyAPIProviderConfig()
        self._base = self.config.base_url.strip().rstrip("/")
        #: Re-run credential discovery on every request (PR codex P2):
        #: an explicit ``config.api_key`` pin always wins, otherwise this
        #: callable (the agent passes ``credentials.get_tabbyapi_key``) is
        #: consulted per call so an engine-side key rotation is picked up
        #: without a process restart — the contract the credentials
        #: module documents. Auth headers are therefore built per request
        #: rather than baked into the client at construction.
        self._api_key_provider = api_key_provider
        #: Injectable for tests (a fake exposing async ``post``/``get``
        #: returning objects with ``status_code`` / ``json()`` /
        #: ``raise_for_status()``); production builds a real httpx client.
        self._http = http_client or httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
        )

    def _resolve_api_key(self) -> str:
        """Config-pinned key wins; else discover fresh via the provider."""
        pinned = self.config.api_key
        if isinstance(pinned, str) and pinned.strip():
            return pinned.strip()
        if self._api_key_provider is not None:
            try:
                discovered = self._api_key_provider()
            except Exception as exc:  # noqa: BLE001 — discovery is best-effort
                logger.debug("tabbyapi key discovery failed: %s", exc)
                return ""
            if isinstance(discovered, str) and discovered.strip():
                return discovered.strip()
        return ""

    def is_provisioned(self) -> bool:
        """True when a usable API key is currently resolvable.

        The rule-table selection path uses this to degrade gracefully on
        boxes without the resident engine (no key, no engine) instead of
        deterministically failing every default-routed review with
        ``auth_not_provisioned``.
        """
        return bool(self._resolve_api_key())

    async def is_available(self, timeout_s: float = 2.0) -> bool:
        """Cheap liveness + auth gate for the rule-table selection path.

        False when no key is resolvable (box has no engine), the engine
        is stopped/hung/unreachable (codex round-2 P1), OR the resolved
        key is stale/rejected (codex round-3 P2 — an authenticated
        ``/v1/models`` GET validates reachability AND credentials in one
        probe; the unauthenticated ``/health`` route would pass a
        misprovisioned host straight into a guaranteed
        ``auth_not_provisioned`` on the review call). The probe is a
        local-loopback GET (~1-2ms when the engine is up;
        connection-refused fails in microseconds), so the per-review
        overhead is negligible next to inference.
        """
        if not self.is_provisioned():
            return False
        try:
            response = await self._http.get(
                f"{self._base}/models",
                timeout=timeout_s,
                headers=_auth_headers(self._resolve_api_key()),
            )
            return 200 <= int(response.status_code) < 300
        except Exception as exc:  # noqa: BLE001 — any probe failure means "don't route here"
            logger.debug("tabbyapi availability probe failed: %s", exc)
            return False

    async def healthcheck(self) -> None:
        """Verify the endpoint is reachable and credentials are accepted.

        ``/v1/models`` is TabbyAPI's cheapest authenticated read (it
        lists the single loaded model). Failure categories mirror the
        ollama healthcheck so agent boot handles all backends alike.
        """
        try:
            response = await self._http.get(
                f"{self._base}/models",
                headers=_auth_headers(self._resolve_api_key()),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                raise TabbyAPIAuthError(
                    f"tabbyapi endpoint at {self._base} rejected "
                    f"credentials (HTTP {status}): {exc}"
                ) from exc
            raise TabbyAPIHealthcheckError(
                f"tabbyapi healthcheck failed at {self._base} "
                f"(HTTP {status}): {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise TabbyAPIHealthcheckError(
                f"tabbyapi endpoint not reachable at {self._base}: {exc}"
            ) from exc

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
        model = _resolve_model(request, self.config.default_model)
        started_wall = time.time()
        started_mono = time.monotonic()

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.disable_thinking:
            # The switch that actually suppresses the Qwen3 <think>
            # preamble (verified live; see module docstring). Harmless
            # for templates without the kwarg — jinja ignores unused
            # template vars.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        # ONE attempt, no retry: the model is resident (no cold-start to
        # ride out) and a retry would double the worst-case latency past
        # the caller's degrade-to-skip budget — the dog_d6895752 cascade.
        try:
            http_response = await self._http.post(
                f"{self._base}/chat/completions",
                json=payload,
                timeout=self.config.timeout_seconds,
                headers=_auth_headers(self._resolve_api_key()),
            )
            http_response.raise_for_status()
            response = http_response.json()
        except httpx.TimeoutException:
            return _errored(
                request,
                error=(
                    f"tabbyapi request timed out after "
                    f"{self.config.timeout_seconds}s"
                ),
                # Same category as ollama's timeout so the caller-side
                # degrade-to-skip logic treats resident-engine timeouts
                # uniformly with the rest of the hot tier.
                error_category="backend_timeout",
                model=model,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                return _errored(
                    request,
                    error=f"tabbyapi rejected credentials (HTTP {status}): {exc}",
                    error_category="auth_not_provisioned",
                    model=model,
                    started_wall=started_wall,
                    duration_ms=_elapsed_ms(started_mono),
                )
            return _errored(
                request,
                error=f"tabbyapi returned HTTP {status}: {exc}",
                error_category="backend_error",
                model=model,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )
        except httpx.HTTPError as exc:
            return _errored(
                request,
                error=f"tabbyapi endpoint unreachable: {exc}",
                error_category="backend_error",
                model=model,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            return _errored(
                request,
                error=f"tabbyapi response body was not valid JSON: {exc}",
                error_category="malformed_envelope",
                model=model,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )

        result = _parse_response(
            response,
            request=request,
            model=model,
            started_wall=started_wall,
            duration_ms=_elapsed_ms(started_mono),
        )
        logger.debug(
            "tabbyapi review done: disposition=%s category=%s model=%s "
            "tokens_in=%s tokens_out=%s duration_ms=%s",
            result.disposition,
            result.error_category or "-",
            result.model,
            result.usage.input_tokens if result.usage else 0,
            result.usage.output_tokens if result.usage else 0,
            result.usage.duration_ms if result.usage else 0,
        )
        return result


def _resolve_model(request: ReviewRequest, default: str) -> str:
    """Use request-supplied model if present, else the provider default."""
    override = request.metadata.get("model")
    if isinstance(override, str) and override.strip():
        return override.strip()
    return default


def _auth_headers(api_key: str | None) -> dict[str, str]:
    """Build the ``Authorization`` header; empty/whitespace key sends none."""
    if isinstance(api_key, str) and api_key.strip():
        return {"Authorization": f"Bearer {api_key.strip()}"}
    return {}


#: A leading reasoning block from a thinking model. Anchored at the start
#: (after optional whitespace) so a legitimate ``<think>`` mention inside a
#: finding body never gets eaten; only the preamble form is stripped.
#: ``disable_thinking`` normally prevents these entirely — this is the
#: defensive layer for a per-call model override or a template that
#: ignores the kwarg (observed live: ``/no_think`` still yields an empty
#: ``<think>\n\n</think>`` block).
_THINK_BLOCK_RE = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)

#: A markdown-fenced body: the model wrapped its JSON in ```json … ```
#: (observed live even with ``response_format`` set — TabbyAPI accepts
#: the field but does not grammar-constrain decoding). Only a fence that
#: wraps the WHOLE remaining content is unwrapped.
_FENCE_RE = re.compile(r"\A\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*\Z", re.DOTALL)


def _clean_content(content: str) -> str:
    """Strip a leading think-block, then unwrap a whole-body JSON fence."""
    cleaned = _THINK_BLOCK_RE.sub("", content, count=1)
    fence = _FENCE_RE.match(cleaned)
    if fence:
        cleaned = fence.group(1)
    return cleaned.strip()


def _parse_response(
    response: Any,
    *,
    request: ReviewRequest,
    model: str,
    started_wall: float,
    duration_ms: int,
) -> ReviewResult:
    """Translate an OpenAI chat-completions response into a ReviewResult."""
    content = _extract_message_content(response)
    if content is None:
        return _errored(
            request,
            error="tabbyapi response contained no message content",
            error_category="malformed_envelope",
            model=model,
            started_wall=started_wall,
            duration_ms=duration_ms,
            response=response,
        )

    try:
        payload = json.loads(_clean_content(content))
    except json.JSONDecodeError as exc:
        return _errored(
            request,
            error=f"tabbyapi response content is not JSON: {exc}",
            error_category="malformed_envelope",
            model=model,
            started_wall=started_wall,
            duration_ms=duration_ms,
            response=response,
        )

    if not isinstance(payload, dict):
        return _errored(
            request,
            error=(
                "tabbyapi response content is not a JSON object "
                f"(type={type(payload).__name__})"
            ),
            error_category="malformed_envelope",
            model=model,
            started_wall=started_wall,
            duration_ms=duration_ms,
            response=response,
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

    # Same verdict scoping + coverage contract as every other backend:
    # verdicts surface only for opted-in binary-questions pr_diff reviews,
    # and incomplete coverage is a contract failure, not a partial score.
    verdicts = (
        parse_verdicts(payload) if binary_questions_active(request) else []
    )
    if binary_questions_active(request):
        coverage_error = validate_verdict_coverage(verdicts)
        if coverage_error:
            return _errored(
                request,
                error=(
                    "tabbyapi response failed binary-questions contract: "
                    f"{coverage_error}"
                ),
                error_category="malformed_envelope",
                model=model,
                started_wall=started_wall,
                duration_ms=duration_ms,
                response=response,
            )

    usage = _build_usage(
        response,
        request=request,
        model=model,
        started_wall=started_wall,
        duration_ms=duration_ms,
        disposition="posted",
    )

    return ReviewResult(
        request_id=request.request_id,
        summary=summary,
        findings=findings,
        verdicts=verdicts,
        disposition="posted",
        usage=usage,
        backend=TabbyAPIProvider.name,
        model=model,
        created_at=started_wall,
    )


def _extract_message_content(response: Any) -> str | None:
    """Pull the first choice's message content from a dict-shaped response."""
    if not isinstance(response, dict):
        return None
    choices = response.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    return None


def _build_usage(
    response: Any,
    *,
    request: ReviewRequest,
    model: str,
    started_wall: float,
    duration_ms: int,
    disposition: str,
    error: str = "",
    error_category: str = "",
) -> UsageEvent:
    # The current TabbyAPI build returns ``"usage": null`` on
    # non-streaming completions (verified live 2026-07-02). Tolerate any
    # non-dict shape and record honest zeros — the local engine is
    # zero-cost in the pricing table, so nothing downstream depends on
    # exact counts; duration_ms carries the latency signal.
    usage_raw = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage_raw, dict):
        usage_raw = {}

    return UsageEvent(
        timestamp=started_wall,
        backend=TabbyAPIProvider.name,
        model=model,
        input_tokens=_safe_int(usage_raw.get("prompt_tokens")),
        output_tokens=_safe_int(usage_raw.get("completion_tokens")),
        cache_read_tokens=0,
        cache_creation_tokens=0,
        duration_ms=duration_ms,
        disposition=disposition,  # type: ignore[arg-type]
        request_id=request.request_id,
        repo=str(request.metadata.get("repo", "")),
        pr_number=_int_or_none(request.metadata.get("pr_number")),
        # Local resident engine — zero marginal cost per call.
        estimated_api_cost_usd=0.0,
        error=error,
        error_category=error_category,
    )


def _errored(
    request: ReviewRequest,
    *,
    error: str,
    error_category: str,
    model: str,
    started_wall: float,
    duration_ms: int,
    response: Any = None,
) -> ReviewResult:
    usage = _build_usage(
        response,
        request=request,
        model=model,
        started_wall=started_wall,
        duration_ms=duration_ms,
        disposition="errored",
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
        backend=TabbyAPIProvider.name,
        model=model,
        created_at=started_wall,
    )


def _elapsed_ms(started_mono: float) -> int:
    return int((time.monotonic() - started_mono) * 1000)


def _safe_int(value: Any) -> int:
    """Coerce a usage count to int; anything unusable is 0."""
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


__all__ = [
    "TabbyAPIAuthError",
    "TabbyAPIHealthcheckError",
    "TabbyAPIProvider",
    "TabbyAPIProviderConfig",
]
