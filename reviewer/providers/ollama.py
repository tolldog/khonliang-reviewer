"""Ollama review provider via the OpenAI-compatible endpoint.

Hits Ollama's ``/v1/chat/completions`` endpoint with the ``openai`` SDK
so any Ollama-served model — local (``qwen3.5``, ``glm-4.7-flash``) or
cloud (``kimi-k2.5:cloud``, ``glm-5:cloud``) — can drive a review from
the same provider. Model choice is per-request via
``request.metadata["model"]`` with a config-level fallback.

Unlike Claude-via-CLI, Ollama does not return a cost-per-call in its
response. ``UsageEvent.estimated_api_cost_usd`` is populated later from
a ``ModelPricing`` table (WU5 territory); WU3 leaves it at 0.0 so the
field is still consistent with the library contract.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import openai

from khonliang_reviewer import (
    ReviewFinding,
    ReviewProvider,
    ReviewRequest,
    ReviewResult,
    UsageEvent,
)

from reviewer.providers._prompt import REVIEW_RESPONSE_SCHEMA, build_review_prompt


logger = logging.getLogger(__name__)


class OllamaHealthcheckError(RuntimeError):
    """Raised when :meth:`OllamaProvider.healthcheck` cannot reach the endpoint.

    Distinct from generic :class:`RuntimeError` so agent boot can catch
    the reachability-specific case without confusing it with auth or
    model-missing problems.
    """


@dataclass
class OllamaProviderConfig:
    """Construction-time configuration for :class:`OllamaProvider`."""

    base_url: str = "http://localhost:11434/v1"
    #: Ollama ignores the key for local models but the openai SDK requires
    #: something. Use a descriptive placeholder so leaked logs are
    #: obviously-local, not mistaken for a real key.
    api_key: str = "ollama"
    default_model: str = "qwen3.5"
    timeout_seconds: float = 300.0


class OllamaProvider(ReviewProvider):
    """Review provider backed by Ollama's OpenAI-compatible endpoint.

    Accepts a client override for testing; production construction
    builds the client from :class:`OllamaProviderConfig`.
    """

    name = "ollama"

    def __init__(
        self,
        config: OllamaProviderConfig | None = None,
        *,
        client: Any | None = None,
    ):
        self.config = config or OllamaProviderConfig()
        self._client = client or openai.AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
        )

    async def healthcheck(self) -> None:
        """Verify the endpoint is reachable and speaks the OpenAI-compat API.

        Hits ``models.list`` which Ollama implements as a lightweight
        "what models are available here" query. Raises on any transport
        or auth failure.
        """
        try:
            await self._client.models.list()
        except openai.APIError as exc:
            raise OllamaHealthcheckError(
                f"ollama endpoint not reachable at {self.config.base_url}: {exc}"
            ) from exc

    async def review(self, request: ReviewRequest) -> ReviewResult:
        prompt = build_review_prompt(request, include_schema=True)
        model = _resolve_model(request, self.config.default_model)
        started_wall = time.time()
        started_mono = time.monotonic()

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=self.config.timeout_seconds,
            )
        except openai.APIConnectionError as exc:
            return _errored(
                request,
                error=f"ollama endpoint unreachable: {exc}",
                error_category="backend_error",
                model=model,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )
        except openai.APITimeoutError:
            return _errored(
                request,
                error=f"ollama request timed out after {self.config.timeout_seconds}s",
                error_category="backend_error",
                model=model,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )
        except openai.AuthenticationError as exc:
            return _errored(
                request,
                error=f"ollama rejected credentials: {exc}",
                error_category="auth_not_provisioned",
                model=model,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )
        except openai.NotFoundError as exc:
            return _errored(
                request,
                error=f"ollama reported model or route not found: {exc}",
                error_category="backend_error",
                model=model,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )
        except openai.APIError as exc:
            return _errored(
                request,
                error=f"ollama API error: {exc}",
                error_category="backend_error",
                model=model,
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )

        duration_ms = _elapsed_ms(started_mono)
        result = _parse_response(
            response,
            request=request,
            model=model,
            started_wall=started_wall,
            duration_ms=duration_ms,
        )
        logger.debug(
            "ollama review done: disposition=%s category=%s model=%s tokens_in=%s tokens_out=%s duration_ms=%s",
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
    if isinstance(override, str) and override:
        return override
    return default


def _parse_response(
    response: Any,
    *,
    request: ReviewRequest,
    model: str,
    started_wall: float,
    duration_ms: int,
) -> ReviewResult:
    """Translate a chat-completions response into a :class:`ReviewResult`."""
    content = _extract_message_content(response)
    if content is None:
        return _errored(
            request,
            error="ollama response contained no message content",
            error_category="malformed_envelope",
            model=model,
            started_wall=started_wall,
            duration_ms=duration_ms,
            response=response,
        )

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return _errored(
            request,
            error=f"ollama response content is not JSON: {exc}",
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
                "ollama response content is not a JSON object "
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
        )
        for item in raw_findings
        if isinstance(item, dict)
    ]

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
        disposition="posted",
        usage=usage,
        backend=OllamaProvider.name,
        model=model,
        created_at=started_wall,
    )


def _extract_message_content(response: Any) -> str | None:
    """Pull the first choice's message content. Tolerates dict or SDK object."""
    choices = _get(response, "choices", [])
    if not choices:
        return None
    first = choices[0]
    message = _get(first, "message", None)
    if message is None:
        return None
    content = _get(message, "content", None)
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
    usage_raw = _get(response, "usage", None)
    if not isinstance(usage_raw, dict):
        usage_raw = _usage_to_dict(usage_raw)

    return UsageEvent(
        timestamp=started_wall,
        backend=OllamaProvider.name,
        model=model,
        input_tokens=_safe_int(usage_raw.get("prompt_tokens")),
        output_tokens=_safe_int(usage_raw.get("completion_tokens")),
        # Ollama does not report cache tokens through the OpenAI-compat
        # surface. Left at 0; pricing layer will cope.
        cache_read_tokens=0,
        cache_creation_tokens=0,
        duration_ms=duration_ms,
        disposition=disposition,  # type: ignore[arg-type]
        request_id=request.request_id,
        repo=str(request.metadata.get("repo", "")),
        pr_number=_int_or_none(request.metadata.get("pr_number")),
        # Ollama does not return per-call cost. WU5's model_pricing table
        # populates estimated_api_cost_usd for this backend.
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
    response: Any | None = None,
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
        backend=OllamaProvider.name,
        model=model,
        created_at=started_wall,
    )


def _get(obj: Any, name: str, default: Any) -> Any:
    """Read ``name`` from either a dict or an attribute-style SDK object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _usage_to_dict(usage_obj: Any) -> dict[str, Any]:
    """Best-effort conversion of an SDK usage object to a plain dict.

    The ``openai`` SDK returns ``CompletionUsage`` pydantic models.
    Falling back to ``model_dump`` keeps both test mocks (plain dicts)
    and production responses (SDK models) on the same code path.
    """
    if usage_obj is None:
        return {}
    if hasattr(usage_obj, "model_dump"):
        try:
            dumped = usage_obj.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    # Last-ditch attribute sniffing — tolerant, not authoritative.
    result: dict[str, Any] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = getattr(usage_obj, field, None)
        if val is not None:
            result[field] = val
    return result


def _elapsed_ms(started_mono: float) -> int:
    return int((time.monotonic() - started_mono) * 1000)


def _int_or_none(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        pass
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int:
    result = _int_or_none(val)
    return 0 if result is None else result


__all__ = [
    "OllamaHealthcheckError",
    "OllamaProvider",
    "OllamaProviderConfig",
]
