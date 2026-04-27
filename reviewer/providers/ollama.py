"""Ollama review provider via the OpenAI-compatible endpoint.

Hits Ollama's ``/v1/chat/completions`` endpoint with the ``openai`` SDK
so any Ollama-served model — local (``qwen2.5-coder:14b``, ``glm-4.7-flash``) or
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
import math
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

from reviewer.providers._prompt import build_review_prompt


logger = logging.getLogger(__name__)


class OllamaHealthcheckError(RuntimeError):
    """Base for :meth:`OllamaProvider.healthcheck` failures.

    Distinct from generic :class:`RuntimeError` so agent boot can catch
    all healthcheck problems broadly while still distinguishing specific
    cases via subclasses. Today's subclass is :class:`OllamaAuthError`
    for credential failures; reachability / generic failures raise the
    base class itself.
    """


class OllamaAuthError(OllamaHealthcheckError):
    """Raised when the endpoint rejects credentials during healthcheck.

    Separated from the general reachability case so agent boot can
    surface "wrong key / expired token" differently from "server
    unreachable"; callers that want either of them can still catch
    :class:`OllamaHealthcheckError`.
    """


@dataclass
class OllamaProviderConfig:
    """Construction-time configuration for :class:`OllamaProvider`."""

    base_url: str = "http://localhost:11434/v1"
    #: Ollama ignores the key for local models but the openai SDK requires
    #: something. Use a descriptive placeholder so leaked logs are
    #: obviously-local, not mistaken for a real key.
    api_key: str = "ollama"
    default_model: str = "qwen2.5-coder:14b"
    timeout_seconds: float = 300.0
    #: Operator-pinned ``num_ctx`` for every Ollama review unless the
    #: caller overrides via ``request.metadata["num_ctx"]``. ``None``
    #: (default) falls through to the auto-bump heuristic
    #: (:func:`_suggest_num_ctx`) so existing configs keep working
    #: unchanged. Useful for measurement runs that want a fixed
    #: context window so duration / token-count comparisons hold the
    #: ``num_ctx`` axis constant; the auto-bump heuristic varies with
    #: prompt size.
    num_ctx: int | None = None


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
        """Verify the endpoint is reachable and accepts our credentials.

        Hits ``models.list`` — Ollama implements it as a lightweight
        "what models are available here" query. Failures split into
        three categories so agent boot can act on each:

        - :class:`OllamaAuthError` — credentials rejected (cloud-hosted
          Ollama typically). Caller should fix the API key.
        - :class:`OllamaHealthcheckError` ``"not reachable"`` — transport
          couldn't reach the server. Caller should check it's running
          and the URL is right.
        - :class:`OllamaHealthcheckError` ``"healthcheck failed"`` —
          any other API error (bad request, rate limit, etc.). Caller
          reads the message and investigates.
        """
        try:
            await self._client.models.list()
        except openai.AuthenticationError as exc:
            raise OllamaAuthError(
                f"ollama endpoint at {self.config.base_url} rejected credentials: {exc}"
            ) from exc
        except openai.APIConnectionError as exc:
            raise OllamaHealthcheckError(
                f"ollama endpoint not reachable at {self.config.base_url}: {exc}"
            ) from exc
        except openai.APIError as exc:
            raise OllamaHealthcheckError(
                f"ollama healthcheck failed at {self.config.base_url}: {exc}"
            ) from exc

    async def review(self, request: ReviewRequest) -> ReviewResult:
        # ``_khonliang_repo_prompts`` / ``_khonliang_example_format`` are
        # in-process-only passthrough metadata (see reviewer.agent).
        # Absent keys or wrong types collapse to the pre-FR behavior —
        # a plain built-in prompt with no repo-side merge.
        repo_prompts = request.metadata.get("_khonliang_repo_prompts")
        example_format = request.metadata.get("_khonliang_example_format")
        prompt = build_review_prompt(
            request,
            include_schema=True,
            repo_prompts=repo_prompts,
            example_format=example_format if isinstance(example_format, str) else None,
        )
        model = _resolve_model(request, self.config.default_model)
        started_wall = time.time()
        started_mono = time.monotonic()

        # ``num_ctx`` resolution order (highest precedence wins):
        # 1. ``request.metadata["num_ctx"]`` — caller-supplied per-call
        #    override. Skips the auto-bump heuristic entirely; useful
        #    for measurement runs that want to hold ``num_ctx`` constant
        #    across calls regardless of prompt size.
        # 2. ``self.config.num_ctx`` — operator-pinned default for
        #    every review served by this provider instance. Mirrors
        #    the runtime kwarg so config-file pinning is supported.
        # 3. :func:`_suggest_num_ctx` — auto-bump based on prompt
        #    length. Without this, large diffs silently truncate at
        #    Ollama's 4096-token default and the model returns a
        #    near-empty response (bug_reviewer_663d0d62), which is
        #    indistinguishable from a clean approval.
        # 4. ``None`` — let Ollama apply the model's documented
        #    default. Only fires when (1)-(3) all return None.
        num_ctx_override = _resolve_num_ctx(request, self.config, prompt)
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "timeout": self.config.timeout_seconds,
        }
        if num_ctx_override is not None:
            # Ollama's OpenAI-compatible endpoint accepts native Ollama
            # options under ``options.num_ctx``. The ``openai`` SDK
            # forwards ``extra_body`` verbatim so this works without
            # depending on a future schema-flag for context window.
            create_kwargs["extra_body"] = {
                "options": {"num_ctx": num_ctx_override}
            }

        try:
            response = await self._client.chat.completions.create(**create_kwargs)
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
        # Truncation heuristic: a large input that produced a tiny
        # output is the silent-truncation signature observed in
        # bug_reviewer_663d0d62 (``output_tokens=8`` with
        # ``disposition=posted`` and zero findings on a ~500-line
        # diff). Surface a WARNING so operators can see the trap even
        # when the rule-table / num_ctx auto-sizing didn't catch it
        # (e.g. some other backend, off-by-one rounding, or a model
        # that genuinely produces sparse output for some reason). The
        # warning is informational — the result is still returned —
        # because heuristics shouldn't reject otherwise-valid reviews.
        if result.usage is not None:
            input_tokens = result.usage.input_tokens
            output_tokens = result.usage.output_tokens
            if (
                input_tokens > _TRUNCATION_INPUT_THRESHOLD
                and output_tokens < _TRUNCATION_OUTPUT_THRESHOLD
            ):
                logger.warning(
                    "ollama review may have been truncated: "
                    "input_tokens=%s output_tokens=%s num_ctx=%s model=%s — "
                    "low output relative to input is the silent-truncation "
                    "signature; verify num_ctx covers the prompt and that "
                    "the model can produce structured JSON at this size",
                    input_tokens,
                    output_tokens,
                    num_ctx_override or f"default({_NUM_CTX_DEFAULT})",
                    model,
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


# Threshold for the truncation-warning heuristic. ``input_tokens >
# _TRUNCATION_INPUT_THRESHOLD`` plus ``output_tokens <
# _TRUNCATION_OUTPUT_THRESHOLD`` is the signature of a review that ran
# but couldn't fit the prompt — Ollama silently truncates and the
# model returns a near-empty response. Tuned conservatively so a real
# clean review on a small diff (which legitimately produces few output
# tokens) doesn't fire the warning. 2000 input is well above any
# plausible "small diff". 32 output captures the silent-truncation
# signature observed in bug_reviewer_663d0d62 (8 output tokens on a
# 500-line diff): a meaningful review — even a clean approval — emits
# a real summary string, which alone runs longer than 32 tokens once
# you account for word boundaries; near-empty envelopes
# (``{"summary":"","findings":[]}``) only require ~12 tokens, so 32
# strikes a balance between catching truncation noise and not firing
# on legitimate-but-terse reviews. The schema only *requires*
# ``summary`` (a single short string would technically satisfy it),
# so the threshold is empirical — observed truncated outputs cluster
# under 16 tokens; observed legitimate reviews cluster well above.
_TRUNCATION_INPUT_THRESHOLD = 2000
_TRUNCATION_OUTPUT_THRESHOLD = 32


# Standard ``num_ctx`` step sizes Ollama exposes. We round up the
# estimated token count to one of these so the model gets a
# power-of-two-ish window — which is what the underlying llama.cpp
# implementations expect — rather than an arbitrary number.
_NUM_CTX_LADDER: tuple[int, ...] = (8192, 16384, 32768, 65536, 131072)

# Ollama's compiled-in default. We only override above this; below it
# the override would be a no-op and risks rejection from local models
# that don't accept smaller-than-default ``num_ctx``.
_NUM_CTX_DEFAULT = 4096

# Conservative UTF-8 BYTES-per-token ratio for any script. Counting
# bytes (not characters) keeps the estimator uniform across scripts:
# ASCII is 1 byte/char and ~3-4 chars/token (so ~3-4 bytes/token);
# CJK is 3 bytes/char and ~1 char/token (so ~3 bytes/token). Across
# every script the ratio sits at roughly 3-5 bytes/token. Picking 3
# as the lower bound biases toward overshooting — overestimating
# tokens just sizes ``num_ctx`` slightly large; underestimating is
# the trap this whole helper exists to close.
#
# Earlier versions used ``len(prompt)`` (character count) which
# silently underestimated for CJK / RTL / emoji-heavy content where a
# single character can be one token. ``len(prompt.encode('utf-8'))``
# is the canonical fix.
_BYTES_PER_TOKEN = 3

# Headroom reserved for the model's response so ``num_ctx`` covers
# both prompt and completion. Real reviews emit summary + N findings;
# 1024 tokens covers a typical structured response with several
# medium-bodied findings.
_RESPONSE_TOKEN_HEADROOM = 1024


def _resolve_num_ctx(
    request: ReviewRequest,
    config: OllamaProviderConfig,
    prompt: str,
) -> int | None:
    """Apply the documented num_ctx resolution order.

    Returns ``None`` only when all three layers agree the model's
    documented default is appropriate. The auto-bump heuristic is
    the floor — if neither the request nor config supplies a value,
    the prompt-length estimate kicks in and bumps to the smallest
    ladder step that fits.

    Resolution:

    1. ``request.metadata["num_ctx"]`` — caller per-call override.
    2. ``config.num_ctx`` — operator-pinned default.
    3. :func:`_suggest_num_ctx` — auto-bump from prompt length.
    4. ``None`` — let Ollama apply the model default.

    Non-int / non-positive values at layers 1 or 2 are treated as
    "unset" and fall through. The bus boundary can deliver any JSON
    shape under metadata, so this is a defensive check rather than
    an error path: a misconfigured rule never silently skips the
    auto-bump fallback.
    """
    # Layer 1: caller per-call override.
    caller_value = request.metadata.get("num_ctx")
    caller_int = _coerce_positive_int(caller_value)
    if caller_int is not None:
        return caller_int

    # Layer 2: operator-pinned config default.
    config_int = _coerce_positive_int(config.num_ctx)
    if config_int is not None:
        return config_int

    # Layer 3: auto-bump fallback (current pre-FR behavior).
    return _suggest_num_ctx(prompt)


def _coerce_positive_int(value: Any) -> int | None:
    """Return ``value`` as a positive int, or ``None`` for non-int /
    non-positive / unparseable inputs.

    The bus boundary delivers ``num_ctx`` as JSON; YAML configs
    deliver it as an int already. Both paths run through here so a
    misconfigured payload (None, "16384", float, list) falls through
    to the auto-bump heuristic rather than crashing the provider.
    """
    if isinstance(value, bool):
        # ``bool`` is an ``int`` subclass in Python; reject explicitly
        # so ``num_ctx=True`` doesn't silently land as 1.
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _suggest_num_ctx(prompt: str) -> int | None:
    """Suggest a ``num_ctx`` value that fits the prompt + a response.

    Returns the smallest standard window in :data:`_NUM_CTX_LADDER`
    that holds an estimated prompt-token count plus
    :data:`_RESPONSE_TOKEN_HEADROOM`. Returns ``None`` when the
    estimate fits inside Ollama's default 4096-token window — keeping
    the fast path one-keyword-shorter and avoiding overrides that
    might confuse smaller local models.

    The estimator is deliberately conservative — UTF-8 byte length
    divided by :data:`_BYTES_PER_TOKEN` (3, the lower bound of the
    real ~3-5 bytes/token range across scripts), with ``math.ceil``
    rather than floor division — to bias toward overshooting: a
    slightly larger ``num_ctx`` costs a small amount of GPU memory,
    while the bug this closes (``num_ctx`` too small) silently hides
    review findings. Byte-counting (rather than character counting)
    keeps the estimator uniform across CJK / RTL / emoji-heavy
    content where a single character can be one token. Ceiling
    division prevents borderline prompts from rounding *down* under
    the threshold and skipping the override.
    """
    estimated_tokens = (
        math.ceil(len(prompt.encode("utf-8")) / _BYTES_PER_TOKEN)
        + _RESPONSE_TOKEN_HEADROOM
    )
    if estimated_tokens <= _NUM_CTX_DEFAULT:
        return None
    for ctx in _NUM_CTX_LADDER:
        if estimated_tokens <= ctx:
            return ctx
    # Beyond the largest ladder step the model probably can't usefully
    # consume the prompt anyway; cap at the largest entry so the
    # request still goes through with a clear truncation-warning if
    # the model output is small.
    return _NUM_CTX_LADDER[-1]


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
    "OllamaAuthError",
    "OllamaHealthcheckError",
    "OllamaProvider",
    "OllamaProviderConfig",
]
