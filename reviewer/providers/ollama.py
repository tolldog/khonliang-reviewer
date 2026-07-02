"""Ollama review provider via the **native** ``/api/chat`` endpoint.

Hits Ollama's native ``/api/chat`` endpoint (``httpx``) so any
Ollama-served model — local (``qwen2.5-coder:14b``, ``glm-4.7-flash``)
or cloud (``kimi-k2.5:cloud``, ``glm-5:cloud``) — can drive a review
from the same provider. Model choice is per-request via
``request.metadata["model"]`` with a config-level fallback.

**Why native, not the OpenAI-compat ``/v1`` shim:** the ``/v1`` shim
silently DROPS ``options.num_ctx`` (verified 2026-05-30 — a 66k-token
prompt reports ``prompt_tokens=4096`` with or without the override),
so every review whose prompt exceeded the 4096-token default was
truncated and returned a near-empty response indistinguishable from a
clean approval (bug_reviewer_832a909b / bug_reviewer_a21581dc). The
native endpoint honors ``options.num_ctx``, so the auto-bump heuristic
(:func:`_suggest_num_ctx`) actually takes effect. Native responses are
shaped back into the OpenAI chat-completion form by
:func:`_native_to_compat` so the existing parsers stay unchanged.

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

    #: Kept pointing at the OpenAI-compat ``/v1`` base for config
    #: back-compat; :func:`_native_base_url` strips the ``/v1`` suffix to
    #: reach the native ``/api/*`` surface (the only one that honors
    #: ``options.num_ctx`` — bug_reviewer_832a909b). A base_url without
    #: ``/v1`` is used as-is.
    base_url: str = "http://localhost:11434/v1"
    #: Sent as ``Authorization: Bearer <api_key>`` on every native
    #: request. Local Ollama ignores it; an auth-required deployment
    #: (cloud Ollama, or a reverse-proxied endpoint that enforces auth)
    #: needs it, and it's what makes the 401/403 handling reachable-by-
    #: config. The ``"ollama"`` default is a harmless placeholder for
    #: local use — set a real token for an auth-gated endpoint.
    api_key: str = "ollama"
    default_model: str = "qwen2.5-coder:14b"
    timeout_seconds: float = 300.0
    #: Retry a single time when the FIRST ``/api/chat`` call times out.
    #: The pre-push hot-tier reviewer intermittently times out on the
    #: *first* call after the model has been evicted from VRAM: Ollama
    #: cold-loads the model (seconds to tens of seconds) inside the same
    #: request, blowing ``timeout_seconds`` — but the model is then
    #: resident, so a warm second attempt succeeds first-try (the
    #: "recovered on retry" tell in dog_833295f0). Retrying once turns
    #: that cold-start blip into a completed review instead of a
    #: degraded-to-skipped one (fr_khonliang-reviewer_26734e09). Only the
    #: TIMEOUT path retries — HTTP/auth/transport/parse errors return
    #: immediately (a retry wouldn't help and would just double the
    #: latency of a hard failure). Capped at one retry (two attempts
    #: total). Set ``False`` to disable and fail fast on the first
    #: timeout, matching the pre-FR behavior.
    retry_on_timeout: bool = True
    #: Operator-pinned ``num_ctx`` for every Ollama review unless the
    #: caller overrides via ``request.metadata["num_ctx"]``. ``None``
    #: (default) falls through to the auto-bump heuristic
    #: (:func:`_suggest_num_ctx`) so existing configs keep working
    #: unchanged. Useful for measurement runs that want a fixed
    #: context window so duration / token-count comparisons hold the
    #: ``num_ctx`` axis constant; the auto-bump heuristic varies with
    #: prompt size.
    num_ctx: int | None = None
    #: Native Ollama ``format`` parameter (e.g. ``"json"``). Sent on the
    #: native ``/api/chat`` call to grammar-constrain decoding. Use to
    #: force structured output from small models that otherwise produce
    #: free-form text — see the 2026-04-22 evaluator-gate experiment
    #: where ``llama3.2:3b`` failed to produce parseable JSON in every
    #: prompt configuration tested. Caller can override per-call via
    #: ``request.metadata["format"]``. ``None`` (default) → the provider
    #: applies ``"json"`` (the review schema is JSON and the parser
    #: errors on non-JSON), so unlike the old ``/v1`` path — where
    #: ``response_format`` was silently ignored for local models — JSON
    #: is now actually enforced.
    format: str | None = None


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
        http_client: Any | None = None,
    ):
        self.config = config or OllamaProviderConfig()
        #: Native Ollama API root. The ``/v1`` OpenAI-compat shim drops
        #: ``options.num_ctx``; the native ``/api/*`` surface honors it
        #: (bug_reviewer_832a909b), so all calls go through the native
        #: root derived from ``base_url``.
        self._native_url = _native_base_url(self.config.base_url)
        #: Injectable for tests (a fake exposing async ``post``/``get``
        #: returning objects with ``status_code`` / ``json()`` /
        #: ``raise_for_status()``); production builds a real httpx client.
        #: ``api_key`` is threaded as a Bearer token so auth-required
        #: deployments work (local Ollama ignores it).
        self._http = http_client or httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            headers=_auth_headers(self.config.api_key),
        )

    async def healthcheck(self) -> None:
        """Verify the native endpoint is reachable and serving models.

        Hits the native ``/api/tags`` route — Ollama's lightweight "what
        models are available here" query. Failures split into categories
        so agent boot can act on each:

        - :class:`OllamaAuthError` — credentials rejected (HTTP 401/403;
          cloud-hosted Ollama typically). Caller should fix the API key.
        - :class:`OllamaHealthcheckError` ``"not reachable"`` — transport
          couldn't reach the server. Caller should check it's running
          and the URL is right.
        - :class:`OllamaHealthcheckError` ``"healthcheck failed"`` —
          any other HTTP error (bad request, rate limit, etc.). Caller
          reads the message and investigates.
        """
        try:
            response = await self._http.get(f"{self._native_url}/api/tags")
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                raise OllamaAuthError(
                    f"ollama endpoint at {self._native_url} rejected "
                    f"credentials (HTTP {status}): {exc}"
                ) from exc
            raise OllamaHealthcheckError(
                f"ollama healthcheck failed at {self._native_url} "
                f"(HTTP {status}): {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaHealthcheckError(
                f"ollama endpoint not reachable at {self._native_url}: {exc}"
            ) from exc

    async def review(self, request: ReviewRequest) -> ReviewResult:
        # ``_khonliang_repo_prompts`` / ``_khonliang_example_format`` are
        # in-process-only passthrough metadata (see reviewer.agent).
        # Absent keys or wrong types collapse to the pre-FR behavior —
        # a plain built-in prompt with no repo-side merge.
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
        # Default to JSON-constrained decoding: the review schema is JSON
        # and _parse_response errors on non-JSON. ``format`` on the NATIVE
        # endpoint actually constrains the grammar (the OpenAI-compat
        # ``response_format`` was silently ignored for local models).
        # Caller / config can override via the format axis.
        fmt = _resolve_format(request, self.config) or "json"

        # NATIVE ``/api/chat`` payload. Critically, ``options.num_ctx`` is
        # honored here (the ``/v1`` shim drops it — bug_reviewer_832a909b),
        # so a large prompt gets a context window that fits it instead of
        # being silently truncated to 4096.
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": fmt,
        }
        if num_ctx_override is not None:
            payload["options"] = {"num_ctx": num_ctx_override}

        # Cold-start timeout resilience (fr_khonliang-reviewer_26734e09):
        # the first ``/api/chat`` call after a model eviction cold-loads
        # the model inside the request and can blow ``timeout_seconds``; a
        # warm second attempt then succeeds first-try (dog_833295f0). Retry
        # ONLY on the timeout path, capped at one retry. Every non-timeout
        # error returns immediately from inside the loop — a retry wouldn't
        # help and would just double the latency of a hard failure.
        # ``started_wall`` / ``started_mono`` stay fixed before the loop so
        # a persistent-timeout ``duration_ms`` spans both attempts.
        max_attempts = 2 if self.config.retry_on_timeout else 1
        native: Any = None
        for attempt in range(max_attempts):
            try:
                http_response = await self._http.post(
                    f"{self._native_url}/api/chat",
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                http_response.raise_for_status()
                native = http_response.json()
                break
            except httpx.TimeoutException:
                if attempt + 1 < max_attempts:
                    # Warm-up retry. This log line is the visible signal in
                    # real pre-push cycles that a cold-start timeout was
                    # recovered rather than degraded to a skip.
                    logger.warning(
                        "ollama request timed out after %ss (attempt %s/%s); "
                        "retrying once — likely cold-start model load, warm "
                        "retry usually succeeds (model=%s)",
                        self.config.timeout_seconds,
                        attempt + 1,
                        max_attempts,
                        model,
                    )
                    continue
                return _errored(
                    request,
                    error=f"ollama request timed out after {self.config.timeout_seconds}s",
                    # Distinct from the generic ``backend_error`` (HTTP / connection
                    # failures below) so the sign-off path can degrade a *timeout*
                    # to "review-skipped" — a non-blocking pre-push outcome — while
                    # other backend failures stay hard errors. Parallels the
                    # claude_cli provider's ``subprocess_timeout`` category.
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
                        error=f"ollama rejected credentials (HTTP {status}): {exc}",
                        error_category="auth_not_provisioned",
                        model=model,
                        started_wall=started_wall,
                        duration_ms=_elapsed_ms(started_mono),
                    )
                return _errored(
                    request,
                    error=f"ollama returned HTTP {status}: {exc}",
                    error_category="backend_error",
                    model=model,
                    started_wall=started_wall,
                    duration_ms=_elapsed_ms(started_mono),
                )
            except httpx.HTTPError as exc:
                # ConnectError / transport / protocol failure — the native
                # endpoint couldn't be reached or spoke badly.
                return _errored(
                    request,
                    error=f"ollama endpoint unreachable: {exc}",
                    error_category="backend_error",
                    model=model,
                    started_wall=started_wall,
                    duration_ms=_elapsed_ms(started_mono),
                )
            except (json.JSONDecodeError, ValueError) as exc:
                return _errored(
                    request,
                    error=f"ollama response body was not valid JSON: {exc}",
                    error_category="malformed_envelope",
                    model=model,
                    started_wall=started_wall,
                    duration_ms=_elapsed_ms(started_mono),
                )

        duration_ms = _elapsed_ms(started_mono)
        result = _parse_response(
            _native_to_compat(native),
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
    if isinstance(override, str) and override.strip():
        # Strip stray whitespace; a whitespace-only override is "unset",
        # not a (broken) model name that would 404 at the backend.
        return override.strip()
    return default


def _auth_headers(api_key: str | None) -> dict[str, str]:
    """Build the ``Authorization`` header for native requests.

    Mirrors what the old openai SDK did with ``api_key`` (it sent it as a
    Bearer token). Empty / whitespace keys produce no header so a
    deliberately-unset key doesn't send ``Bearer`` with nothing. The
    default ``"ollama"`` placeholder is harmless for local endpoints and
    keeps auth-required deployments fixable purely by config.
    """
    if isinstance(api_key, str) and api_key.strip():
        # Use the stripped value — a config/bus key with stray
        # leading/trailing whitespace would otherwise produce a bad
        # ``Bearer <key> `` token and fail auth on an otherwise-valid key.
        return {"Authorization": f"Bearer {api_key.strip()}"}
    return {}


def _native_base_url(base_url: str) -> str:
    """Map the OpenAI-compat base (``.../v1``) to the native Ollama root.

    Ollama serves its native API at the host root (``/api/chat``,
    ``/api/tags``) while the OpenAI-compat shim lives under ``/v1``. The
    shim silently ignores ``options.num_ctx``; the native API honors it
    (bug_reviewer_832a909b). Configs still point ``base_url`` at the
    ``/v1`` shim for back-compat, so strip a trailing ``/v1`` (and any
    trailing slash) to reach the native surface. A ``base_url`` that
    already lacks ``/v1`` is returned unchanged (minus trailing slash).
    """
    # ``.strip()`` first: an env/file-sourced base_url often carries a
    # trailing newline ("http://host:11434/v1\n"), which would otherwise
    # defeat the ``/v1`` suffix check and yield an unrequestable URL.
    trimmed = base_url.strip().rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return trimmed or base_url.strip()


def _native_to_compat(native: Any) -> dict[str, Any]:
    """Shape a native ``/api/chat`` response like an OpenAI chat-completion.

    Lets the existing OpenAI-shaped parsers (:func:`_parse_response`,
    :func:`_extract_message_content`, :func:`_build_usage`) consume the
    native response unchanged. Field mapping:

    - ``message.content``    -> ``choices[0].message.content``
    - ``done_reason``        -> ``choices[0].finish_reason``
    - ``prompt_eval_count``  -> ``usage.prompt_tokens``
    - ``eval_count``         -> ``usage.completion_tokens``

    Tolerant of a non-dict body (returns an empty-content shape so the
    parser raises the usual ``malformed_envelope`` rather than crashing).
    """
    body = native if isinstance(native, dict) else {}
    message = body.get("message")
    if not isinstance(message, dict):
        message = {}
    return {
        "choices": [
            {
                "message": {"content": message.get("content")},
                "finish_reason": body.get("done_reason"),
            }
        ],
        "usage": {
            "prompt_tokens": body.get("prompt_eval_count"),
            "completion_tokens": body.get("eval_count"),
        },
    }


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


# Round the sizing budget *up* to a multiple of this so the window
# tracks the prompt at a fine granularity instead of leaping to the next
# power-of-two. llama.cpp accepts an arbitrary ``n_ctx`` — there is no
# hard power-of-two requirement — so a granular ceiling is safe. This is
# sizing *hygiene* (explicit, fine-grained, the window means what the
# budget says) rather than a VRAM win: once the budget below accounts for
# worst-case token density + chat-template overhead, the resulting window
# lands *comparable* to the old coarse ladder — sometimes tighter (mid
# range), sometimes slightly larger (the ladder's accidental slack
# happened to under-cover dense diffs there). The real decode-speed /
# VRAM lever for the timeout this milestone targets is the fast pre-push
# tier (part 2) and timeout→skip degradation (part 3), not this.
_NUM_CTX_GRANULARITY = 2048

# Hard ceiling on the auto-bumped window. Beyond this the model can't
# usefully consume the prompt anyway; cap here so an oversized prompt
# still goes through (with a truncation warning if output is small)
# rather than requesting an absurd context.
_NUM_CTX_MAX = 131072

# Ollama's compiled-in default. We only override above this; below it
# the override would be a no-op and risks rejection from local models
# that don't accept smaller-than-default ``num_ctx``.
_NUM_CTX_DEFAULT = 4096

# Headroom reserved for the model's response so ``num_ctx`` covers
# both prompt and completion. Real reviews emit summary + N findings;
# 1024 tokens covers a typical structured response with several
# medium-bodied findings.
_RESPONSE_TOKEN_HEADROOM = 1024

# Fixed allowance for the chat-template / control tokens Ollama's
# ``/api/chat`` injects around the prompt (system + role markers, BOS/EOS,
# tool framing) *before* enforcing ``options.num_ctx``. These are not in
# ``prompt`` itself, so without an explicit allowance a window sized to
# exactly ``worst_case + headroom`` (e.g. landing on a granularity
# multiple) would leave zero room for them and truncate. 256 is generous
# for any single system+user+assistant wrapper.
_CHAT_TEMPLATE_TOKEN_OVERHEAD = 256

# Worst-case UTF-8 bytes-per-token floor — the SINGLE density assumption
# behind both "do we need to override?" and "how big a window?". Review
# prompts are code diffs dense with punctuation, short identifiers, and
# newlines, whose worst hunks tokenize down to ~2.3 bytes/token. Using one
# floor for both the decision and the sizing keeps them consistent: there
# is no band where we decline to override yet would have sized a window
# larger than the default (the silent-truncation gap this helper exists to
# close). Counting bytes (not characters) keeps the estimate uniform across
# scripts — ASCII ~3-4 bytes/token, CJK ~3 bytes/token, all ≥ this floor;
# ``len(prompt.encode('utf-8'))`` is the canonical measure (an earlier
# character-count version under-counted CJK / RTL / emoji-heavy content).
# The old coarse ladder (8k/16k/32k/…) absorbed density variance by
# accident — its next-power-of-two jump left a large cushion. Granular
# rounding removes that accidental cushion, so we restore a *deliberate,
# provable* one: round the budget up to the granularity. Because rounding
# only ever increases the value, the window is guaranteed ≥ the token count
# a real 2.3-bytes/token tokenization plus its chat wrapper needs — no
# boundary off-by-one. Net window size lands comparable to the old ladder
# (the point is correctness + explicit budgeting, not a VRAM reduction).
_BYTES_PER_TOKEN_FLOOR = 2.3


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


def _resolve_format(
    request: ReviewRequest,
    config: OllamaProviderConfig,
) -> str | None:
    """Apply the documented ``format`` resolution order.

    Resolution:

    1. ``request.metadata["format"]`` — caller per-call override.
    2. ``config.format`` — operator-pinned default.
    3. ``None`` — unset; the caller (:meth:`OllamaProvider.review`)
       then applies ``"json"`` as the effective default, since the
       review schema is JSON.

    Non-string / empty values at layers 1 or 2 are treated as "unset"
    and fall through. Mirrors :func:`_resolve_num_ctx` so a malformed
    bus payload (``format=42``, ``format=""``, ``format=True``) lands
    silently as the default rather than crashing the provider.

    No enum restriction here — Ollama accepts ``"json"`` today and may
    add schema-constrained variants tomorrow. Whatever the caller sends
    is forwarded verbatim to the native ``format`` field; an unsupported
    value yields an HTTP error that reaches the caller via the
    :class:`httpx.HTTPStatusError` handler in :meth:`review`.
    """
    caller_value = request.metadata.get("format")
    caller_str = _coerce_format_value(caller_value)
    if caller_str is not None:
        return caller_str
    config_str = _coerce_format_value(config.format)
    if config_str is not None:
        return config_str
    return None


def _coerce_format_value(value: Any) -> str | None:
    """Return ``value`` as a non-empty string, or ``None`` otherwise.

    The bus boundary delivers ``format`` as JSON; YAML configs deliver
    it as a string already. Both paths run through here so a
    misconfigured payload (None, int, bool, list, empty string) falls
    through to the next resolution layer rather than crashing.
    """
    if isinstance(value, str) and value.strip():
        # Strip so a whitespace-only value is "unset" (matching the
        # docstring) instead of being forwarded to Ollama as a bogus
        # ``format`` that triggers an avoidable backend error.
        return value.strip()
    return None


def _suggest_num_ctx(prompt: str) -> int | None:
    """Suggest a ``num_ctx`` value that fits the prompt + a response.

    Computes a single budget — worst-case token-dense tokenization
    (``bytes`` / :data:`_BYTES_PER_TOKEN_FLOOR`) plus
    :data:`_RESPONSE_TOKEN_HEADROOM` plus
    :data:`_CHAT_TEMPLATE_TOKEN_OVERHEAD` — and uses it for *both* the
    override decision and the window size. Returns ``None`` when that
    budget already fits inside Ollama's default 4096-token window (keeping
    the fast path one-keyword-shorter and avoiding overrides that might
    confuse smaller local models); otherwise rounds the budget *up* to the
    next multiple of :data:`_NUM_CTX_GRANULARITY`, capped at
    :data:`_NUM_CTX_MAX`, and returns that.

    Using one budget for both purposes is deliberate: an earlier version
    decided on a *typical* bytes/3 estimate but sized on the worst-case
    floor, which left a band of mid-size prompts that we declined to
    override yet would have sized above the default — i.e. they were sent
    at 4096 and could silently truncate, the exact bug this helper exists
    to close. One budget removes that gap.

    Byte-counting (rather than character counting) keeps the budget uniform
    across CJK / RTL / emoji-heavy content where a single character can be
    one token; ``math.ceil`` rather than floor division biases toward
    overshooting, so a borderline prompt rounds *up* into an override
    rather than down past the threshold.
    """
    prompt_bytes = len(prompt.encode("utf-8"))
    # One budget, used for both the decision and the size. Every consumer
    # of the window is accounted for — worst-case dense prompt, the
    # response, and the chat-template / control tokens /api/chat injects —
    # so a window sized to this budget can never under-provision relative
    # to the decision that produced it.
    budget = (
        math.ceil(prompt_bytes / _BYTES_PER_TOKEN_FLOOR)
        + _RESPONSE_TOKEN_HEADROOM
        + _CHAT_TEMPLATE_TOKEN_OVERHEAD
    )
    if budget <= _NUM_CTX_DEFAULT:
        return None
    # Snug-round up to the granularity; rounding only ever increases the
    # value, so the window is provably ≥ the budget — no boundary off-by-one.
    snug = math.ceil(budget / _NUM_CTX_GRANULARITY) * _NUM_CTX_GRANULARITY
    return min(snug, _NUM_CTX_MAX)


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
            section=item.get("section") if isinstance(item.get("section"), str) else None,
        )
        for item in raw_findings
        if isinstance(item, dict)
    ]

    # Only opted-in pr_diff reviews surface verdicts (codex PR B R6): the
    # lib contract documents ReviewResult.verdicts as populated only when
    # scoring_mode='binary_questions', so an unsolicited model-emitted
    # 'verdicts' key on a holistic review is dropped, not forwarded.
    verdicts = (
        parse_verdicts(payload) if binary_questions_active(request) else []
    )
    # Ollama has no transport-level schema enforcement — the schema is prompt
    # text — so in binary-questions mode the model can omit or underfill
    # ``verdicts`` and the review would silently degrade to a holistic shape
    # (codex PR B R5). Enforce the fixed-dimensions contract here: incomplete
    # coverage is the same class as unparseable JSON — the model failed the
    # response contract.
    if binary_questions_active(request):
        coverage_error = validate_verdict_coverage(verdicts)
        if coverage_error is not None:
            return _errored(
                request,
                error=f"ollama response failed binary-questions contract: {coverage_error}",
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
    """Best-effort conversion of a usage object to a plain dict.

    The native path hands :func:`_native_to_compat` a plain dict, so
    this is normally a passthrough. Retained defensively for any
    attribute-style object (e.g. an injected fake or a pydantic model):
    ``model_dump`` then attribute-sniffing keeps every shape on one
    code path.
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
