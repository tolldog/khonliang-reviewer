"""Claude-via-CLI review provider.

Shells out to ``claude -p --output-format=json --json-schema ...`` so the
agent drives Claude's subscription quota via ``CLAUDE_CODE_OAUTH_TOKEN``
without going through the API pay-per-token path.

Subprocess is the only sanctioned integration for subscription-backed
Claude: the Anthropic SDK does not accept the subscription OAuth token,
and using it through third-party SDKs violates Anthropic's 2026 Consumer
TOS. ``claude -p`` is the officially supported headless path.

The CLI returns a structured JSON envelope that carries both the model
response and the per-call token accounting — including ``total_cost_usd``
which we record as ``UsageEvent.estimated_api_cost_usd``. For
subscription-backed runs that field is the hypothetical pay-per-token
cost, which is exactly the comparison we want for subscription-vs-API
evaluation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from khonliang_reviewer import (
    ReviewFinding,
    ReviewProvider,
    ReviewRequest,
    ReviewResult,
    UsageEvent,
)

from reviewer.providers._prompt import REVIEW_RESPONSE_SCHEMA, build_review_prompt


logger = logging.getLogger(__name__)


class ClaudeCliAuthError(RuntimeError):
    """Raised when ``claude auth status`` reports a logged-out state.

    Distinct from generic :class:`RuntimeError` so agent boot can catch the
    auth-specific case and surface it to operators without mixing it into
    other startup failures. All ``review()`` error paths that can be
    attributed to auth instead populate
    ``ReviewResult.error_category="auth_not_provisioned"``.
    """


@dataclass
class ClaudeCliProviderConfig:
    """Construction-time configuration for :class:`ClaudeCliProvider`."""

    binary: str = "claude"
    timeout_seconds: float = 300.0
    append_system_prompt: str | None = None


class ClaudeCliProvider(ReviewProvider):
    """Review provider backed by Claude Code's headless ``-p`` mode.

    Subscription-backed: runs draw from the ``CLAUDE_CODE_OAUTH_TOKEN``
    quota (provision once per machine via ``claude setup-token``), not
    from ``ANTHROPIC_API_KEY``. Honor :class:`ClaudeCliProviderConfig`'s
    ``timeout_seconds`` to bound runaway invocations.
    """

    name = "claude_cli"

    def __init__(self, config: ClaudeCliProviderConfig | None = None):
        self.config = config or ClaudeCliProviderConfig()

    async def healthcheck(self) -> None:
        """Verify the CLI is authenticated. Intended for agent boot.

        Call once at startup; on success, subsequent ``review()`` calls
        can assume auth is in place (OAuth tokens from
        ``claude setup-token`` are long-lived). If auth is revoked
        mid-session the regular ``review()`` error path still catches it
        and surfaces the failure with ``error_category="backend_error"``
        or ``"nonzero_exit"`` depending on how the CLI manifests it.

        Raises:
            FileNotFoundError: if the ``claude`` binary is not on PATH.
            ClaudeCliAuthError: if the CLI reports logged-out.
            RuntimeError: for other unexpected failures (non-zero exit,
                malformed output).
        """
        proc = await asyncio.create_subprocess_exec(
            self.config.binary,
            "auth",
            "status",
            "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError("claude auth status timed out after 15s")

        if proc.returncode != 0:
            stderr_text = stderr.decode(errors="replace").strip()[:500]
            raise RuntimeError(
                f"claude auth status exited with {proc.returncode}"
                + (f": {stderr_text}" if stderr_text else "")
            )

        try:
            info = json.loads(stdout.decode(errors="replace"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"claude auth status returned non-JSON: {exc}")

        if not info.get("loggedIn"):
            raise ClaudeCliAuthError(
                "claude CLI is not authenticated; run `claude setup-token` "
                "or `claude login` (CLAUDE_CODE_OAUTH_TOKEN) before using "
                "the provider"
            )

    async def review(self, request: ReviewRequest) -> ReviewResult:
        # ``_khonliang_repo_prompts`` / ``_khonliang_example_format`` are
        # in-process-only passthrough metadata (see reviewer.agent).
        # Absent keys or wrong types collapse to the pre-FR behavior —
        # a plain built-in prompt with no repo-side merge.
        repo_prompts = request.metadata.get("_khonliang_repo_prompts")
        example_format = request.metadata.get("_khonliang_example_format")
        prompt = build_review_prompt(
            request,
            include_schema=False,
            repo_prompts=repo_prompts,
            example_format=example_format if isinstance(example_format, str) else None,
        )
        started_wall = time.time()
        started_mono = time.monotonic()

        cmd = [
            self.config.binary,
            "-p",
            "--output-format=json",
            "--json-schema",
            json.dumps(REVIEW_RESPONSE_SCHEMA),
        ]
        if self.config.append_system_prompt:
            cmd += ["--append-system-prompt", self.config.append_system_prompt]
        # Caller-specified model flows via request.metadata["model"].
        # `claude -p --model <spec>` accepts either an alias (`opus`, `sonnet`)
        # or a fully-qualified id (`claude-opus-4-7`).
        requested_model = request.metadata.get("model")
        if isinstance(requested_model, str) and requested_model:
            cmd += ["--model", requested_model]
        # Prompt goes via stdin, not argv: diffs easily exceed OS ARG_MAX
        # (~128KB on Linux) and argv is visible to other users via `ps`.

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return _errored(
                request,
                error=f"claude binary not found at {self.config.binary!r}",
                error_category="binary_not_found",
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode()),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return _errored(
                request,
                error=f"claude -p timed out after {self.config.timeout_seconds}s",
                error_category="subprocess_timeout",
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )

        duration_ms = _elapsed_ms(started_mono)

        if proc.returncode != 0:
            stderr_text = stderr.decode(errors="replace").strip()[:500]
            category = (
                "auth_not_provisioned"
                if _stderr_suggests_auth_failure(stderr_text)
                else "nonzero_exit"
            )
            return _errored(
                request,
                error=(
                    f"claude -p exited with {proc.returncode}"
                    + (f": {stderr_text}" if stderr_text else "")
                ),
                error_category=category,
                started_wall=started_wall,
                duration_ms=duration_ms,
            )

        try:
            envelope = json.loads(stdout.decode(errors="replace"))
        except json.JSONDecodeError as exc:
            return _errored(
                request,
                error=f"claude -p returned non-JSON output: {exc}",
                error_category="malformed_envelope",
                started_wall=started_wall,
                duration_ms=duration_ms,
            )

        if not isinstance(envelope, dict):
            return _errored(
                request,
                error=(
                    "claude -p returned JSON that is not an object "
                    f"(type={type(envelope).__name__})"
                ),
                error_category="malformed_envelope",
                started_wall=started_wall,
                duration_ms=duration_ms,
            )

        result = _parse_envelope(
            envelope,
            request=request,
            started_wall=started_wall,
            fallback_duration_ms=duration_ms,
        )
        logger.debug(
            "claude_cli review done: disposition=%s category=%s model=%s tokens_in=%s tokens_out=%s duration_ms=%s",
            result.disposition,
            result.error_category or "-",
            result.model,
            result.usage.input_tokens if result.usage else 0,
            result.usage.output_tokens if result.usage else 0,
            result.usage.duration_ms if result.usage else 0,
        )
        return result


def _parse_envelope(
    envelope: dict[str, Any],
    *,
    request: ReviewRequest,
    started_wall: float,
    fallback_duration_ms: int,
) -> ReviewResult:
    """Translate a ``claude -p`` JSON envelope into a :class:`ReviewResult`."""
    if envelope.get("is_error") or envelope.get("type") != "result":
        return _errored(
            request,
            error=(
                envelope.get("api_error_status")
                or envelope.get("result")
                or "claude -p reported an error"
            ),
            error_category="backend_error",
            started_wall=started_wall,
            duration_ms=int(envelope.get("duration_ms") or fallback_duration_ms),
            envelope=envelope,
        )

    duration_ms = int(envelope.get("duration_ms") or fallback_duration_ms)
    model = _pick_primary_model(envelope) or "claude"

    # When ``claude -p`` is invoked with ``--json-schema``, the CLI
    # validates the model's response against the schema and returns the
    # parsed object in ``structured_output`` — the ``result`` field is
    # left empty in that mode. Prefer ``structured_output`` when it's
    # present and dict-shaped; fall back to parsing ``result`` as JSON
    # for plain (non-schema) invocations.
    structured = envelope.get("structured_output")
    if isinstance(structured, dict):
        payload: dict[str, Any] = structured
    else:
        result_text = envelope.get("result") or ""
        try:
            parsed = json.loads(result_text)
        except json.JSONDecodeError as exc:
            return _errored(
                request,
                error=f"claude -p result was not JSON: {exc}",
                error_category="malformed_envelope",
                started_wall=started_wall,
                duration_ms=duration_ms,
                envelope=envelope,
            )
        if not isinstance(parsed, dict):
            return _errored(
                request,
                error=(
                    "claude -p result JSON is not an object "
                    f"(type={type(parsed).__name__})"
                ),
                error_category="malformed_envelope",
                started_wall=started_wall,
                duration_ms=duration_ms,
                envelope=envelope,
            )
        payload = parsed

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
        envelope,
        request=request,
        started_wall=started_wall,
        duration_ms=duration_ms,
        model=model,
        disposition="posted",
    )

    return ReviewResult(
        request_id=request.request_id,
        summary=summary,
        findings=findings,
        disposition="posted",
        usage=usage,
        backend=ClaudeCliProvider.name,
        model=model,
        created_at=started_wall,
    )


def _pick_primary_model(envelope: dict[str, Any]) -> str | None:
    """Return the highest-``costUSD`` model from the envelope's ``modelUsage``."""
    model_usage = envelope.get("modelUsage") or {}
    if not isinstance(model_usage, dict) or not model_usage:
        return None
    best_name: str | None = None
    best_cost = -1.0
    for name, stats in model_usage.items():
        stats_dict = stats if isinstance(stats, dict) else {}
        try:
            cost = float(stats_dict.get("costUSD") or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        if cost > best_cost:
            best_cost = cost
            best_name = name
    return best_name


def _build_usage(
    envelope: dict[str, Any],
    *,
    request: ReviewRequest,
    started_wall: float,
    duration_ms: int,
    model: str,
    disposition: str,
    error: str = "",
    error_category: str = "",
) -> UsageEvent:
    usage_raw = envelope.get("usage")
    if not isinstance(usage_raw, dict):
        usage_raw = {}
    try:
        cost = float(envelope.get("total_cost_usd") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    return UsageEvent(
        timestamp=started_wall,
        backend=ClaudeCliProvider.name,
        model=model,
        input_tokens=_safe_int(usage_raw.get("input_tokens")),
        output_tokens=_safe_int(usage_raw.get("output_tokens")),
        cache_read_tokens=_safe_int(usage_raw.get("cache_read_input_tokens")),
        cache_creation_tokens=_safe_int(usage_raw.get("cache_creation_input_tokens")),
        duration_ms=duration_ms,
        disposition=disposition,  # type: ignore[arg-type]
        request_id=request.request_id,
        repo=str(request.metadata.get("repo", "")),
        pr_number=_int_or_none(request.metadata.get("pr_number")),
        estimated_api_cost_usd=cost,
        error=error,
        error_category=error_category,
    )


def _errored(
    request: ReviewRequest,
    *,
    error: str,
    error_category: str,
    started_wall: float,
    duration_ms: int,
    envelope: dict[str, Any] | None = None,
) -> ReviewResult:
    envelope = envelope or {}
    model = _pick_primary_model(envelope) or "claude"
    usage = _build_usage(
        envelope,
        request=request,
        started_wall=started_wall,
        duration_ms=duration_ms,
        model=model,
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
        backend=ClaudeCliProvider.name,
        model=model,
        created_at=started_wall,
    )


_AUTH_FAILURE_HINTS = (
    "not authenticated",
    "not logged in",
    "please log in",
    "login required",
    "authentication required",
    "invalid token",
    "expired token",
    "unauthorized",
)


def _stderr_suggests_auth_failure(stderr_text: str) -> bool:
    """Heuristic: does a non-zero-exit stderr look like an auth problem?

    Used only when the CLI exits non-zero without a structured envelope,
    so we can upgrade the default ``nonzero_exit`` category to the more
    specific ``auth_not_provisioned``. Intentionally conservative — the
    sanctioned path for detecting auth is :meth:`ClaudeCliProvider.healthcheck`
    at agent boot; this is a best-effort fallback for mid-session
    surprises.
    """
    lowered = stderr_text.lower()
    return any(hint in lowered for hint in _AUTH_FAILURE_HINTS)


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
    """Coerce to int, returning 0 for anything that doesn't cleanly convert.

    Handles ``"12"``, ``"12.0"``, ``12.0``, ``None``, arbitrary objects
    (always returns 0 on failure). Used at the untrusted envelope
    boundary so the provider never raises translating tokens.
    """
    result = _int_or_none(val)
    return 0 if result is None else result


__all__ = [
    "REVIEW_RESPONSE_SCHEMA",
    "ClaudeCliAuthError",
    "ClaudeCliProvider",
    "ClaudeCliProviderConfig",
]
