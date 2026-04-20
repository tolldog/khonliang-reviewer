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


logger = logging.getLogger(__name__)


REVIEW_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["summary"],
    "properties": {
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["severity", "title", "body"],
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["nit", "comment", "concern"],
                    },
                    "category": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "path": {"type": ["string", "null"]},
                    "line": {"type": ["integer", "null"]},
                    "suggestion": {"type": ["string", "null"]},
                },
            },
        },
    },
}


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

    async def review(self, request: ReviewRequest) -> ReviewResult:
        prompt = _build_prompt(request)
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
        cmd.append(prompt)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return _errored(
                request,
                error=f"claude binary not found at {self.config.binary!r}",
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.config.timeout_seconds
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return _errored(
                request,
                error=f"claude -p timed out after {self.config.timeout_seconds}s",
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )

        duration_ms = _elapsed_ms(started_mono)

        if proc.returncode != 0:
            stderr_text = stderr.decode(errors="replace").strip()[:500]
            return _errored(
                request,
                error=(
                    f"claude -p exited with {proc.returncode}"
                    + (f": {stderr_text}" if stderr_text else "")
                ),
                started_wall=started_wall,
                duration_ms=duration_ms,
            )

        try:
            envelope = json.loads(stdout.decode(errors="replace"))
        except json.JSONDecodeError as exc:
            return _errored(
                request,
                error=f"claude -p returned non-JSON output: {exc}",
                started_wall=started_wall,
                duration_ms=duration_ms,
            )

        return _parse_envelope(
            envelope,
            request=request,
            started_wall=started_wall,
            fallback_duration_ms=duration_ms,
        )


def _build_prompt(request: ReviewRequest) -> str:
    """Assemble the positional prompt argument from a :class:`ReviewRequest`.

    The prompt instructs Claude to return ONLY the JSON matching the
    schema; the ``--json-schema`` flag additionally enforces validation.
    """
    lines = [
        f"You are a code reviewer for the khonliang ecosystem. Read the {request.kind!r}",
        "content below and return ONLY a JSON object matching the schema you were",
        "given. No prose outside the JSON.",
        "",
    ]
    if request.instructions:
        lines += ["## Review Instructions", "", request.instructions, ""]
    if request.context:
        lines += [
            "## Context",
            "",
            json.dumps(request.context, indent=2, sort_keys=True),
            "",
        ]
    lines += ["## Content", "", request.content]
    return "\n".join(lines)


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
            started_wall=started_wall,
            duration_ms=int(envelope.get("duration_ms") or fallback_duration_ms),
            envelope=envelope,
        )

    duration_ms = int(envelope.get("duration_ms") or fallback_duration_ms)
    model = _pick_primary_model(envelope) or "claude"
    result_text = envelope.get("result") or ""

    try:
        payload = json.loads(result_text)
    except json.JSONDecodeError as exc:
        return _errored(
            request,
            error=f"claude -p result was not JSON: {exc}",
            started_wall=started_wall,
            duration_ms=duration_ms,
            envelope=envelope,
        )

    summary = str(payload.get("summary", ""))
    findings = [
        ReviewFinding(
            severity=item.get("severity", "comment"),
            title=str(item.get("title", "")),
            body=str(item.get("body", "")),
            category=str(item.get("category", "")),
            path=item.get("path"),
            line=item.get("line"),
            suggestion=item.get("suggestion"),
        )
        for item in (payload.get("findings") or [])
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
        try:
            cost = float((stats or {}).get("costUSD") or 0.0)
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
) -> UsageEvent:
    usage_raw = envelope.get("usage") or {}
    return UsageEvent(
        timestamp=started_wall,
        backend=ClaudeCliProvider.name,
        model=model,
        input_tokens=int(usage_raw.get("input_tokens") or 0),
        output_tokens=int(usage_raw.get("output_tokens") or 0),
        cache_read_tokens=int(usage_raw.get("cache_read_input_tokens") or 0),
        cache_creation_tokens=int(usage_raw.get("cache_creation_input_tokens") or 0),
        duration_ms=duration_ms,
        disposition=disposition,  # type: ignore[arg-type]
        request_id=request.request_id,
        repo=str(request.metadata.get("repo", "")),
        pr_number=_int_or_none(request.metadata.get("pr_number")),
        estimated_api_cost_usd=float(envelope.get("total_cost_usd") or 0.0),
        error=error,
    )


def _errored(
    request: ReviewRequest,
    *,
    error: str,
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
    )
    return ReviewResult(
        request_id=request.request_id,
        summary="",
        findings=[],
        disposition="errored",
        error=error,
        usage=usage,
        backend=ClaudeCliProvider.name,
        model=model,
        created_at=started_wall,
    )


def _elapsed_ms(started_mono: float) -> int:
    return int((time.monotonic() - started_mono) * 1000)


def _int_or_none(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


__all__ = [
    "REVIEW_RESPONSE_SCHEMA",
    "ClaudeCliProvider",
    "ClaudeCliProviderConfig",
]
