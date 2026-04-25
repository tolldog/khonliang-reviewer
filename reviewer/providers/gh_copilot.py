"""GitHub-Copilot-via-CLI review provider.

Shells out to ``copilot -p`` (the GitHub Copilot CLI's non-interactive
mode) so the agent drives the operator's GitHub Copilot subscription
quota without going through the OpenAI / Anthropic API pay-per-token
paths. Fourth subscription-CLI backend alongside :mod:`reviewer.providers.claude_cli`,
:mod:`reviewer.providers.codex_cli`, and the local-only Ollama path.

Subprocess is the sanctioned integration: GitHub Copilot's OAuth
credential (in ``~/.copilot/`` or the OS keyring) is bound to the
``copilot`` binary; reusing it from a third-party SDK is not
sanctioned. ``copilot -p`` is the supported headless path.

Schema enforcement is prompt-side
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike Claude (``--json-schema``) and Codex (``--output-schema``),
the GitHub Copilot CLI has no schema-enforcement flag. The
``--output-format json`` flag emits a JSONL event stream — useful
for parsing — but does not constrain the model's response shape.
This provider therefore mirrors the Ollama path: it embeds the
:data:`reviewer.providers._prompt.REVIEW_RESPONSE_SCHEMA` in the
prompt body (``include_schema=True``) and parses the model's reply
defensively, coercing wrong-typed fields and unknown enum values.

Tool surface neutralization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``copilot -p`` requires ``--allow-all-tools`` for non-interactive
mode (it's the gate that lets the binary run without prompting for
permission). Combining that with ``--available-tools=`` (empty list)
restricts the surface to nothing — the model can think but not act.
Review is text-in / verdict-out; no tool invocations are needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
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

from reviewer.providers._prompt import build_review_prompt


logger = logging.getLogger(__name__)


class GhCopilotAuthError(RuntimeError):
    """Raised when no GitHub Copilot auth is detected at agent boot.

    Distinct from generic :class:`RuntimeError` so agent boot can
    catch the auth-specific case and surface it to operators without
    mixing it into other startup failures. Mirrors the
    :class:`ClaudeCliAuthError` / :class:`CodexCliAuthError` pattern.
    """


_AUTH_ENV_VARS: tuple[str, ...] = (
    "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
)


@dataclass
class GhCopilotProviderConfig:
    """Construction-time configuration for :class:`GhCopilotProvider`."""

    binary: str = "copilot"
    timeout_seconds: float = 300.0
    #: Empty string causes the provider to omit ``-m`` from argv, in
    #: which case ``copilot -p`` falls back to its own ambient
    #: default model (per the operator's ``copilot`` configuration).
    #: Set this to a specific model id (e.g. ``gpt-5.4``,
    #: ``claude-sonnet-4.5``) when you want a deterministic
    #: per-provider default symmetric with codex_cli / claude_cli.
    default_model: str = ""
    #: Maps to the CLI's ``--effort`` flag, which accepts
    #: ``low|medium|high|xhigh``. Empty string omits the flag and
    #: lets ``copilot -p`` use its compiled-in default. Surfaced as
    #: a config knob so a future rule-table integration can pick
    #: depth per (kind, size) without per-provider plumbing.
    reasoning_effort: str = ""


class GhCopilotProvider(ReviewProvider):
    """Review provider backed by ``copilot -p`` (GitHub Copilot CLI).

    Subscription-backed: runs draw from the operator's GitHub
    Copilot Pro/Pro+/Business quota via ``copilot login`` (OAuth
    stored in ``~/.copilot/`` or the OS keyring) OR via one of the
    env vars in :data:`_AUTH_ENV_VARS`. The CLI accepts a
    fine-grained PAT with the "Copilot Requests" permission, an
    OAuth token from the GitHub Copilot CLI app, or an OAuth token
    from the GitHub CLI (``gh``) app — but NOT classic ``ghp_``
    PATs.
    """

    name = "gh_copilot"

    def __init__(self, config: GhCopilotProviderConfig | None = None):
        self.config = config or GhCopilotProviderConfig()

    async def healthcheck(self) -> None:
        """Verify the binary + auth at agent boot.

        Cheap probes only — no model invocation, no network calls.
        Mirrors the codex_cli auth-path: env-var presence
        short-circuits the OAuth check (operator can run headlessly
        with a PAT). A missing binary always raises FileNotFoundError
        regardless of auth, since the env vars don't help if the
        binary isn't installed.

        Raises:
            FileNotFoundError: if the ``copilot`` binary is not on PATH.
            GhCopilotAuthError: if neither a token env var nor the
                ``~/.copilot/`` credentials directory is present.
        """
        if shutil.which(self.config.binary) is None:
            raise FileNotFoundError(
                f"copilot binary not found at {self.config.binary!r}"
            )
        if not _auth_present():
            raise GhCopilotAuthError(
                "GitHub Copilot CLI is not authenticated; run `copilot "
                "login` (OAuth) or set one of "
                f"{', '.join(_AUTH_ENV_VARS)} (PAT / GH OAuth token) "
                "before using the provider"
            )

    async def review(self, request: ReviewRequest) -> ReviewResult:
        # ``_khonliang_repo_prompts`` / ``_khonliang_example_format`` are
        # in-process-only passthrough metadata (see reviewer.agent).
        # Absent keys or wrong types collapse to the pre-FR behavior —
        # a plain built-in prompt with no repo-side merge.
        repo_prompts = request.metadata.get("_khonliang_repo_prompts")
        example_format = request.metadata.get("_khonliang_example_format")
        # Schema MUST be embedded in the prompt: the GitHub Copilot
        # CLI has no ``--json-schema`` / ``--output-schema``
        # equivalent (see module docstring). Mirrors the Ollama path.
        prompt = build_review_prompt(
            request,
            include_schema=True,
            repo_prompts=repo_prompts,
            example_format=example_format if isinstance(example_format, str) else None,
        )
        started_wall = time.time()
        started_mono = time.monotonic()

        model = _resolve_model(request, self.config.default_model)

        cmd = [
            self.config.binary,
            "-p",
            prompt,
            "--output-format",
            "json",
            # ``--allow-all-tools`` is the non-interactive gate. It
            # allows the binary to run without prompting; tool surface
            # is narrowed independently below.
            "--allow-all-tools",
            # Empty list → no tools available. Review is text-in /
            # verdict-out; the model can reason but cannot edit
            # files, run shell commands, or hit the GitHub MCP
            # server (which auto-loads otherwise).
            "--available-tools=",
            "--no-color",
        ]
        if model:
            cmd += ["-m", model]
        if self.config.reasoning_effort:
            cmd += ["--effort", self.config.reasoning_effort]

        # ``copilot -p <prompt>`` carries the prompt as a single argv
        # element (the CLI has no ``--prompt-file`` / stdin variant as
        # of GitHub Copilot CLI 1.0.36 — verified 2026-04-25). Large
        # diffs can exceed OS ``ARG_MAX`` (~128KB on Linux,
        # ~1MB on macOS) and raise ``OSError`` (errno 7, ``E2BIG``)
        # from ``execve``; the exec call also raises plain ``OSError``
        # for various platform-specific failures beyond the
        # ``FileNotFoundError`` subclass. Catch the broad ``OSError``
        # parent so neither mode crashes the bus skill call —
        # translate to a structured errored ReviewResult instead.
        # ``FileNotFoundError`` is a subclass and gets the
        # binary_not_found category; everything else (E2BIG, ENOMEM,
        # transient platform errors) maps to backend_error so
        # operators can tell the difference.
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return _errored(
                request,
                error=f"copilot binary not found at {self.config.binary!r}",
                error_category="binary_not_found",
                model=model or "copilot",
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )
        except OSError as exc:
            # Most likely E2BIG on a large diff — argv overflow. The
            # error message names the prompt size so operators can
            # see the trigger condition without re-running.
            prompt_bytes = len(prompt.encode("utf-8", errors="replace"))
            return _errored(
                request,
                error=(
                    f"copilot exec failed to spawn (errno={exc.errno}): {exc}. "
                    f"Prompt was {prompt_bytes} bytes; "
                    f"copilot CLI passes the prompt via argv (no stdin / "
                    f"--prompt-file variant), so very large diffs may exceed "
                    f"OS ARG_MAX. Consider chunking the diff or routing "
                    f"large reviews to a different backend."
                ),
                error_category="backend_error",
                model=model or "copilot",
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return _errored(
                request,
                error=f"copilot -p timed out after {self.config.timeout_seconds}s",
                error_category="subprocess_timeout",
                model=model or "copilot",
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
                    f"copilot -p exited with {proc.returncode}"
                    + (f": {stderr_text}" if stderr_text else "")
                ),
                error_category=category,
                model=model or "copilot",
                started_wall=started_wall,
                duration_ms=duration_ms,
            )

        # Pull the final ``assistant.message`` out of the JSONL
        # stream and parse its ``content`` as the structured review
        # JSON we asked for in the prompt.
        try:
            payload, output_tokens = _extract_final_message(stdout)
        except _CopilotEnvelopeError as exc:
            return _errored(
                request,
                error=str(exc),
                error_category="malformed_envelope",
                model=model or "copilot",
                started_wall=started_wall,
                duration_ms=duration_ms,
            )

        result = _parse_payload(
            payload,
            request=request,
            model=model or "copilot",
            started_wall=started_wall,
            duration_ms=duration_ms,
            output_tokens=output_tokens,
        )
        logger.debug(
            "gh_copilot review done: disposition=%s category=%s model=%s "
            "output_tokens=%s duration_ms=%s",
            result.disposition,
            result.error_category or "-",
            result.model,
            output_tokens,
            duration_ms,
        )
        return result


# ----------------------------------------------------------------------
# JSONL stream parsing
# ----------------------------------------------------------------------


class _CopilotEnvelopeError(RuntimeError):
    """Internal signal for parser failures — caught and surfaced as
    ``error_category='malformed_envelope'``."""


def _extract_final_message(stdout: bytes) -> tuple[dict[str, Any], int]:
    """Extract the structured review payload from copilot's JSONL stream.

    Returns ``(payload_dict, output_tokens)``. Searches for the last
    ``assistant.message`` event with ``phase == 'final_answer'`` (or,
    failing that, the last ``assistant.message`` event with non-empty
    ``content``). Parses ``content`` as JSON — that's the structured
    review the prompt asked for.

    Raises :class:`_CopilotEnvelopeError` when the stream is empty,
    can't be parsed line-by-line, or contains no usable assistant
    message.
    """
    text = stdout.decode(errors="replace").strip()
    if not text:
        raise _CopilotEnvelopeError("copilot -p produced empty stdout")

    final_event: dict[str, Any] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Tolerate stray non-JSON lines — copilot occasionally
            # emits empty progress newlines or banner text before
            # JSONL starts. Skipping them is safer than failing the
            # whole stream.
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") != "assistant.message":
            continue
        data = event.get("data") or {}
        if not isinstance(data, dict):
            continue
        if data.get("phase") == "final_answer":
            final_event = event  # latest final_answer wins
        elif final_event is None and isinstance(data.get("content"), str) and data["content"]:
            # Fallback: any assistant.message with content, if no
            # final_answer was emitted (defensive — shouldn't happen
            # in practice but covers older CLI versions).
            final_event = event

    if final_event is None:
        raise _CopilotEnvelopeError(
            "no assistant.message event in copilot -p output"
        )

    data = final_event.get("data") or {}
    content = data.get("content")
    if not isinstance(content, str) or not content.strip():
        raise _CopilotEnvelopeError(
            "copilot -p assistant.message had empty content"
        )

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise _CopilotEnvelopeError(
            f"copilot -p response was not JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise _CopilotEnvelopeError(
            "copilot -p response is not a JSON object "
            f"(type={type(payload).__name__})"
        )

    output_tokens_raw = data.get("outputTokens")
    output_tokens = _safe_int(output_tokens_raw)
    return payload, output_tokens


# ----------------------------------------------------------------------
# Defensive payload coercion (mirrors codex_cli pattern)
# ----------------------------------------------------------------------


_VALID_SEVERITIES: frozenset[str] = frozenset(("nit", "comment", "concern"))


def _coerce_str(val: Any, default: str = "") -> str:
    if isinstance(val, str):
        return val
    return default


def _coerce_severity(val: Any) -> str:
    if isinstance(val, str) and val in _VALID_SEVERITIES:
        return val
    return "comment"


def _parse_payload(
    payload: dict[str, Any],
    *,
    request: ReviewRequest,
    model: str,
    started_wall: float,
    duration_ms: int,
    output_tokens: int,
) -> ReviewResult:
    """Translate a parsed copilot response into a :class:`ReviewResult`.

    The schema is enforced via the prompt body (``include_schema=True``)
    rather than CLI flags, so this parser is the validation point —
    null fields, wrong types, and off-spec severities all get
    coerced to safe defaults rather than leaking into the
    :class:`ReviewFinding` dataclass.
    """
    summary = _coerce_str(payload.get("summary"))
    raw_findings = payload.get("findings") or []
    if not isinstance(raw_findings, list):
        raw_findings = []
    findings = [
        ReviewFinding(
            severity=_coerce_severity(item.get("severity")),  # type: ignore[arg-type]
            title=_coerce_str(item.get("title")),
            body=_coerce_str(item.get("body")),
            category=_coerce_str(item.get("category")),
            path=item.get("path") if isinstance(item.get("path"), str) else None,
            line=_int_or_none(item.get("line")),
            suggestion=item.get("suggestion") if isinstance(item.get("suggestion"), str) else None,
        )
        for item in raw_findings
        if isinstance(item, dict)
    ]

    usage = _build_usage(
        request=request,
        model=model,
        started_wall=started_wall,
        duration_ms=duration_ms,
        output_tokens=output_tokens,
        disposition="posted",
    )

    return ReviewResult(
        request_id=request.request_id,
        summary=summary,
        findings=findings,
        disposition="posted",
        usage=usage,
        backend=GhCopilotProvider.name,
        model=model,
        created_at=started_wall,
    )


def _build_usage(
    *,
    request: ReviewRequest,
    model: str,
    started_wall: float,
    duration_ms: int,
    output_tokens: int,
    disposition: str,
    error: str = "",
    error_category: str = "",
) -> UsageEvent:
    # GitHub Copilot CLI does NOT report prompt-token counts in the
    # JSONL events (verified 2026-04-25). ``outputTokens`` is on the
    # final ``assistant.message`` event; ``input_tokens`` left at 0.
    # ``result.usage.premiumRequests`` exists at end-of-stream but
    # isn't surfaced here — quota observability is a follow-up FR
    # if it becomes load-bearing.
    return UsageEvent(
        timestamp=started_wall,
        backend=GhCopilotProvider.name,
        model=model,
        input_tokens=0,
        output_tokens=output_tokens,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        duration_ms=duration_ms,
        disposition=disposition,  # type: ignore[arg-type]
        request_id=request.request_id,
        repo=str(request.metadata.get("repo", "")),
        pr_number=_int_or_none(request.metadata.get("pr_number")),
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
) -> ReviewResult:
    usage = _build_usage(
        request=request,
        model=model,
        started_wall=started_wall,
        duration_ms=duration_ms,
        output_tokens=0,
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
        backend=GhCopilotProvider.name,
        model=model,
        created_at=started_wall,
    )


# ----------------------------------------------------------------------
# Auth + helper functions
# ----------------------------------------------------------------------


def _auth_present() -> bool:
    """Return True if any of the env vars or the ``~/.copilot/`` dir
    is present.

    **Known blind spot:** macOS systems where the operator authenticated
    via ``copilot login`` (web flow) and the token went straight into
    the system credential store WITHOUT ``~/.copilot/`` ever being
    created can slip past this probe. In practice the directory is
    created the first time ``copilot`` runs (for config / history
    files, even when the token lives in the keyring), so any operator
    who has run the CLI once will hit the directory check correctly.
    Operators on a freshly-provisioned box that hasn't run ``copilot``
    interactively — but has the keyring entry from a prior install —
    must set one of :data:`_AUTH_ENV_VARS` explicitly to be detected
    by this probe.

    The GitHub Copilot CLI does not expose a ``copilot login status``
    or analogous cheap-probe command (verified 2026-04-25, CLI 1.0.36),
    so a true keyring lookup would require either a real
    ``copilot -p`` invocation (which spends subscription quota) or a
    platform-specific keyring read. Both are heavier than the
    probe's cheap-static contract — out of scope.

    Mirrors :func:`reviewer.registry._gh_copilot_auth_present`. Kept
    separate so the provider's healthcheck doesn't import from the
    registry (which would be a layering violation — the registry
    consumes providers, not the other way around).
    """
    for env in _AUTH_ENV_VARS:
        if os.environ.get(env):
            return True
    return os.path.isdir(os.path.expanduser("~/.copilot"))


_AUTH_FAILURE_HINTS = (
    "not authenticated",
    "not logged in",
    "please log in",
    "login required",
    "authentication required",
    "invalid token",
    "expired token",
    "unauthorized",
    "401",
)


def _stderr_suggests_auth_failure(stderr_text: str) -> bool:
    """Heuristic: does a non-zero-exit stderr look like an auth problem?"""
    lowered = stderr_text.lower()
    return any(hint in lowered for hint in _AUTH_FAILURE_HINTS)


def _resolve_model(request: ReviewRequest, default: str) -> str:
    """Use request-supplied model if present, else the provider default."""
    override = request.metadata.get("model")
    if isinstance(override, str) and override:
        return override
    return default


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
    "GhCopilotAuthError",
    "GhCopilotProvider",
    "GhCopilotProviderConfig",
]
