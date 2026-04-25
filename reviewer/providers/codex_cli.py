"""Codex-via-CLI review provider.

Shells out to ``codex exec --output-schema <schema>`` so the agent drives
the OpenAI Codex ChatGPT subscription quota without going through the
``OPENAI_API_KEY`` pay-per-token path.

Subprocess is the sanctioned integration for subscription-backed Codex:
the ChatGPT subscription token in ``~/.codex/auth.json`` is OAuth-bound
to the ``codex`` binary and is not consumable by third-party SDKs.
``codex exec`` is the only path that respects the subscription quota.

The ``codex exec review`` subcommand is *not* used. It is repo-bound
(diff sourced from ``--uncommitted`` / ``--base`` / ``--commit``, no
stdin diff path) and lacks ``--output-schema``, so it cannot honor the
``ReviewFinding`` contract. ``codex exec`` (the parent) with
``--output-schema`` and stdin-piped prompt is the correct primitive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
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


class CodexCliAuthError(RuntimeError):
    """Raised when ``codex login status`` reports a logged-out state.

    Distinct from generic :class:`RuntimeError` so agent boot can catch
    the auth-specific case and surface it to operators without mixing it
    into other startup failures. ``review()`` error paths attributable
    to auth populate ``ReviewResult.error_category="auth_not_provisioned"``.
    """


def _materialize_schema_file() -> str:
    """Materialize REVIEW_RESPONSE_SCHEMA to a fresh tempfile and return its path.

    Uses :func:`tempfile.mkstemp` so the file is created with ``O_EXCL``
    (no clobber of a hostile pre-existing path) and an unpredictable
    suffix — closes the symlink-following / clobber vector that a
    fixed-name path under ``gettempdir()`` would expose. Best-effort
    cleanup is left to the OS at process exit; the schema file is tiny
    (a few hundred bytes) and a single instance writes it at most once
    via lazy init in :meth:`CodexCliProvider._get_schema_path`.
    """
    fd, path = tempfile.mkstemp(
        prefix="khonliang_codex_review_schema_", suffix=".json"
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(REVIEW_RESPONSE_SCHEMA, f)
    return path


@dataclass
class CodexCliProviderConfig:
    """Construction-time configuration for :class:`CodexCliProvider`."""

    binary: str = "codex"
    timeout_seconds: float = 300.0
    #: Empty string causes the provider to omit ``-m`` from argv, in
    #: which case codex falls back to its own **built-in default
    #: model** — NOT the user's ``~/.codex/config.toml``, because the
    #: subprocess argv passes ``--ignore-user-config`` for
    #: deterministic behavior across operators. Set this to a specific
    #: model id when an operator wants a deterministic per-provider
    #: default regardless of the codex binary's compiled-in choice or
    #: the global selector default.
    default_model: str = ""


class CodexCliProvider(ReviewProvider):
    """Review provider backed by ``codex exec`` non-interactive mode.

    Subscription-backed: runs draw from the ChatGPT-authenticated
    ``~/.codex/auth.json`` quota provisioned by ``codex login``, not
    from ``OPENAI_API_KEY``. Honor :class:`CodexCliProviderConfig`'s
    ``timeout_seconds`` to bound runaway invocations.
    """

    name = "codex_cli"

    def __init__(self, config: CodexCliProviderConfig | None = None):
        self.config = config or CodexCliProviderConfig()
        # Lazy: defer the tempfile write until the first ``review()`` call so
        # that a tempdir / disk-full failure at startup doesn't prevent the
        # whole reviewer agent from booting. Boot-time eagerness was the
        # earlier shape; Copilot flagged it as a bus-wide single point of
        # failure for any deployment that wires codex_cli into the default
        # selector but never actually calls it.
        self._schema_path: str | None = None

    def _get_schema_path(self) -> str:
        """Return the on-disk schema path, materializing it on first use."""
        if self._schema_path is None:
            self._schema_path = _materialize_schema_file()
        return self._schema_path

    async def healthcheck(self) -> None:
        """Verify the CLI is authenticated. Intended for agent boot.

        Two acceptable auth paths:

        1. ChatGPT subscription via ``codex login`` — detected by running
           ``codex login status`` and matching the substring ``"Logged in"``
           in its output. There is no ``--json`` flag on ``login status``
           as of codex 0.125.0, so a string match is the available
           contract.
        2. API-key fallback via ``OPENAI_API_KEY`` env var — codex reads
           this automatically when ``~/.codex/auth.json`` is absent. If
           the env var is set we accept that path without invoking
           ``codex login status``: a logged-out subscription state is fine
           when the API key is present.

        Raises:
            FileNotFoundError: if the ``codex`` binary is not on PATH.
            CodexCliAuthError: if neither auth path is available.
            RuntimeError: for other unexpected failures (non-zero exit
                without an auth-shaped message).
        """
        if os.environ.get("OPENAI_API_KEY"):
            # API-key path is sufficient on its own; skip the login
            # probe so we don't fail on a logged-out subscription when
            # the operator has explicitly chosen the env-var route.
            return

        proc = await asyncio.create_subprocess_exec(
            self.config.binary,
            "login",
            "status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError("codex login status timed out after 15s")

        combined = (stdout + stderr).decode(errors="replace")
        if proc.returncode != 0:
            stderr_text = stderr.decode(errors="replace").strip()[:500]
            raise RuntimeError(
                f"codex login status exited with {proc.returncode}"
                + (f": {stderr_text}" if stderr_text else "")
            )
        if "Logged in" not in combined:
            raise CodexCliAuthError(
                "codex CLI is not authenticated; run `codex login` (ChatGPT) "
                "before using the provider, or set OPENAI_API_KEY for the "
                "API-key fallback"
            )

    async def review(self, request: ReviewRequest) -> ReviewResult:
        # ``_khonliang_repo_prompts`` / ``_khonliang_example_format`` are
        # in-process-only passthrough metadata (see reviewer.agent).
        # Absent keys or wrong types collapse to the pre-FR behavior —
        # a plain built-in prompt with no repo-side merge.
        repo_prompts = request.metadata.get("_khonliang_repo_prompts")
        example_format = request.metadata.get("_khonliang_example_format")
        # ``--output-schema`` enforces the response shape externally —
        # mirrors claude_cli's ``--json-schema`` arrangement, so the
        # prompt body does not need to carry the schema.
        prompt = build_review_prompt(
            request,
            include_schema=False,
            repo_prompts=repo_prompts,
            example_format=example_format if isinstance(example_format, str) else None,
        )
        started_wall = time.time()
        started_mono = time.monotonic()

        model = _resolve_model(request, self.config.default_model)

        # Materialize the schema file before assembling argv. The
        # lazy-init helper writes to a tempfile via ``mkstemp``; under
        # rare conditions (tempdir not writable, disk full, EMFILE on
        # the process) that ``open()`` can raise OSError. Catching
        # here turns what would otherwise be an uncaught crash of the
        # bus skill call into a structured ``errored`` ReviewResult so
        # the caller sees the same shape every other failure path
        # produces. Categorized as ``backend_error`` — the codex
        # binary isn't the problem; the local environment is.
        try:
            schema_path = self._get_schema_path()
        except OSError as exc:
            return _errored(
                request,
                error=f"failed to materialize codex output-schema file: {exc}",
                error_category="backend_error",
                model=model or "codex",
                started_wall=started_wall,
                duration_ms=_elapsed_ms(started_mono),
            )

        cmd = [
            self.config.binary,
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--output-schema",
            schema_path,
        ]
        if model:
            cmd += ["-m", model]
        # ``-`` directs codex to read the prompt from stdin. Diffs easily
        # exceed OS ARG_MAX (~128KB on Linux); piping also keeps the
        # content out of the ``ps`` listing.
        cmd.append("-")

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
                error=f"codex binary not found at {self.config.binary!r}",
                error_category="binary_not_found",
                model=model or "codex",
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
                error=f"codex exec timed out after {self.config.timeout_seconds}s",
                error_category="subprocess_timeout",
                model=model or "codex",
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
                    f"codex exec exited with {proc.returncode}"
                    + (f": {stderr_text}" if stderr_text else "")
                ),
                error_category=category,
                model=model or "codex",
                started_wall=started_wall,
                duration_ms=duration_ms,
            )

        # ``codex exec --output-schema`` writes the schema-validated
        # final agent message to stdout. Progress events go to stderr
        # and are discarded.
        stdout_text = stdout.decode(errors="replace").strip()
        if not stdout_text:
            return _errored(
                request,
                error="codex exec produced empty stdout",
                error_category="malformed_envelope",
                model=model or "codex",
                started_wall=started_wall,
                duration_ms=duration_ms,
            )

        try:
            payload = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            return _errored(
                request,
                error=f"codex exec returned non-JSON output: {exc}",
                error_category="malformed_envelope",
                model=model or "codex",
                started_wall=started_wall,
                duration_ms=duration_ms,
            )

        if not isinstance(payload, dict):
            return _errored(
                request,
                error=(
                    "codex exec returned JSON that is not an object "
                    f"(type={type(payload).__name__})"
                ),
                error_category="malformed_envelope",
                model=model or "codex",
                started_wall=started_wall,
                duration_ms=duration_ms,
            )

        result = _parse_payload(
            payload,
            request=request,
            model=model or "codex",
            started_wall=started_wall,
            duration_ms=duration_ms,
        )
        logger.debug(
            "codex_cli review done: disposition=%s category=%s model=%s duration_ms=%s",
            result.disposition,
            result.error_category or "-",
            result.model,
            result.usage.duration_ms if result.usage else 0,
        )
        return result


def _resolve_model(request: ReviewRequest, default: str) -> str:
    """Use request-supplied model if present, else the provider default.

    Returns an empty string when neither is set, in which case the
    provider omits ``-m`` from argv. Note: the subprocess argv also
    includes ``--ignore-user-config``, so codex does **not** read
    ``~/.codex/config.toml`` for a default model in that case — it
    falls back to its own built-in default model selection (whatever
    the codex binary's compiled-in default is at the running version).
    Operators who want a deterministic per-provider default should set
    ``CodexCliProviderConfig.default_model`` explicitly (or pass
    ``model`` per request).
    """
    override = request.metadata.get("model")
    if isinstance(override, str) and override:
        return override
    return default


_VALID_SEVERITIES: frozenset[str] = frozenset(("nit", "comment", "concern"))


def _coerce_str(val: Any, default: str = "") -> str:
    """Defensive str coercion that treats ``None`` / non-strings as ``default``.

    ``str(None)`` would produce the literal string ``"None"`` which then
    leaks into the ``ReviewResult.summary`` and ``ReviewFinding.title``
    fields — surprising operators reading the output. The reviewer's
    bus-boundary-validation principle (per ``CLAUDE.md``) says external
    payloads must be validated before constructing library dataclasses;
    this function is the validation point for string-typed fields.
    """
    if isinstance(val, str):
        return val
    return default


def _coerce_severity(val: Any) -> str:
    """Map an external severity value to the contract enum or default to ``comment``.

    ``ReviewFinding.severity`` is a ``Literal["nit", "comment", "concern"]``;
    a typo or wrong type from the model output would silently land an
    out-of-contract value in the dataclass. Coerce to ``comment`` (a
    safe middle severity) when the value is unknown so downstream
    severity_floor filtering still works correctly.
    """
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
) -> ReviewResult:
    """Translate a schema-validated codex payload into a :class:`ReviewResult`.

    Even though ``--output-schema`` enforces the response shape upstream,
    the parser still validates types defensively. The codex JSON-schema
    surface is a recently-added subprocess contract; any future schema
    laxity (or an off-spec model that emits ``null`` for required
    fields) would otherwise produce surprising values like
    ``summary == "None"`` or out-of-enum severities reaching the bus.
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
        disposition="posted",
    )

    return ReviewResult(
        request_id=request.request_id,
        summary=summary,
        findings=findings,
        disposition="posted",
        usage=usage,
        backend=CodexCliProvider.name,
        model=model,
        created_at=started_wall,
    )


def _build_usage(
    *,
    request: ReviewRequest,
    model: str,
    started_wall: float,
    duration_ms: int,
    disposition: str,
    error: str = "",
    error_category: str = "",
) -> UsageEvent:
    # ``codex exec`` without ``--json`` does not emit token counts on
    # stdout — only the schema-validated final message. Token fields
    # left at 0 mirrors the ollama path and is acceptable for the
    # subscription-quota use case (cost is tracked elsewhere via
    # default_pricing.yaml). Switch to ``--json`` event-stream parsing
    # in a follow-up FR if per-call token detail becomes load-bearing.
    return UsageEvent(
        timestamp=started_wall,
        backend=CodexCliProvider.name,
        model=model,
        input_tokens=0,
        output_tokens=0,
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
        backend=CodexCliProvider.name,
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
    "auth.json",
)


def _stderr_suggests_auth_failure(stderr_text: str) -> bool:
    """Heuristic: does a non-zero-exit stderr look like an auth problem?

    Used only when ``codex exec`` exits non-zero, so we can upgrade the
    default ``nonzero_exit`` category to ``auth_not_provisioned``. The
    sanctioned path for detecting auth is :meth:`CodexCliProvider.healthcheck`
    at agent boot; this is a best-effort fallback for mid-session
    surprises (e.g. token revoked while the agent is running).
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


__all__ = [
    "CodexCliAuthError",
    "CodexCliProvider",
    "CodexCliProviderConfig",
]
