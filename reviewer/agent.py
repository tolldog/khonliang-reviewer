"""Reviewer as a bus agent.

Skills live here. Providers + selection policy live in sibling modules:

- ``reviewer.providers`` — concrete :class:`ReviewProvider`
  implementations (Ollama, Claude-via-CLI).
- ``reviewer.selector`` — resolves ``(backend, model)`` to a provider
  instance. Currently a stubbed selector (caller override + config
  default); rule-table-driven selection wires in once WU4 (rule table)
  lands on main.

Usage::

    # Install into the bus
    python -m reviewer.agent install --id reviewer-primary --bus http://localhost:8787 --config config.yaml

    # Start (normally done by the bus on boot)
    python -m reviewer.agent --id reviewer-primary --bus http://localhost:8787 --config config.yaml

    # Uninstall
    python -m reviewer.agent uninstall --id reviewer-primary --bus http://localhost:8787
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import typing
import uuid
from typing import Any

from khonliang_bus import BaseAgent, Skill, Welcome, WelcomeEntryPoint, handler
from dataclasses import replace as dataclass_replace

from khonliang_reviewer import (
    SEVERITY_ORDER,
    ReviewFinding,
    ReviewRequest,
    ReviewResult,
    UsageEvent,
    severity_rank,
)

from reviewer.distill import run_pipeline
from reviewer.rules.distill import Audience, DistillConfig

from reviewer.config.prompts import (
    RepoPrompts,
    load_repo_prompts,
)
from reviewer.config.repo import (
    RepoConfig,
    RepoConfigUnreachableError,
    load as load_repo_config,
    provider_to_vendor,
)
from reviewer.github_client import GithubClientError, ReviewerGithubClient
from reviewer.providers import (
    ClaudeCliProvider,
    ClaudeCliProviderConfig,
    CodexCliProvider,
    CodexCliProviderConfig,
    GhCopilotProvider,
    GhCopilotProviderConfig,
    OllamaProvider,
    OllamaProviderConfig,
)
from reviewer.pricing_seed import load_default_pricing
from reviewer.registry import ProviderRegistry
from reviewer.rules import PolicyDecision, PolicyInput, decide
from reviewer.selector import ProviderSelector, SelectorConfig, UnknownBackendError
from reviewer.storage import UsageStore, open_usage_store


logger = logging.getLogger(__name__)


_REQUEST_ID_PREFIX = "rev-"


def _generate_request_id() -> str:
    return f"{_REQUEST_ID_PREFIX}{uuid.uuid4().hex[:16]}"


def _as_dict(val: Any) -> dict[str, Any]:
    """Return ``val`` as a dict, or an empty dict when it isn't one."""
    return val if isinstance(val, dict) else {}


_SEVERITY_LABELS = {
    "nit": "🟢 Nit",
    "comment": "🟡 Comment",
    "concern": "🔴 Concern",
}


#: GitHub-supported values for the ``event`` parameter on a review
#: submission the reviewer agent is allowed to use. ``APPROVE`` is
#: deliberately excluded: FR fr_developer_e72d8835 pins approval
#: authority to humans. If a future FR ever grants machine approval,
#: it should add the value here behind an explicit opt-in flag rather
#: than being broadly accepted on the ``event`` arg.
_VALID_REVIEW_EVENTS = frozenset({"COMMENT", "REQUEST_CHANGES", "PENDING"})


#: Built-in default severity floor. ``"nit"`` means "keep everything" —
#: noise reduction is strictly opt-in so adding severity_floor to the
#: agent can't silently change the review output of repos that haven't
#: configured it. Matches the FR's precedence tail.
_DEFAULT_SEVERITY_FLOOR = "nit"


#: In-process-only passthrough keys on :attr:`ReviewRequest.metadata`.
#:
#: ``_khonliang_repo_prompts`` carries a :class:`RepoPrompts` snapshot
#: into providers so they can merge repo-side calibration material into
#: the LLM prompt. ``_khonliang_example_format`` carries the active
#: model config's ``example_format`` string (``xml`` | ``json`` |
#: ``markdown``) so the wrapping layer knows which vendor framing to
#: use for example blocks.
#:
#: Underscore-prefixed convention signals to anything reading metadata
#: naively (GitHub logs, serializers, bus event consumers) that these
#: keys are implementation detail — skip them if you don't know what
#: they mean. The usage event construction in both providers reads
#: specific named fields (``repo``, ``pr_number``), not the whole
#: metadata dict, so these non-JSON-serializable values never leave
#: the in-process path.
_METADATA_REPO_PROMPTS_KEY = "_khonliang_repo_prompts"
_METADATA_EXAMPLE_FORMAT_KEY = "_khonliang_example_format"

#: Reserved prefix for internal-only passthrough keys on
#: :attr:`ReviewRequest.metadata`. Every key carrying values the agent
#: itself injects (see ``_METADATA_*`` constants above) starts with this
#: prefix. Callers MUST NOT supply keys with this prefix via
#: ``args["metadata"]`` — :func:`_strip_reserved_metadata` scrubs any
#: that slip in before the caller dict is merged into the request. The
#: scrub is the single, authoritative defense: downstream providers and
#: prompt-assembly code can then trust that when a reserved key is
#: present the agent put it there, and can use the expected type
#: without redundant ``isinstance`` checks on a trust boundary the
#: agent already enforces.
_RESERVED_METADATA_PREFIX = "_khonliang_"


def _strip_reserved_metadata(user_metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``user_metadata`` with reserved keys removed.

    Keys starting with :data:`_RESERVED_METADATA_PREFIX` are reserved
    for internal agent-side passthrough (repo-prompts snapshots,
    example-format hints). Accepting them from caller-supplied metadata
    would let an untrusted caller inject values into prompt-assembly
    code paths that expect specific in-process Python types — e.g.
    a :class:`RepoPrompts` instance the agent just built from a git-
    show read. Stripping before merge keeps the boundary simple: by
    the time the request reaches a provider, any reserved key it sees
    was put there by the agent itself.
    """
    if not user_metadata:
        return {}
    return {
        key: value
        for key, value in user_metadata.items()
        if not (isinstance(key, str) and key.startswith(_RESERVED_METADATA_PREFIX))
    }


class SeverityFloorError(ValueError):
    """Raised when a caller-supplied ``severity_floor`` isn't a known value.

    Subclass of :class:`ValueError` so existing ``except ValueError``
    blocks catch it, but distinct enough that callers wanting to branch
    on "validation error vs. other ValueError" can key on the class.
    """


def _validate_severity_floor(value: str, *, source: str) -> str:
    """Validate ``value`` is a known severity string; return it unchanged.

    Empty string is rejected by the caller (it's the "not set" sentinel
    at the skill-arg layer; callers resolve precedence before calling
    here). Unknown values raise :class:`SeverityFloorError` with a
    message naming ``source`` so operators can tell whether a skill
    arg or a config file is to blame.
    """
    if value not in SEVERITY_ORDER:
        raise SeverityFloorError(
            f"{source}: severity_floor={value!r} is not valid; "
            f"expected one of {list(SEVERITY_ORDER)}"
        )
    return value


class AudienceError(ValueError):
    """Raised when caller-supplied ``audience`` isn't a known value.

    The skill-arg layer accepts the empty string as a sentinel
    meaning "use the agent_consumption default"; non-empty values
    must be one of :data:`reviewer.rules.distill.Audience`'s
    Literal members or this error fires and the handler returns
    a structured error response.
    """


_VALID_AUDIENCES: frozenset[str] = frozenset(typing.get_args(Audience))
_DEFAULT_AUDIENCE: Audience = "agent_consumption"


def _resolve_audience(value: Any) -> Audience:
    """Resolve the caller's ``audience`` arg to a validated literal.

    Empty / non-string / unset values fall back to
    ``agent_consumption`` — the safe default the rule table also
    emits when no audience-specific row matches. Non-empty strings
    that don't match a known audience raise :class:`AudienceError`.
    """
    if not isinstance(value, str) or not value:
        return _DEFAULT_AUDIENCE
    if value not in _VALID_AUDIENCES:
        raise AudienceError(
            f"audience={value!r} is not valid; expected one of "
            f"{sorted(_VALID_AUDIENCES)}"
        )
    return value  # type: ignore[return-value]


class ConsensusError(ValueError):
    """Raised when caller-supplied ``consensus_runs`` / ``consensus_min``
    fail validation.

    Subclass of :class:`ValueError` so existing ``except ValueError``
    blocks catch it, but distinct enough that callers wanting to branch
    on "validation error vs. other ValueError" can key on the class.
    """


class EvaluatorError(ValueError):
    """Raised when caller-supplied ``evaluator_hot`` spec fails validation.

    The skill-arg layer accepts ``"<backend>:<model>"`` strings.
    Empty string is the absence sentinel (no evaluator); non-empty
    values must parse cleanly and reference a registered backend, or
    they raise this error and the handler converts it to a structured
    error response.
    """


def _parse_evaluator_spec(spec: str) -> tuple[str, str]:
    """Parse ``"<backend>:<model>"`` into the pair.

    Both halves must be non-empty; the colon is the only separator
    (model strings rarely contain colons in this codebase, and the
    Ollama wire spec ``model:tag`` uses a different syntax in the
    tag namespace — model strings here are the *tag-included* form
    treated as opaque after the first colon). Validation errors raise
    :class:`EvaluatorError` with a message naming the offending value
    so callers can correct it.
    """
    head, _, tail = spec.partition(":")
    head = head.strip()
    tail = tail.strip()
    if not head or not tail:
        raise EvaluatorError(
            f"evaluator_hot={spec!r} must be of the form "
            f"'<backend>:<model>' with both halves non-empty "
            f"(after whitespace trimming)"
        )
    return head, tail


def _coerce_consensus_int(value: Any, *, default: int) -> int:
    """Return a positive int if ``value`` is one, else ``default``.

    The bus boundary delivers ints as JSON; the schema default is ``1``
    (effectively absent — single-pass review). Treat any non-int /
    non-positive payload as absent rather than crashing the handler;
    explicit out-of-range values still get rejected by
    :func:`_validate_consensus` so callers don't silently land in a
    weird state. Accepts ``int`` only (rejects ``bool`` because
    ``bool`` is an ``int`` subclass and ``consensus_runs=True`` would
    otherwise quietly become 1).
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value > 0:
        return value
    return default


#: Hard cap on ``consensus_runs`` to keep a typo from spawning a
#: cluster of concurrent provider calls. 16 is generous — best-of-N
#: literature typically uses 3-7; the cb081fa8 FR's token-budget
#: calculation assumes N ≤ 5. Above 16 is almost certainly a mistake
#: or an abuse vector and the agent fails fast rather than overloading
#: the local Ollama (or burning subscription quota on Claude-CLI).
_CONSENSUS_RUNS_MAX = 16


def _validate_consensus(runs: int, min_count: int) -> None:
    """Cross-validate the consensus pair after coercion.

    ``runs >= 1`` and ``min_count >= 1`` are guaranteed by
    :func:`_coerce_consensus_int`. This function performs the
    cross-checks that don't fit in the per-field coercion:

    - ``runs <= _CONSENSUS_RUNS_MAX`` — typo / abuse guard against
      spawning a flood of concurrent calls.
    - ``min_count <= runs`` — catches caller bugs like
      ``consensus_runs=2, consensus_min=3`` where no finding could
      ever survive (and the trickier
      ``consensus_runs=1, consensus_min=2`` which would otherwise
      silently take the single-call fast path).

    Raises :class:`ConsensusError` with a message naming the
    offending pair so callers can act on the validation failure.
    """
    if runs > _CONSENSUS_RUNS_MAX:
        raise ConsensusError(
            f"consensus_runs={runs} exceeds max {_CONSENSUS_RUNS_MAX}; "
            f"refusing to spawn more concurrent provider calls"
        )
    if min_count > runs:
        raise ConsensusError(
            f"consensus_min={min_count} cannot exceed "
            f"consensus_runs={runs}; no finding could survive"
        )


def _consensus_finding_key(
    finding: ReviewFinding,
) -> tuple[str, str, int, str]:
    """Anchor key used to detect "same finding" across consensus runs.

    Inline findings (path **and** line both set) anchor on
    ``(severity, path, line, normalized_title)`` — the line gives the
    strongest signal that two findings refer to the same code site.
    Summary-level findings (path / line absent) anchor on
    ``(severity, "", 0, normalized_title)`` so the title carries the
    overlap detection alone.

    Findings with ``path`` set but ``line=None`` (or vice versa) are
    treated as summary-level — same as the both-absent case — so they
    consolidate with other summary-level findings sharing the same
    title rather than producing a third "weird" key shape that
    nothing else groups with. (Copilot R4 PR#37: the prior code
    used ``finding.path or ""`` and ``finding.line or 0`` independently,
    which silently created a partial-anchor key for the path-only /
    line-only case and contradicted the docstring.)

    Title normalization (lowercase + whitespace collapse) tolerates
    minor wording drift across runs without softening the
    "10×-outlier survives unchanged" property — a unique outlier
    still has a unique normalized title and is never grouped with an
    unrelated finding.
    """
    title_norm = " ".join(finding.title.lower().split())
    if finding.path and finding.line is not None:
        return (finding.severity, finding.path, finding.line, title_norm)
    return (finding.severity, "", 0, title_norm)


def _consolidate_consensus_results(
    results: list[ReviewResult],
    *,
    min_count: int,
    base_request_id: str,
) -> ReviewResult:
    """Merge N successful results into one via finding-overlap consolidation.

    Findings are grouped by :func:`_consensus_finding_key`; groups
    smaller than ``min_count`` are dropped. Surviving groups
    contribute their first-occurring finding verbatim — no body
    merging, no severity averaging. Picking the first occurrence (in
    run order) is deterministic and preserves the canonical bytes the
    model produced rather than synthesizing prose that no model wrote.

    Usage records are summed (tokens, cost) except for ``duration_ms``
    which takes the max (the runs are concurrent so wall-clock is the
    longest, not the sum). The returned result reports the base
    ``request_id`` and the first run's ``backend`` / ``model`` /
    ``disposition`` — they're identical across runs in the typical
    case (same provider, same model, same outcome).
    """
    groups: dict[tuple[str, str, int, str], list[ReviewFinding]] = {}
    insertion_order: list[tuple[str, str, int, str]] = []
    for result in results:
        for finding in result.findings:
            key = _consensus_finding_key(finding)
            if key not in groups:
                groups[key] = []
                insertion_order.append(key)
            groups[key].append(finding)

    surviving = [
        groups[key][0]
        for key in insertion_order
        if len(groups[key]) >= min_count
    ]

    base = results[0]
    summary = next((r.summary for r in results if r.summary), "")
    usage_events = [r.usage for r in results if r.usage is not None]
    usage = _merge_usage_events(usage_events, base_request_id) if usage_events else None

    return ReviewResult(
        request_id=base_request_id,
        summary=summary,
        findings=surviving,
        disposition=base.disposition,
        error="",
        error_category="",
        usage=usage,
        backend=base.backend,
        model=base.model,
    )


def _merge_usage_events(
    events: list[UsageEvent],
    base_request_id: str,
) -> UsageEvent:
    """Merge the available usage events for a consensus request.

    Caller guarantees ``events`` is non-empty, but the originating
    review runs need not all have succeeded or emitted usage. Two
    call sites exercise this helper:

    1. Success path (:func:`_consolidate_consensus_results`) — every
       run completed successfully and contributes a usage event.
    2. Error path (:meth:`ReviewerAgent._run_consensus`) — the
       first errored run is surfaced, but usage from runs that
       completed before the error is still merged so cost-accounting
       captures real spend. The error-path caller subsequently
       overrides ``disposition`` / ``error`` / ``error_category``
       from the failing run on the returned ``UsageEvent`` because
       this helper inherits those fields from the first event
       (typically a successful run).

    Token / cost fields sum across the provided usage events (true
    compute spend for the usage we observed); ``duration_ms`` is the
    max of the per-run durations (concurrent wall-clock, not serial).
    Identity fields (``backend``, ``model``, ``repo``, ``pr_number``,
    ``timestamp``, ``disposition``) take the first event's value —
    they're invariant across runs in the typical case, and the
    first-event pick keeps the trace anchored at the earliest start
    time.

    ``request_id`` is rewritten to ``base_request_id`` (sans the
    ``-rN`` per-run suffix) so the usage record is searchable under
    the same id the caller asked for.
    """
    base = events[0]
    return UsageEvent(
        timestamp=base.timestamp,
        backend=base.backend,
        model=base.model,
        input_tokens=sum(e.input_tokens for e in events),
        output_tokens=sum(e.output_tokens for e in events),
        cache_read_tokens=sum(e.cache_read_tokens for e in events),
        cache_creation_tokens=sum(e.cache_creation_tokens for e in events),
        duration_ms=max(e.duration_ms for e in events),
        disposition=base.disposition,
        request_id=base_request_id,
        repo=base.repo,
        pr_number=base.pr_number,
        estimated_api_cost_usd=sum(e.estimated_api_cost_usd for e in events),
        error="",
        error_category="",
        # Reset to 0 — the severity_floor pass that runs after
        # consensus will bump this on the merged result.
        findings_filtered_count=0,
    )


def _load_repo_prompts_from_context(
    context: dict[str, Any],
) -> RepoPrompts | None:
    """Best-effort load of ``.reviewer/prompts/`` from context hints.

    Symmetric with :func:`_resolve_repo_severity_floor`: activates only
    when the caller threads ``repo_root`` + ``base_sha`` through
    ``context``. Orchestrators without a local clone (``review_pr``
    today, which fetches via the GitHub API) skip this layer and the
    review falls back to the built-in prompt only.

    Returns ``None`` — not an empty :class:`RepoPrompts` — when the
    layer is not applicable. Infrastructure failures (shallow clone,
    git subsystem error) log at warning and collapse to ``None``; the
    reviewer still runs, just without the repo-side prompt additions.
    A hard failure here would defeat the point of the whole graceful-
    absence contract that the rest of the ``.reviewer/`` loader
    carries.
    """
    repo_root = context.get("repo_root")
    base_sha = context.get("base_sha")
    if not isinstance(repo_root, str) or not repo_root:
        return None
    if not isinstance(base_sha, str) or not base_sha:
        return None
    try:
        prompts: RepoPrompts = load_repo_prompts(repo_root, base_sha=base_sha)
    except RepoConfigUnreachableError as exc:
        logger.warning("repo prompts load skipped: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("repo prompts load failed: %s", exc)
        return None
    if prompts.is_empty:
        # Normalise "tree present but empty" to "tree absent" so
        # downstream code has one signal for "nothing to merge" —
        # avoids branching on ``.is_empty`` at every call site.
        return None
    return prompts


def _load_repo_config_from_context(
    context: dict[str, Any],
) -> RepoConfig | None:
    """Best-effort load of ``.reviewer/config.yaml`` from context hints.

    Consolidates the git-plumbing + YAML-parse path that was previously
    duplicated inside :func:`_resolve_repo_severity_floor` and
    :func:`_resolve_example_format_from_config`. Callers that need the
    config for multiple resolutions (e.g. ``handle_review_text`` resolves
    both the severity floor and the model-config example_format) should
    invoke this helper once and thread the result into each resolver —
    otherwise every resolver re-runs ``git show`` and re-parses the YAML.

    Returns ``None`` when the context lacks ``repo_root`` / ``base_sha``
    hints, or when the config is unreachable / fails to load. The
    graceful-absence contract is preserved: a config-loader failure
    never fails the review.
    """
    repo_root = context.get("repo_root")
    base_sha = context.get("base_sha")
    if not isinstance(repo_root, str) or not repo_root:
        return None
    if not isinstance(base_sha, str) or not base_sha:
        return None
    try:
        return load_repo_config(repo_root, base_sha=base_sha)
    except RepoConfigUnreachableError as exc:
        logger.warning("repo config read skipped: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("repo config read failed: %s", exc)
        return None


def _resolve_example_format_from_config(
    cfg: RepoConfig | None,
    *,
    kind: str,
    vendor: str,
    model: str,
) -> str | None:
    """Consult ``.reviewer/models/<vendor>/<model>.yaml`` for ``example_format``.

    Accepts a pre-loaded :class:`RepoConfig` (or ``None`` when the
    context didn't yield one) rather than re-loading from
    ``repo_root``/``base_sha`` — the caller is expected to have obtained
    the config via :func:`_load_repo_config_from_context` and pass it in
    so the underlying ``git show`` + YAML parse happens once per
    ``handle_review_text`` invocation.

    Returns the string value verbatim when present and a string; any
    other value (missing file, wrong type, empty) returns ``None`` so
    the caller falls through to the markdown default inside
    :func:`build_review_prompt`.
    """
    if cfg is None:
        return None
    resolved = cfg.resolve(kind=kind, vendor=vendor, model=model)
    value = resolved.get("example_format")
    if isinstance(value, str) and value:
        return value
    return None


def _resolve_repo_severity_floor(cfg: RepoConfig | None) -> str | None:
    """Best-effort read of ``review.severity_floor`` from ``.reviewer/config.yaml``.

    Accepts a pre-loaded :class:`RepoConfig` (or ``None``) rather than
    re-loading from context — the caller is expected to obtain the config
    once via :func:`_load_repo_config_from_context` and thread it into
    both this helper and
    :func:`_resolve_example_format_from_config`. This prevents the
    duplicate git-plumbing + YAML-parse that would otherwise fire per
    ``handle_review_text`` invocation.

    Returns ``None`` when the layer is not applicable (config absent /
    value unset). The skill-arg layer is still honored by the caller
    because it's resolved first.
    """
    if cfg is None:
        return None
    return cfg.severity_floor


def _format_for_github(
    review_result: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Translate a :class:`ReviewResult` dict into GitHub review shape.

    Returns ``(summary_body, inline_comments)``:

    - ``summary_body`` — top-level review body. Always carries the
      review summary; summary-level findings (those without
      ``path``/``line``) are appended as a short bullet list so they
      don't get lost.
    - ``inline_comments`` — list of GitHub inline-comment dicts
      ``{"path", "line", "side": "RIGHT", "body"}``, one per finding
      that carries ``path`` + ``line``. Body includes a severity
      label, the finding title, its body, and an optional
      ````suggestion```` block.
    """
    summary = str(review_result.get("summary") or "")
    findings = review_result.get("findings") or []
    if not isinstance(findings, list):
        findings = []

    inline_comments: list[dict[str, Any]] = []
    summary_extras: list[str] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        title = str(finding.get("title") or "").strip()
        body_text = str(finding.get("body") or "").strip()
        severity = str(finding.get("severity") or "comment")
        label = _SEVERITY_LABELS.get(severity, severity)
        path = finding.get("path")
        line = finding.get("line")
        suggestion = finding.get("suggestion")

        # GitHub review-comments require a string path and a POSITIVE
        # line number; ``bool`` is excluded explicitly because it
        # subclasses ``int`` (``True`` would otherwise read as line 1).
        # Findings that don't satisfy both fall back to summary-level
        # notes to avoid 422 Unprocessable Entity from GitHub.
        anchored = (
            isinstance(path, str)
            and bool(path)
            and isinstance(line, int)
            and not isinstance(line, bool)
            and line > 0
        )
        if anchored:
            parts = [f"**{label} — {title}**" if title else f"**{label}**"]
            if body_text:
                parts.append(body_text)
            if isinstance(suggestion, str) and suggestion:
                parts.append(f"```suggestion\n{suggestion}\n```")
            inline_comments.append(
                {
                    "path": path,
                    "line": int(line),
                    "side": "RIGHT",
                    "body": "\n\n".join(parts),
                }
            )
        else:
            headline = f"- **{label}**"
            if title:
                headline += f" — {title}"
            if body_text:
                headline += f": {body_text}"
            summary_extras.append(headline)

    body = summary.strip()
    if summary_extras:
        # Always mark the extras with a heading so the bullet list has
        # a clear context, even when the model returned an empty
        # top-level summary.
        separator = "\n\n" if body else ""
        body += f"{separator}### Additional notes\n\n" + "\n".join(summary_extras)
    if not body:
        body = "No findings."
    return body, inline_comments


def _estimate_diff_size(content: str, kind: str) -> tuple[int, int]:
    """Rough (line_count, file_count) for rule-table input.

    Only non-zero for ``kind == "pr_diff"``. Cheap to compute and good
    enough for coarse-grained routing; callers with authoritative
    counts can pass them through ``context["diff_line_count"]`` /
    ``context["diff_file_count"]`` to override.
    """
    if kind != "pr_diff":
        return 0, 0
    line_count = content.count("\n")
    file_count = content.count("\ndiff --git")
    if content.startswith("diff --git"):
        file_count += 1
    return line_count, file_count


def _resolve_payload_arg(args: dict[str, Any], *, prefer: str = "content") -> str:
    """Resolve the review payload from either ``content`` or ``diff``.

    Both ``review_text`` and ``review_diff`` skills accept either arg
    name so subagents can call whichever feels natural — the two
    skills differ in framing (freeform text vs unified-diff bytes),
    not in field name. Resolution rules:

    - The ``prefer``-named arg wins when it's a non-empty string.
    - The other arg is consulted when the preferred one is missing,
      empty, or non-string.
    - Returns ``""`` when neither carries a non-empty string (the
      caller surfaces "required" as an error).

    The ``prefer`` parameter lets each skill's handler keep its
    canonical name authoritative on ties: review_text prefers
    ``content``, review_diff prefers ``diff``.
    """
    primary = args.get(prefer)
    if isinstance(primary, str) and primary:
        return primary
    other_name = "diff" if prefer == "content" else "content"
    secondary = args.get(other_name)
    if isinstance(secondary, str) and secondary:
        return secondary
    return ""


def _coerce_default_models(val: Any) -> dict[str, str]:
    """Coerce ``config['default_models']`` to a ``dict[str, str]``.

    The bus boundary may deliver any YAML-shape (None, list, int,
    nested dict) under that key; the selector contract is a flat
    ``backend -> model`` mapping. Non-dict inputs collapse to an
    empty dict so a misconfigured ``config.yaml`` falls through to
    the legacy ``default_model`` path rather than crashing the
    selector. Non-string keys / values are skipped silently — same
    "treat malformed as absent" pattern used elsewhere in this
    module.
    """
    if not isinstance(val, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in val.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def _positive_float_or_none(val: Any) -> float | None:
    """Coerce ``val`` to a positive float, or None when it's falsy/invalid.

    The ``usage_summary`` skill accepts ``since``/``until`` as floats
    with a default of 0. Treating 0 as "no filter" lets callers omit
    the field on the bus wire without constructing explicit nulls.
    """
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def _policy_input_for(
    *, kind: str, content: str, context: dict[str, Any]
) -> PolicyInput:
    """Build a :class:`PolicyInput` from the pieces available in the handler.

    Callers can supply authoritative ``diff_line_count`` /
    ``diff_file_count`` / ``profile`` in ``context``; otherwise they're
    estimated from ``content`` (for diffs) or left empty.
    """
    est_lines, est_files = _estimate_diff_size(content, kind)
    return PolicyInput(
        kind=kind,
        diff_line_count=int(context.get("diff_line_count") or est_lines),
        diff_file_count=int(context.get("diff_file_count") or est_files),
        profile=(
            context.get("profile")
            if isinstance(context.get("profile"), dict)
            else None
        ),
    )


class ReviewerAgent(BaseAgent):
    """Bus-native reviewer agent.

    Exposes ``review_text`` + ``review_diff`` skills. Provider selection
    uses the stubbed :class:`ProviderSelector` — caller override wins,
    config default otherwise. Rule-table routing joins once WU4 merges.

    The agent lazily constructs its provider selector on first use from
    its ``config.yaml``. Tests inject a pre-built selector via the
    ``selector=`` kwarg to avoid touching real Ollama / Claude CLI
    subprocesses.
    """

    # Cold-start orientation surface (fr_khonliang-bus-lib_6a82732c).
    WELCOME = Welcome(
        role="cross-vendor code review authority",
        mission=(
            "Reviews diffs, PRs, and free-form code/text via pluggable LLM "
            "backends (Ollama / Claude CLI / OpenAI). Produces structured "
            "findings (severity / category / location / observation / "
            "suggestion). Hot path is local Ollama (qwen2.5-coder:14b "
            "default — 83% accuracy / 0 external tokens per the 2026-04-22 "
            "benchmark). External LLMs are escalation, not default."
        ),
        not_responsible_for=[
            "code authorship (Claude session via developer)",
            "FR / spec / milestone lifecycle (developer)",
            "ingestion (researcher)",
        ],
        delegates_to={
            "developer": "PR + branch lifecycle around the review",
            "store": "review-finding artifacts (planned per fr_reviewer_127af052)",
        },
        entry_points=[
            WelcomeEntryPoint(
                skill="review_diff",
                when_to_use="review a raw git diff (preferred for pre-push checks; pass diff bytes, not summaries)",
            ),
            WelcomeEntryPoint(
                skill="review_pr",
                when_to_use="review a GitHub PR by URL — fetches the diff and PR metadata",
            ),
            WelcomeEntryPoint(
                skill="review_text",
                when_to_use="review free-form code or text snippets that aren't a diff or PR",
            ),
            WelcomeEntryPoint(
                skill="usage_summary",
                when_to_use="token + cost summary across recent calls; useful for budget tracking",
            ),
        ],
    )

    agent_id = "reviewer-primary"
    agent_type = "reviewer"
    module_name = "reviewer.agent"

    def __init__(
        self,
        *,
        selector: ProviderSelector | None = None,
        registry: ProviderRegistry | None = None,
        usage_store: UsageStore | None = None,
        github_client: ReviewerGithubClient | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._injected_selector = selector
        self._cached_selector: ProviderSelector | None = None
        self._injected_registry = registry
        self._cached_registry: ProviderRegistry | None = None
        self._injected_store = usage_store
        self._cached_store: UsageStore | None = None
        self._injected_github = github_client
        self._cached_github: ReviewerGithubClient | None = None

    async def start(self) -> None:
        """Eager-init the usage store so the SQLite file lands on launch.

        Operators rely on seeing ``data/reviewer.db`` appear as soon as
        the agent boots (for tailing, backups, monitoring) rather than
        waiting for the first skill call to create it lazily. The
        selector + github client stay lazy — provider construction can
        be expensive (Ollama HTTP client, Claude CLI probe) and is only
        worth paying for when a review actually runs.
        """
        # _ensure_usage_store is idempotent; tests that inject an
        # in-memory store skip the filesystem touch entirely.
        try:
            self._ensure_usage_store()
        except Exception as exc:
            logger.warning("reviewer usage store init failed at start(): %s", exc)
        await super().start()

    # -- skill surface -------------------------------------------------

    def register_skills(self) -> list[Skill]:
        return [
            Skill(
                "review_text",
                "Run a review over arbitrary content. Returns structured "
                "findings + usage record. The payload may be passed as "
                "`content` (canonical) OR `diff` (alias accepted for "
                "callers coming from the `review_diff` shape) — whichever "
                "is non-empty wins, with `content` taking precedence "
                "when both are supplied.",
                {
                    "kind": {"type": "string", "required": True},
                    # Canonical payload field. ``diff`` is also accepted
                    # as an alias so callers don't have to remember
                    # which shape this skill uses; the handler resolves
                    # whichever is non-empty (content wins on tie).
                    "content": {"type": "string", "default": ""},
                    "diff": {"type": "string", "default": ""},
                    "instructions": {"type": "string", "default": ""},
                    "context": {"type": "object", "default": {}},
                    "backend": {"type": "string", "default": ""},
                    "model": {"type": "string", "default": ""},
                    "request_id": {"type": "string", "default": ""},
                    "metadata": {"type": "object", "default": {}},
                    # severity_floor: drop findings below this severity
                    # from the returned result. "" = no skill-arg
                    # override (fall through to config then default
                    # "nit"). Valid values: "nit"|"comment"|"concern".
                    "severity_floor": {"type": "string", "default": ""},
                    # audience: distill-pipeline output-shape selector.
                    # "" / unset = "agent_consumption" (low aggression,
                    # the safe default). "audit_corpus" short-circuits
                    # the entire pipeline so audit / benchmark callers
                    # receive raw provider output. "github_comment" /
                    # "developer_handoff" / "human_review" reserved
                    # for future shaping transforms.
                    "audience": {"type": "string", "default": ""},
                    # num_ctx: per-call Ollama context-window override.
                    # 0 / unset = no skill-arg override; provider falls
                    # through to its config default and then the
                    # auto-bump heuristic. Positive int pins the
                    # provider's num_ctx for this call (useful for
                    # measurement runs that hold the axis constant).
                    # Other backends ignore this field.
                    "num_ctx": {"type": "integer", "default": 0},
                    # format: per-call Ollama structured-output toggle.
                    # "" / unset = no skill-arg override; provider
                    # falls through to its config default. "json" =
                    # enforce JSON-formatted output via Ollama's
                    # native ``format`` parameter (grammar-constrained
                    # decoding). Used by callers that route findings
                    # through small models which otherwise produce
                    # mixed output (per the 2026-04-22 evaluator-gate
                    # experiment). Other backends ignore this field.
                    "format": {"type": "string", "default": ""},
                    # consensus_runs: number of parallel provider
                    # invocations whose findings get consolidated by
                    # overlap. 1 (default) = no consensus; single
                    # provider call. >1 = run N times concurrently and
                    # keep findings appearing in >= consensus_min runs.
                    "consensus_runs": {"type": "integer", "default": 1},
                    # consensus_min: minimum run count a finding must
                    # appear in (under the same anchor key) to survive
                    # consolidation. Must be in [1, consensus_runs].
                    # 1 (default) = keep all findings (no real
                    # filtering), useful for warming up the consensus
                    # path; a strict majority is
                    # consensus_min = floor(runs/2)+1 — operators decide.
                    "consensus_min": {"type": "integer", "default": 1},
                    # evaluator_hot: optional second-pass filter over
                    # the consensus result's findings. "" (default)
                    # = no evaluator; the consensus result returns
                    # unchanged. "<backend>:<model>" = run the named
                    # provider over the candidate findings; only the
                    # findings the evaluator marks as real survive.
                    # Fail-open on evaluator error (keep all findings,
                    # log a warning). Escalation / cold tiers are a
                    # follow-up FR per the cb081fa8 split.
                    "evaluator_hot": {"type": "string", "default": ""},
                },
                since="0.1.0",
            ),
            Skill(
                "review_diff",
                "Shortcut for review_text with kind='pr_diff'. The "
                "payload may be passed as `diff` (canonical) OR "
                "`content` (alias accepted for callers coming from the "
                "`review_text` shape). The two skills differ in framing "
                "(diff bytes vs freeform text), not field name.",
                {
                    # Canonical payload field. ``content`` is also
                    # accepted as an alias.
                    "diff": {"type": "string", "default": ""},
                    "content": {"type": "string", "default": ""},
                    "instructions": {"type": "string", "default": ""},
                    "context": {"type": "object", "default": {}},
                    "backend": {"type": "string", "default": ""},
                    "model": {"type": "string", "default": ""},
                    "request_id": {"type": "string", "default": ""},
                    "metadata": {"type": "object", "default": {}},
                    "severity_floor": {"type": "string", "default": ""},
                    "audience": {"type": "string", "default": ""},
                    "num_ctx": {"type": "integer", "default": 0},
                    "format": {"type": "string", "default": ""},
                    "consensus_runs": {"type": "integer", "default": 1},
                    "consensus_min": {"type": "integer", "default": 1},
                    "evaluator_hot": {"type": "string", "default": ""},
                },
                since="0.1.0",
            ),
            Skill(
                "review_pr",
                "Fetch a GitHub PR diff + metadata, run review_text over it, "
                "and post the result as a GitHub PR review. Returns the "
                "ReviewResult augmented with the posted-review info (or a "
                "dry-run payload when dry_run=true).",
                {
                    "repo": {"type": "string", "required": True},
                    "pr_number": {"type": "integer", "required": True},
                    "instructions": {"type": "string", "default": ""},
                    "backend": {"type": "string", "default": ""},
                    "model": {"type": "string", "default": ""},
                    "dry_run": {"type": "boolean", "default": False},
                    "event": {"type": "string", "default": "COMMENT"},
                    "severity_floor": {"type": "string", "default": ""},
                },
                since="0.1.0",
            ),
            Skill(
                "sign_off_trailer",
                "Format an Agent-Reviewed-by trailer line from a "
                "ReviewResult. Two call shapes: (a) result-only — "
                "pass `result` (a ReviewResult dict) to format the "
                "trailer from a review the caller already has; "
                "(b) pass-through — pass review_text/review_diff "
                "args (kind/content/diff/...) to run a review "
                "internally and format the trailer in one call. "
                "Returns {verdict, trailer_line}; the caller "
                "stitches trailer_line into a commit message.",
                {
                    # Result-only path: the caller passes a
                    # serialized ReviewResult dict here. When set,
                    # the pass-through path is bypassed.
                    "result": {"type": "object", "default": {}},
                    # Pass-through path: same shape as review_text /
                    # review_diff. The handler runs review_text
                    # internally then formats the trailer from the
                    # resulting ReviewResult.
                    "kind": {"type": "string", "default": ""},
                    "content": {"type": "string", "default": ""},
                    "diff": {"type": "string", "default": ""},
                    "instructions": {"type": "string", "default": ""},
                    "context": {"type": "object", "default": {}},
                    "backend": {"type": "string", "default": ""},
                    "model": {"type": "string", "default": ""},
                    "request_id": {"type": "string", "default": ""},
                    "metadata": {"type": "object", "default": {}},
                    "severity_floor": {"type": "string", "default": ""},
                    # Optional formatting kwargs:
                    "role": {"type": "string", "default": "khonliang-reviewer"},
                    "reason": {"type": "string", "default": ""},
                },
                since="0.1.0",
            ),
            Skill(
                "usage_summary",
                "Aggregated token usage + estimated API cost grouped by "
                "(backend, model). All filters optional; omit to summarize "
                "the full history.",
                {
                    "backend": {"type": "string", "default": ""},
                    "model": {"type": "string", "default": ""},
                    "since": {"type": "number", "default": 0},
                    "until": {"type": "number", "default": 0},
                },
                since="0.1.0",
            ),
            Skill(
                "list_models",
                "Enumerate registered review backends + their declared "
                "models. Each entry includes a cheap availability hint "
                "(binary on PATH, auth file present, etc.) — does NOT "
                "exercise any model. Use this to discover the surface "
                "before picking a `backend` / `model` for review_text / "
                "review_diff / review_pr, or to drive a benchmark sweep.",
                {
                    # Optional filter: when set, return only the matching
                    # backend (or empty list when unknown). Empty string
                    # means "no filter" (return all).
                    "backend": {"type": "string", "default": ""},
                },
                since="0.1.0",
            ),
        ]

    @handler("review_text")
    async def handle_review_text(self, args: dict[str, Any]) -> dict[str, Any]:
        kind = str(args.get("kind") or "").strip()
        if not kind:
            return {"error": "kind is required"}

        # Accept ``content`` (canonical) OR ``diff`` (alias) — review_text
        # and review_diff differ in framing (freeform text vs unified diff
        # bytes), not field name. Subagents drift between the two; the
        # alias collapses the surface so either name works on either skill.
        # ``content`` wins when both are non-empty so the canonical name
        # remains authoritative.
        content = _resolve_payload_arg(args)
        if not content:
            return {"error": "content (or diff) is required and must be a non-empty string"}

        caller_backend = args.get("backend") or None
        # Preserve an explicit empty-string model: ``model=""`` means
        # "let the provider apply its own default", which is the
        # semantic ``ProviderSelector.select()`` honors when the
        # caller passes ``model is not None``. Coalescing ``""`` to
        # ``None`` here would silently route the caller back into the
        # default-resolution rules, defeating the explicit signal.
        # ``None`` (key absent / wrong type) still maps to ``None``;
        # only non-string values are filtered out.
        raw_model = args.get("model")
        caller_model = raw_model if isinstance(raw_model, str) else None
        context = _as_dict(args.get("context"))

        # Load ``.reviewer/config.yaml`` **once** per review. Both the
        # severity_floor resolver and the example_format resolver
        # consult the same file via the same git-show plumbing; loading
        # independently would double the ``git show`` + YAML-parse cost
        # on every review with ``repo_root``/``base_sha`` hints.
        # ``None`` when context lacks hints OR the config is unreachable —
        # the resolvers handle that case by falling through to defaults.
        repo_cfg = _load_repo_config_from_context(context)

        # Resolve severity_floor precedence up-front so a validation
        # error surfaces before we spend a provider call. Order matches
        # the FR (skill arg → .reviewer/config.yaml → built-in default).
        # The ``skill_arg`` step validates eagerly because the caller
        # typo'd; the config-layer step validates because operators
        # can typo their YAML too. Default is trusted (module constant).
        try:
            effective_floor = self._resolve_severity_floor(args, repo_cfg)
        except SeverityFloorError as exc:
            return {"error": str(exc)}

        try:
            effective_audience = _resolve_audience(args.get("audience"))
        except AudienceError as exc:
            return {"error": str(exc)}

        try:
            selector = self._ensure_selector()
            # ``is not None`` (not truthiness) so explicit ``model=""``
            # — meaning "let the provider apply its own default" —
            # takes the caller-override branch instead of falling
            # through to rule-table resolution.
            if caller_backend is not None or caller_model is not None:
                provider, chosen_model = selector.select(
                    backend=caller_backend, model=caller_model
                )
                selection_reason = "caller override"
            else:
                decision = decide(
                    _policy_input_for(kind=kind, content=content, context=context)
                )
                provider, chosen_model = selector.select(
                    backend=decision.backend, model=decision.model
                )
                selection_reason = f"rule-table: {decision.reason}"
        except UnknownBackendError as exc:
            return {"error": str(exc)}

        logger.debug(
            "reviewer.select: backend=%s model=%s reason=%s kind=%s",
            provider.name,
            chosen_model,
            selection_reason,
            kind,
        )

        # Strip any ``_khonliang_*`` keys from caller-supplied metadata
        # before merge — those are reserved for in-process passthrough
        # from the agent to the provider (see ``_RESERVED_METADATA_PREFIX``).
        # A caller cannot be allowed to inject a value for e.g.
        # ``_khonliang_repo_prompts``: providers forward it into
        # :func:`build_review_prompt` expecting a :class:`RepoPrompts`
        # instance the agent just loaded from a trusted base SHA.
        metadata = {
            **_strip_reserved_metadata(_as_dict(args.get("metadata"))),
            "model": chosen_model,
        }
        # ``num_ctx`` is plumbed through metadata so providers that
        # care about it (Ollama) can read it without changing the
        # ReviewRequest dataclass shape. Other backends ignore it.
        # 0 / unset = no skill-arg override; only forward positive
        # ints so the provider's resolution order treats the absent
        # case identically to "key not present".
        num_ctx_arg = args.get("num_ctx")
        if isinstance(num_ctx_arg, int) and not isinstance(num_ctx_arg, bool) and num_ctx_arg > 0:
            metadata["num_ctx"] = num_ctx_arg
        # ``format`` is plumbed the same way for Ollama's native
        # grammar-constrained decoding. Empty string = no skill-arg
        # override; any non-empty string is forwarded verbatim and
        # the provider's resolution order treats absence identically
        # to "key not present". Ollama validates the value server-side.
        format_arg = args.get("format")
        if isinstance(format_arg, str) and format_arg:
            metadata["format"] = format_arg

        # Repo-side prompt merge (FR fr_reviewer_92453047). Loaded here
        # rather than inside the provider so both Ollama and Claude-CLI
        # providers get the same merge behavior without re-implementing
        # the git-show plumbing. Graceful-absence: when context lacks
        # ``repo_root``/``base_sha``, both helpers return ``None`` and
        # providers fall back to the built-in prompt bytes.
        repo_prompts = _load_repo_prompts_from_context(context)
        # ``provider.name`` is the transport identifier (``ollama`` /
        # ``claude_cli``). ``.reviewer/models/<vendor>/`` is keyed on the
        # upstream model family (``ollama`` / ``anthropic``). Translate
        # so a repo's ``anthropic/_default.yaml: example_format: xml``
        # actually reaches Claude-backed reviews. Without this step the
        # resolver looks under ``claude_cli/`` (which will never exist)
        # and silently falls back to the markdown default.
        example_format = _resolve_example_format_from_config(
            repo_cfg,
            kind=kind,
            vendor=provider_to_vendor(provider.name),
            model=chosen_model,
        )
        if repo_prompts is not None:
            metadata[_METADATA_REPO_PROMPTS_KEY] = repo_prompts
        if example_format is not None:
            metadata[_METADATA_EXAMPLE_FORMAT_KEY] = example_format

        request = ReviewRequest(
            kind=kind,
            content=content,
            instructions=str(args.get("instructions") or ""),
            context=context,
            metadata=metadata,
            request_id=str(args.get("request_id") or _generate_request_id()),
        )

        # Consensus path (fr_reviewer_cb081fa8 first cut). When
        # ``consensus_runs > 1`` the same request goes through the
        # provider N times in parallel and the surviving findings are
        # those that appeared (under the same anchor key) in at least
        # ``consensus_min`` runs. Single-run (the default) skips the
        # whole orchestration and follows the pre-FR fast path.
        consensus_runs = _coerce_consensus_int(
            args.get("consensus_runs"), default=1
        )
        consensus_min = _coerce_consensus_int(
            args.get("consensus_min"), default=1
        )
        # Cross-validation runs unconditionally (after coercion) so a
        # caller passing ``consensus_runs=1, consensus_min=2`` fails
        # fast rather than silently taking the single-call fast path —
        # ``min > runs`` is unsatisfiable regardless of which branch
        # would have executed. (Copilot R1 PR#37.)
        try:
            _validate_consensus(consensus_runs, consensus_min)
        except ConsensusError as exc:
            return {"error": str(exc)}

        # Pre-validate ``evaluator_hot`` BEFORE the consensus / provider
        # call so an invalid spec or unknown backend doesn't waste
        # tokens on a review whose result is then thrown away.
        # (Copilot R1 PR#38.) The validation is a cheap string parse
        # plus a dict lookup; running it twice (here and inside
        # ``_run_evaluator_hot``) is negligible defense-in-depth.
        evaluator_hot_arg = args.get("evaluator_hot")
        evaluator_hot_active = bool(
            isinstance(evaluator_hot_arg, str) and evaluator_hot_arg
        )
        if evaluator_hot_active:
            try:
                eval_backend, _ = _parse_evaluator_spec(evaluator_hot_arg)
                if eval_backend not in self._ensure_selector().providers:
                    raise EvaluatorError(
                        f"evaluator_hot backend={eval_backend!r} is not "
                        f"registered; known backends: "
                        f"{sorted(self._ensure_selector().providers)}"
                    )
            except EvaluatorError as exc:
                return {"error": str(exc)}

        if consensus_runs > 1:
            result = await self._run_consensus(
                provider, request, consensus_runs, consensus_min
            )
        else:
            result = await provider.review(request)

        # Evaluator-hot pass (fr_reviewer_cb081fa8 second cut). Runs
        # AFTER consensus consolidation but BEFORE severity_floor —
        # the evaluator sees pre-floor findings so it can drop nits
        # the floor would have kept, and the floor filters the
        # evaluator-survived set so dual-filtering doesn't surprise
        # operators. The spec is already validated upfront so we
        # don't expect EvaluatorError here, but catch defensively in
        # case future code adds a runtime-only failure mode.
        if evaluator_hot_active:
            try:
                result = await self._run_evaluator_hot(
                    result, request, evaluator_hot_arg
                )
            except EvaluatorError as exc:
                # Defense in depth: still run the distill pipeline +
                # record consensus usage before surfacing the error
                # so the spend isn't lost.
                result = run_pipeline(
                    result,
                    DistillConfig(
                        severity_floor=effective_floor,
                        audience=effective_audience,
                    ),
                )
                await self._record_usage(result)
                return {"error": str(exc)}

        # Run the distill pipeline at the edge of the return path
        # (fr_reviewer_de1694a8). The pipeline owns severity_floor
        # filtering + summary stripping + filtered_count bumping;
        # the ``audit_corpus`` audience short-circuits the whole
        # pipeline so audit / benchmark callers always see raw
        # provider output.
        result = run_pipeline(
            result,
            DistillConfig(
                severity_floor=effective_floor,
                audience=effective_audience,
            ),
        )
        await self._record_usage(result)
        return result.to_dict()

    async def _run_consensus(
        self,
        provider: Any,
        request: ReviewRequest,
        runs: int,
        min_count: int,
    ) -> ReviewResult:
        """Run ``provider.review`` ``runs`` times in parallel, consolidate
        findings by overlap.

        First-cut behavior (the simple correct version):

        - Per-run requests carry a ``-r{i+1}`` suffix on
          ``request_id`` for the provider calls; the returned result
          always uses the base request id (per-run suffixes are
          internal plumbing only — see the merged-usage call below
          which rewrites ``request_id`` back to the base).
        - If ANY run errors, skip consolidation and return a new
          ``ReviewResult`` carrying the errored run's summary,
          disposition, error metadata, backend, and model, with
          ``findings=[]`` and usage merged across all completed runs
          (with disposition / error fields overridden from the
          failing run so failure analytics stay accurate). This is
          intentionally simple — partial-consensus over a degraded
          set is a future refinement.
        - Surviving findings preserve the canonical body from the
          first run that produced them. No body merging — averaging
          text would smooth features (see
          ``project_reviewer_distill_principle``).
        - Usage is recorded as a single merged event on the base
          ``request_id``; token counts sum across runs (true compute
          spend) and ``duration_ms`` is the max (concurrent
          wall-clock).
        """
        per_run_requests = [
            dataclass_replace(request, request_id=f"{request.request_id}-r{i + 1}")
            for i in range(runs)
        ]
        results = await asyncio.gather(
            *(provider.review(r) for r in per_run_requests)
        )
        first_errored = next((r for r in results if r.error), None)
        if first_errored is not None:
            # Surface the first error verbatim, but merge usage across
            # ALL runs so the cost-accounting reflects the real spend
            # — runs that completed successfully before the error
            # already burned tokens. (Copilot R1 PR#37: previously
            # we returned the errored result alone and dropped the
            # other runs' usage events.)
            #
            # Then override the merged usage's disposition / error
            # fields from the errored run so the persisted usage row
            # reflects the FAILURE outcome — without this override
            # the merged usage would silently inherit the first
            # successful run's disposition ("posted", empty error)
            # even though the returned ReviewResult is errored, and
            # failure-rate analytics would undercount.
            # (Copilot R2 PR#37.)
            usage_events = [r.usage for r in results if r.usage is not None]
            merged_usage = (
                _merge_usage_events(usage_events, request.request_id)
                if usage_events
                else None
            )
            if merged_usage is not None and first_errored.usage is not None:
                merged_usage.disposition = first_errored.usage.disposition
                merged_usage.error = first_errored.usage.error
                merged_usage.error_category = first_errored.usage.error_category
            elif merged_usage is not None:
                # Errored run had no usage event (some providers
                # return None on hard transport failure). Fall back
                # to the result-level error fields so the usage row
                # still reflects failure.
                merged_usage.disposition = first_errored.disposition
                merged_usage.error = first_errored.error
                merged_usage.error_category = first_errored.error_category
            return ReviewResult(
                request_id=request.request_id,
                summary=first_errored.summary,
                findings=[],
                disposition=first_errored.disposition,
                error=first_errored.error,
                error_category=first_errored.error_category,
                usage=merged_usage,
                backend=first_errored.backend,
                model=first_errored.model,
            )
        return _consolidate_consensus_results(
            results,
            min_count=min_count,
            base_request_id=request.request_id,
        )

    async def _run_evaluator_hot(
        self,
        consensus_result: ReviewResult,
        original_request: ReviewRequest,
        evaluator_spec: str,
    ) -> ReviewResult:
        """Filter ``consensus_result.findings`` through a second-pass evaluator.

        First-cut behavior of the ``evaluator_hot`` tier from
        ``fr_reviewer_cb081fa8``:

        - Parse ``evaluator_spec`` (``"<backend>:<model>"``); look up
          the backend in the agent's selector. Mismatch raises
          :class:`EvaluatorError`.
        - Build a new :class:`ReviewRequest` carrying the original
          diff as ``content`` and the candidate findings (JSON-dumped)
          embedded in ``instructions`` so the evaluator can reason
          about them in the same prompt-template the rest of the
          reviewer uses. No special evaluator schema — the evaluator
          returns a regular :class:`ReviewResult` whose ``findings``
          list is the survivors.
        - **Fail-open** on evaluator error / parse failure: if the
          evaluator returns an errored result, log a warning and
          return ``consensus_result`` unchanged. Conservative default —
          better to over-keep findings than have a flaky evaluator
          silently nuke real concerns. Escalation tier (which would
          fire on this path) is a follow-up FR.
        - Skip the call entirely when ``consensus_result`` is errored
          or carries zero findings: there's nothing to filter.
        - Usage merges across consensus + evaluator so cost-accounting
          captures total spend.
        """
        if consensus_result.error or not consensus_result.findings:
            return consensus_result

        backend, model = _parse_evaluator_spec(evaluator_spec)
        selector = self._ensure_selector()
        if backend not in selector.providers:
            raise EvaluatorError(
                f"evaluator_hot backend={backend!r} is not registered; "
                f"known backends: {sorted(selector.providers)}"
            )
        evaluator_provider = selector.providers[backend]

        # Compact JSON (no indent / minimal separators) — the
        # evaluator reads this as opaque data, not for humans.
        # Indent=2 was costing ~30% extra tokens per candidate
        # finding for zero benefit. (Copilot R4 PR#38.)
        findings_json = json.dumps(
            [f.to_dict() for f in consensus_result.findings],
            separators=(",", ":"),
        )
        evaluator_instructions = (
            "You are evaluating findings from a previous reviewer pass on the "
            "diff below. For each candidate finding, decide whether it "
            "represents a real, actionable concern, or a false positive "
            "(over-eager noise, restatement of the diff, mislabeled "
            "severity).\n\n"
            "Return ONLY the findings you judge as real, preserving each "
            "kept finding's fields (severity, title, body, path, line, "
            "category, suggestion) verbatim. Return an empty findings list "
            "if none are real.\n\n"
            "Candidate findings to evaluate:\n"
            f"{findings_json}"
        )
        if original_request.instructions:
            evaluator_instructions = (
                f"{original_request.instructions}\n\n{evaluator_instructions}"
            )

        eval_metadata = {
            **{
                k: v
                for k, v in original_request.metadata.items()
                if k != "model"
            },
            "model": model,
        }
        eval_request = ReviewRequest(
            kind=original_request.kind,
            content=original_request.content,
            instructions=evaluator_instructions,
            context=original_request.context,
            metadata=eval_metadata,
            request_id=f"{original_request.request_id}-eval",
        )

        eval_result = await evaluator_provider.review(eval_request)

        if eval_result.error:
            logger.warning(
                "reviewer.evaluator_hot: evaluator %s failed (%s); "
                "fail-open — keeping all %d consensus findings",
                evaluator_spec,
                eval_result.error,
                len(consensus_result.findings),
            )
            # The evaluator call already spent tokens before
            # erroring (parse failure on a real LLM response, not
            # a free no-op). Merge usage when identities match so
            # the persisted usage row reflects total compute spend;
            # mismatched identities log the lost spend so operators
            # can see the gap. (Copilot R4 PR#38.)
            consensus_usage = consensus_result.usage
            evaluator_usage = eval_result.usage
            if consensus_usage is not None and evaluator_usage is not None:
                if (
                    consensus_usage.backend == evaluator_usage.backend
                    and consensus_usage.model == evaluator_usage.model
                ):
                    merged_usage = _merge_usage_events(
                        [consensus_usage, evaluator_usage],
                        consensus_result.request_id,
                    )
                    merged_usage = dataclass_replace(
                        merged_usage,
                        duration_ms=(
                            consensus_usage.duration_ms
                            + evaluator_usage.duration_ms
                        ),
                    )
                    return dataclass_replace(consensus_result, usage=merged_usage)
                logger.warning(
                    "reviewer.evaluator_hot: not merging errored evaluator "
                    "usage into consensus usage because identity differs "
                    "(consensus=%s/%s, evaluator=%s/%s); evaluator spend "
                    "of input=%d output=%d not persisted to usage_summary",
                    consensus_usage.backend,
                    consensus_usage.model,
                    evaluator_usage.backend,
                    evaluator_usage.model,
                    evaluator_usage.input_tokens,
                    evaluator_usage.output_tokens,
                )
            return consensus_result

        # Evaluator succeeded — but DON'T trust eval_result.findings
        # blindly. Per Copilot R2 PR#38: the evaluator could hallucinate
        # a new finding (or rephrase a candidate enough that the
        # rewritten copy bypasses the consensus gate). Survivors must
        # be a strict subset of the candidate set, anchored by the
        # same ``_consensus_finding_key`` consensus uses for grouping.
        #
        # We also return the CANDIDATE finding objects verbatim
        # (rather than the evaluator's possibly-rewritten copies) so
        # the prompt's "preserve fields verbatim" instruction is
        # enforced structurally, not just by the model's compliance.
        candidate_by_key: dict[tuple[str, str, int, str], ReviewFinding] = {
            _consensus_finding_key(f): f for f in consensus_result.findings
        }
        survivors: list[ReviewFinding] = []
        seen_keys: set[tuple[str, str, int, str]] = set()
        hallucinated_titles: list[str] = []
        for ev_finding in eval_result.findings:
            key = _consensus_finding_key(ev_finding)
            if key in seen_keys:
                # Evaluator emitted the same key twice; the canonical
                # candidate is already in survivors. Drop silently —
                # idempotent under repeated emit.
                continue
            candidate = candidate_by_key.get(key)
            if candidate is not None:
                survivors.append(candidate)
                seen_keys.add(key)
            else:
                # Evaluator returned a finding whose anchor doesn't
                # match any candidate. Either a hallucination or
                # field-rewrite that drifted past the normalized
                # title threshold. Drop + log so operators can spot
                # rogue evaluator behavior.
                hallucinated_titles.append(ev_finding.title)

        if hallucinated_titles:
            logger.warning(
                "reviewer.evaluator_hot: dropped %d evaluator finding(s) "
                "not in candidate set (likely hallucination or "
                "field-rewrite): %s",
                len(hallucinated_titles),
                hallucinated_titles[:5],
            )

        # Usage merge is identity-aware (Copilot R1 PR#38): only merge
        # when consensus and evaluator share the same (backend, model);
        # otherwise the merged event would misattribute evaluator spend
        # to the consensus backend in usage_summary analytics. When
        # identities differ, the consensus usage is kept as the
        # canonical record and the evaluator spend is logged as a
        # warning so operators can see the lost-attribution event;
        # capturing both stages cleanly needs a usage-list refactor
        # which is a follow-up FR.
        consensus_usage = consensus_result.usage
        evaluator_usage = eval_result.usage
        if consensus_usage is not None and evaluator_usage is not None:
            if (
                consensus_usage.backend == evaluator_usage.backend
                and consensus_usage.model == evaluator_usage.model
            ):
                merged_usage = _merge_usage_events(
                    [consensus_usage, evaluator_usage],
                    consensus_result.request_id,
                )
                # Override duration_ms specifically for this call site:
                # _merge_usage_events takes max(durations) because
                # consensus runs are PARALLEL (concurrent wall-clock),
                # but consensus + evaluator are SEQUENTIAL stages —
                # the wall-clock is the sum, not the max. (Copilot R2
                # PR#38.)
                merged_usage = dataclass_replace(
                    merged_usage,
                    duration_ms=(
                        consensus_usage.duration_ms
                        + evaluator_usage.duration_ms
                    ),
                )
            else:
                logger.warning(
                    "reviewer.evaluator_hot: not merging evaluator usage into "
                    "consensus usage because identity differs "
                    "(consensus=%s/%s, evaluator=%s/%s); evaluator spend "
                    "of input=%d output=%d not persisted to usage_summary",
                    consensus_usage.backend,
                    consensus_usage.model,
                    evaluator_usage.backend,
                    evaluator_usage.model,
                    evaluator_usage.input_tokens,
                    evaluator_usage.output_tokens,
                )
                merged_usage = consensus_usage
        else:
            merged_usage = consensus_usage or evaluator_usage

        return ReviewResult(
            request_id=consensus_result.request_id,
            # Prefer the evaluator's summary when non-empty: the
            # evaluator just filtered the findings list, so its
            # summary describes the post-filter state. Falling back
            # to the consensus summary keeps behavior unchanged for
            # evaluators that don't write a summary. (Copilot R1
            # PR#38: previously the consensus summary stuck around
            # even when it referenced findings the evaluator just
            # dropped.)
            summary=eval_result.summary or consensus_result.summary,
            findings=survivors,
            disposition=consensus_result.disposition,
            error="",
            error_category="",
            usage=merged_usage,
            backend=consensus_result.backend,
            model=consensus_result.model,
        )

    def _resolve_severity_floor(
        self, args: dict[str, Any], cfg: RepoConfig | None
    ) -> str:
        """Resolve the effective severity_floor per the FR precedence chain.

        High-to-low:

        1. Skill-arg ``severity_floor`` (non-empty string) — **strict**.
           A bad value here is a caller bug; raise
           :class:`SeverityFloorError` so the review fails fast with a
           message the caller can act on.
        2. ``review.severity_floor`` / ``checks.severity_floor`` in
           ``.reviewer/config.yaml`` — **lenient**. Only consulted when
           the caller passes a pre-loaded :class:`RepoConfig` (obtained
           via :func:`_load_repo_config_from_context`). A bad value in
           YAML shouldn't nuke every review for that repo — log a
           warning naming the offending value and fall through to the
           built-in default. Reviewing is more important than
           config-layer correctness.
        3. :data:`_DEFAULT_SEVERITY_FLOOR` (``"nit"`` — no filtering).

        Rationale for asymmetric validation: the skill-arg path is a
        programmatic caller (another agent, a test, an orchestrator) —
        strict failure is the correct feedback channel. The config-layer
        path is a human-edited YAML file on a repo; a typo there
        shouldn't silently wedge CI.

        Note: ``cfg`` is threaded in (rather than re-loaded here) so that
        ``handle_review_text`` loads the config exactly once per review
        and shares it with :func:`_resolve_example_format_from_config`.
        """
        arg_value = args.get("severity_floor")
        if isinstance(arg_value, str) and arg_value:
            return _validate_severity_floor(arg_value, source="skill arg")

        config_value = _resolve_repo_severity_floor(cfg)
        if isinstance(config_value, str) and config_value:
            try:
                return _validate_severity_floor(
                    config_value, source=".reviewer/config.yaml"
                )
            except SeverityFloorError as exc:
                # RepoConfig.severity_floor reads review.severity_floor
                # first, falling back to checks.severity_floor — the
                # warning names the resolved key generically so operators
                # aren't misled about which key actually carried the bad
                # value.
                logger.warning(
                    "reviewer: ignoring invalid .reviewer/config.yaml "
                    "severity_floor=%r (checked review.severity_floor "
                    "and checks.severity_floor); falling back to "
                    "default %r (%s)",
                    config_value,
                    _DEFAULT_SEVERITY_FLOOR,
                    exc,
                )
                return _DEFAULT_SEVERITY_FLOOR

        return _DEFAULT_SEVERITY_FLOOR

    @handler("review_diff")
    async def handle_review_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        # Accept ``diff`` (canonical) OR ``content`` (alias). For
        # review_diff the resolution prefers ``diff`` since that's the
        # canonical-for-this-skill name; ``content`` is the legacy /
        # cross-skill alias.
        diff = _resolve_payload_arg(args, prefer="diff")
        if not diff:
            return {"error": "diff (or content) is required and must be a non-empty string"}
        forwarded = {
            k: v for k, v in args.items() if k not in {"diff", "kind", "content"}
        }
        forwarded["kind"] = "pr_diff"
        forwarded["content"] = diff
        return await self.handle_review_text(forwarded)

    @handler("review_pr")
    async def handle_review_pr(self, args: dict[str, Any]) -> dict[str, Any]:
        """End-to-end: fetch PR, review via review_text, post back to GitHub.

        ``event`` defaults to ``"COMMENT"`` — the only event the reviewer
        agent is allowed to use autonomously. ``APPROVE`` is rejected at
        the validation step (approval authority stays human per FR).
        ``REQUEST_CHANGES`` / ``PENDING`` are still accepted for operator
        tooling + human-in-the-loop scenarios.
        """
        repo = str(args.get("repo") or "").strip()
        if not repo:
            return {"error": "repo is required (owner/name form)"}
        pr_raw = args.get("pr_number")
        # Reject bool explicitly — bool subclasses int, so `int(True)` silently
        # becomes PR #1 and `int(False)` becomes 0. Both are wrong and need to
        # surface as errors rather than targeting the wrong PR.
        if isinstance(pr_raw, bool):
            return {"error": "pr_number must be an integer, not a boolean"}
        try:
            pr_number = int(pr_raw)
        except (TypeError, ValueError):
            return {"error": "pr_number is required and must be an integer"}
        if pr_number <= 0:
            return {"error": "pr_number must be positive"}

        dry_run_raw = args.get("dry_run", False)
        # Strict bool: `bool(val)` would accept any truthy value including the
        # string "false", which would unexpectedly skip posting. Force a real
        # boolean so operator typos / YAML-to-JSON mishaps fail loudly.
        if not isinstance(dry_run_raw, bool):
            return {
                "error": (
                    f"dry_run must be a boolean, got {type(dry_run_raw).__name__}"
                )
            }
        dry_run = dry_run_raw
        event_raw = str(args.get("event") or "COMMENT").strip().upper()
        if event_raw not in _VALID_REVIEW_EVENTS:
            return {
                "error": (
                    "event must be one of "
                    f"{sorted(_VALID_REVIEW_EVENTS)}; got {event_raw!r}"
                )
            }
        event = event_raw

        github = self._ensure_github_client()
        try:
            # Fetches are independent; run them concurrently to halve
            # end-to-end latency on high-latency GitHub API links.
            metadata, diff = await asyncio.gather(
                github.get_pr_metadata(repo, pr_number),
                github.get_pr_diff(repo, pr_number),
            )
        except GithubClientError as exc:
            return {"error": f"github fetch failed: {exc}"}

        # Feed the diff into review_text via the shared path so selector,
        # rule table, severity_floor, and usage recording all run exactly
        # once per review, regardless of which entry skill was called.
        review_args: dict[str, Any] = {
            "kind": "pr_diff",
            "content": diff,
            "instructions": str(args.get("instructions") or ""),
            "context": {
                "pr": metadata.to_dict(),
            },
            "backend": args.get("backend") or "",
            # Forward ``model`` verbatim so handle_review_text's empty-
            # string-vs-None distinction reaches the selector. ``None``
            # (caller omitted) and ``""`` (caller-explicit "use
            # provider default") have different meanings; ``or ""``
            # would conflate them.
            "model": args.get("model"),
            "metadata": {"repo": repo, "pr_number": pr_number},
            # review_pr fetches via API (no local clone), so the
            # ``.reviewer/config.yaml`` layer can't activate here —
            # callers who want a config-layer floor should call
            # ``review_text`` directly with repo_root/base_sha in
            # context. Pass through the skill-arg floor only.
            "severity_floor": args.get("severity_floor") or "",
        }
        review_result = await self.handle_review_text(review_args)
        # ReviewResult.to_dict() always carries an ``error`` key (empty
        # string when the review succeeded). Early-return only on a
        # truthy error message.
        if review_result.get("error"):
            return review_result

        posted_body, posted_comments = _format_for_github(review_result)

        if dry_run:
            return {
                **review_result,
                "pr": metadata.to_dict(),
                "github": {
                    "dry_run": True,
                    "body": posted_body,
                    "comments": posted_comments,
                    "event": event,
                },
            }

        try:
            submitted = await github.submit_review(
                repo,
                pr_number,
                body=posted_body,
                comments=posted_comments,
                event=event,
                commit_sha=metadata.head_sha or None,
            )
        except GithubClientError as exc:
            return {
                **review_result,
                "pr": metadata.to_dict(),
                "error": f"github post failed: {exc}",
            }

        return {
            **review_result,
            "pr": metadata.to_dict(),
            "github": {
                "dry_run": False,
                "review": submitted.to_dict(),
                "inline_comments_posted": len(posted_comments),
                "event": event,
            },
        }

    @handler("sign_off_trailer")
    async def handle_sign_off_trailer(
        self, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Format an ``Agent-Reviewed-by`` trailer line.

        Two shapes:

        - **Result-only**: ``args["result"]`` is a ReviewResult dict
          (typically the output of a previous ``review_text`` /
          ``review_diff`` / ``review_pr`` call). The handler skips
          the review and formats the trailer directly. This is the
          common case — pre-push subagents that already ran the
          review and just need the canonical trailer string.

        - **Pass-through**: when ``result`` is absent or empty,
          forward the remaining args to ``review_text`` (with the
          same ``content`` / ``diff`` alias the consistency FR
          accepts) and format the trailer from the resulting
          ReviewResult. Saves the caller a round-trip.

        Returns ``{"verdict": str, "trailer_line": str}``. Errors
        from either path surface as the standard ``{"error": ...}``
        envelope.
        """
        from reviewer.skills.sign_off_trailer import build_trailer

        role = str(args.get("role") or "khonliang-reviewer")
        reason = str(args.get("reason") or "")

        # Result-only path: caller already ran the review.
        result_dict = args.get("result")
        if isinstance(result_dict, dict) and result_dict:
            try:
                result = ReviewResult.from_dict(result_dict)
            except (TypeError, ValueError, KeyError) as exc:
                return {"error": f"sign_off_trailer: malformed result: {exc}"}
            try:
                return build_trailer(result, role=role, reason=reason)
            except (ValueError, TypeError, AttributeError) as exc:
                # ValueError: documented contract (e.g. errored
                # disposition; build_trailer raises so the caller
                # doesn't ship a sign-off for a review that didn't
                # run).
                # TypeError / AttributeError: defense in depth —
                # build_trailer + the helpers coerce inputs, but a
                # weird-shape payload at the bus boundary could
                # still trip something. Surface the standard error
                # envelope rather than letting the handler crash.
                return {"error": f"sign_off_trailer: {exc}"}

        # Pass-through path: run review_text first, then format.
        # Strip the trailer-only fields so the review-call arg
        # envelope only carries what handle_review_text expects.
        review_args = {
            k: v for k, v in args.items() if k not in {"result", "role", "reason"}
        }
        # Default ``kind`` to ``pr_diff`` when the caller didn't
        # supply one. The canonical pre-push sign-off use case is
        # trailer formatting against a diff, so this default keeps
        # the common shape minimal — a freeform-text caller who
        # wants ``kind="spec"`` (etc.) still passes it explicitly.
        if not review_args.get("kind"):
            review_args["kind"] = "pr_diff"
        review_outcome = await self.handle_review_text(review_args)
        # ReviewResult.to_dict() always includes an ``error`` field
        # (default empty); a truthy check distinguishes the actual
        # error envelope (``{"error": "..."}``) from a normal review
        # result that happens to carry the field empty. If the
        # review surfaces a real error, the outcome lacks the rest
        # of the ReviewResult shape — forward verbatim so the caller
        # sees the same envelope review_text would have returned.
        if review_outcome.get("error") and "request_id" not in review_outcome:
            return review_outcome
        try:
            result = ReviewResult.from_dict(review_outcome)
        except (TypeError, ValueError, KeyError) as exc:
            return {"error": f"sign_off_trailer: malformed review result: {exc}"}
        try:
            return build_trailer(result, role=role, reason=reason)
        except (ValueError, TypeError, AttributeError) as exc:
            # ValueError: documented contract (errored disposition).
            # TypeError / AttributeError: defense in depth against
            # a weird-shape payload at the bus boundary.
            # Don't ship a trailer for a failed review — surface
            # the error envelope so the caller sees the failure
            # rather than a misleading ``approved`` sign-off built
            # from zero findings.
            return {"error": f"sign_off_trailer: {exc}"}

    @handler("usage_summary")
    async def handle_usage_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return token + cost aggregates grouped by (backend, model).

        Storage failures (DB path unreadable, permission issue, bad
        YAML on seed) surface as a structured ``{"error": "..."}``
        response rather than raising — keeps the skill surface from
        crashing out when the store is broken.
        """
        try:
            store = self._ensure_usage_store()
            summaries = store.summarize(
                backend=(args.get("backend") or None),
                model=(args.get("model") or None),
                since=_positive_float_or_none(args.get("since")),
                until=_positive_float_or_none(args.get("until")),
            )
        except Exception as exc:
            logger.warning("usage_summary failed: %s", exc)
            return {
                "error": f"usage_summary failed: {exc}",
                "entries": [],
                "total_rows": 0,
                "total_cost_usd": 0.0,
            }
        return {
            "entries": [s.to_dict() for s in summaries],
            "total_rows": sum(s.rows for s in summaries),
            "total_cost_usd": sum(s.total_cost_usd for s in summaries),
        }

    @handler("list_models")
    async def handle_list_models(self, args: dict[str, Any]) -> dict[str, Any]:
        """Enumerate registered backends + declared models.

        Optional ``backend`` filter narrows the response to a single
        backend (or empty when unknown). The cheap availability probe
        in :class:`ProviderRegistry` reports whether the binary / auth
        is in place but does NOT exercise the model — that's what
        :meth:`ReviewProvider.healthcheck` is for.
        """
        backend_filter = args.get("backend")
        if isinstance(backend_filter, str) and backend_filter:
            filt: str | None = backend_filter
        else:
            filt = None
        try:
            registrations = self._ensure_registry().list(backend=filt)
        except Exception as exc:
            logger.warning("list_models failed: %s", exc)
            return {
                "error": f"list_models failed: {exc}",
                "providers": [],
            }
        return {
            "providers": [r.to_dict() for r in registrations],
        }

    # -- internals -----------------------------------------------------

    async def _record_usage(self, result: ReviewResult) -> None:
        """Persist the usage event + emit ``reviewer.usage`` on the bus.

        Called from the review handlers after provider.review() returns.
        Zero-cost events (Ollama-backed) get their ``estimated_api_cost_usd``
        back-filled from the ``model_pricing`` table before storage.

        Every failure path — store open/create, cost back-fill, write,
        publish — is wrapped so accounting problems never propagate
        into the caller's review. That's the whole point of this
        method being separate from the review flow itself.
        """
        if result.usage is None:
            return
        try:
            store = self._ensure_usage_store()
        except Exception as exc:
            logger.warning("reviewer usage store unavailable: %s", exc)
            return

        try:
            filled = store.back_fill_cost(result.usage)
        except Exception as exc:
            logger.warning("reviewer usage back_fill_cost failed: %s", exc)
            filled = result.usage
        # Mutate the result so the caller sees the back-filled cost too.
        result.usage = filled

        try:
            store.write_usage(filled)
        except Exception as exc:
            logger.warning("reviewer_usage write failed: %s", exc)

        try:
            await self.publish("reviewer.usage", filled.to_dict())
        except RuntimeError as exc:
            # Agent not connected (tests, dry-run) — expected in those paths.
            logger.debug("skipping reviewer.usage publish: %s", exc)
        except Exception as exc:
            logger.warning("reviewer.usage publish failed: %s", exc)

    def _ensure_selector(self) -> ProviderSelector:
        if self._injected_selector is not None:
            return self._injected_selector
        if self._cached_selector is None:
            self._cached_selector = self._build_default_selector()
        return self._cached_selector

    def _ensure_github_client(self) -> ReviewerGithubClient:
        if self._injected_github is not None:
            return self._injected_github
        if self._cached_github is None:
            self._cached_github = ReviewerGithubClient()
        return self._cached_github

    def _ensure_usage_store(self) -> UsageStore:
        if self._injected_store is not None:
            return self._injected_store
        if self._cached_store is None:
            config = self._load_config()
            db_path = str(config.get("db_path") or "data/reviewer.db")
            store = open_usage_store(db_path)
            try:
                seeded = store.seed_pricing_if_empty(load_default_pricing())
                if seeded:
                    logger.info("seeded %d default pricing rows", seeded)
            except Exception as exc:
                # Seeding is best-effort — missing YAML or parse error
                # shouldn't block agent startup.
                logger.warning("default pricing seed failed: %s", exc)
            self._cached_store = store
        return self._cached_store

    def _ensure_registry(self) -> ProviderRegistry:
        if self._injected_registry is not None:
            return self._injected_registry
        if self._injected_selector is not None:
            # Test harnesses inject only a selector when they don't
            # care about list_models metadata. Derive a minimal
            # registry from the selector's provider map so
            # ``list_models`` enumerates the SAME backends the
            # selector resolves to. Without this, the agent would
            # silently call ``_build_default_registry`` and
            # instantiate real providers (claude/codex/ollama) under
            # the hood — defeating the harness's isolation contract.
            # The derived registry has no declared_models / default
            # metadata; ``list_models`` callers see backends + a
            # default-empty list, which is the truthful answer for
            # an injection that didn't supply that information.
            if self._cached_registry is None:
                derived = ProviderRegistry()
                for provider in self._injected_selector.providers.values():
                    derived.register(provider)
                self._cached_registry = derived
            return self._cached_registry
        if self._cached_registry is None:
            self._cached_registry = self._build_default_registry()
        return self._cached_registry

    def _build_default_registry(self) -> ProviderRegistry:
        """Construct the canonical ProviderRegistry from agent config.

        Reads ``providers.<backend>`` blocks from the agent config and
        ``default_pricing.yaml`` rows for declared-model lists, then
        registers ClaudeCli / CodexCli / Ollama providers under their
        canonical backend names. Registration order is insertion
        order; ``list_models`` returns providers in that order.
        """
        config = self._load_config()
        providers_cfg = _as_dict(config.get("providers"))
        claude_cfg = _as_dict(providers_cfg.get("claude_cli"))
        codex_cfg = _as_dict(providers_cfg.get("codex_cli"))
        copilot_cfg = _as_dict(providers_cfg.get("gh_copilot"))
        ollama_cfg = _as_dict(providers_cfg.get("ollama"))

        # Per-backend declared-models comes from the pricing YAML;
        # operators add a row per (backend, model) they care about,
        # and list_models surfaces them. Falling through to an empty
        # list is fine — the registry will just show the per-provider
        # default in that case.
        try:
            pricing_rows = load_default_pricing()
        except Exception as exc:
            logger.warning("default pricing seed failed: %s", exc)
            pricing_rows = []
        declared_by_backend: dict[str, list[str]] = {}
        for row in pricing_rows:
            declared_by_backend.setdefault(row.backend, []).append(row.model)

        registry = ProviderRegistry()
        claude_default = str(claude_cfg.get("default_model") or "")
        registry.register(
            ClaudeCliProvider(
                ClaudeCliProviderConfig(
                    binary=str(claude_cfg.get("binary") or "claude"),
                    default_model=claude_default,
                )
            ),
            default_model=claude_default,
            declared_models=declared_by_backend.get("claude_cli", []),
        )
        registry.register(
            CodexCliProvider(
                CodexCliProviderConfig(
                    binary=str(codex_cfg.get("binary") or "codex"),
                    default_model=str(codex_cfg.get("default_model") or ""),
                )
            ),
            default_model=str(codex_cfg.get("default_model") or ""),
            declared_models=declared_by_backend.get("codex_cli", []),
        )
        copilot_default = str(copilot_cfg.get("default_model") or "")
        copilot_effort = str(copilot_cfg.get("reasoning_effort") or "")
        registry.register(
            GhCopilotProvider(
                GhCopilotProviderConfig(
                    binary=str(copilot_cfg.get("binary") or "copilot"),
                    default_model=copilot_default,
                    reasoning_effort=copilot_effort,
                )
            ),
            default_model=copilot_default,
            declared_models=declared_by_backend.get("gh_copilot", []),
        )
        # Source Ollama's provider-default model from
        # ``providers.ollama.default_model`` (per-provider config),
        # falling back to the built-in qwen baseline. Decoupled from
        # the global ``config.default_model`` so an operator who sets
        # ``default_provider: claude_cli`` and ``default_model:
        # claude-opus-4-7`` doesn't accidentally inject a Claude model
        # id into Ollama when a caller picks ``backend: ollama``
        # without specifying a model. The selector deliberately
        # returns ``""`` for non-default-backend selections (see
        # ``ProviderSelector.select``); each provider then applies its
        # own config-level default.
        ollama_default = str(
            ollama_cfg.get("default_model") or "qwen2.5-coder:14b"
        )
        # Thread per-provider knobs through to the dataclass so the
        # config-layer rung of the resolution order ("caller →
        # config → None" / "caller → config → auto-bump") is
        # actually reachable from ``config.yaml``. Without this,
        # the advertised 3-/4-layer resolution skips the operator
        # default and goes straight from caller to None / auto-bump.
        # Treat-malformed-as-absent: a YAML payload with a non-string
        # ``format`` or non-positive ``num_ctx`` collapses to None
        # rather than crashing the provider boot path; the value
        # types are validated again inside the resolution helpers,
        # so this is belt-and-suspenders.
        ollama_format_raw = ollama_cfg.get("format")
        ollama_format = (
            ollama_format_raw
            if isinstance(ollama_format_raw, str) and ollama_format_raw
            else None
        )
        ollama_num_ctx_raw = ollama_cfg.get("num_ctx")
        ollama_num_ctx = (
            ollama_num_ctx_raw
            if isinstance(ollama_num_ctx_raw, int)
            and not isinstance(ollama_num_ctx_raw, bool)
            and ollama_num_ctx_raw > 0
            else None
        )
        registry.register(
            OllamaProvider(
                OllamaProviderConfig(
                    base_url=str(
                        ollama_cfg.get("base_url")
                        or "http://localhost:11434/v1"
                    ),
                    default_model=ollama_default,
                    num_ctx=ollama_num_ctx,
                    format=ollama_format,
                )
            ),
            default_model=ollama_default,
            declared_models=declared_by_backend.get("ollama", []),
        )
        return registry

    def _build_default_selector(self) -> ProviderSelector:
        config = self._load_config()
        return ProviderSelector(
            self._ensure_registry().providers,
            SelectorConfig(
                default_backend=str(config.get("default_provider") or "ollama"),
                default_model=str(config.get("default_model") or "qwen2.5-coder:14b"),
                default_models=_coerce_default_models(config.get("default_models")),
            ),
        )

    def _load_config(self) -> dict[str, Any]:
        """Load ``self.config_path`` as YAML, tolerating missing file / pyyaml / parse or IO errors.

        Every failure path falls back to an empty dict so the agent can
        still start with defaults — useful for tests, first-run setups,
        and operator misconfiguration that would otherwise crash the
        first skill call.
        """
        path = getattr(self, "config_path", "") or ""
        if not path:
            return {}
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("pyyaml not installed; using default reviewer config")
            return {}
        try:
            with open(path) as f:
                loaded = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("reviewer config %s not found; using defaults", path)
            return {}
        except yaml.YAMLError as exc:
            logger.warning(
                "reviewer config %s failed to parse (%s); using defaults", path, exc
            )
            return {}
        except OSError as exc:
            # Covers PermissionError, IsADirectoryError, and other IO failures.
            logger.warning(
                "reviewer config %s not readable (%s); using defaults", path, exc
            )
            return {}
        if isinstance(loaded, dict):
            return loaded
        if loaded is not None:
            logger.warning(
                "reviewer config %s must have a mapping at the YAML root; "
                "got %s; using defaults",
                path,
                type(loaded).__name__,
            )
        return {}


def create_reviewer_agent(
    *, agent_id: str, bus_url: str, config_path: str
) -> ReviewerAgent:
    agent = ReviewerAgent(
        agent_id=agent_id,
        bus_url=bus_url,
        config_path=config_path,
    )
    return agent


def main() -> None:
    from khonliang_bus import add_version_flag

    parser = argparse.ArgumentParser(
        prog="khonliang-reviewer",
        description="khonliang-reviewer bus agent",
    )
    add_version_flag(parser)
    parser.add_argument(
        "command",
        nargs="?",
        choices=["install", "uninstall"],
        help="install or uninstall from the bus",
    )
    parser.add_argument("--id", default="reviewer-primary")
    parser.add_argument("--bus", default="http://localhost:8787")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if args.command in ("install", "uninstall"):
        ReviewerAgent.from_cli(
            [
                args.command,
                "--id", args.id,
                "--bus", args.bus,
                "--config", args.config,
            ]
        )
        return

    agent = create_reviewer_agent(
        agent_id=args.id,
        bus_url=args.bus,
        config_path=args.config,
    )
    asyncio.run(agent.start())


if __name__ == "__main__":
    main()
