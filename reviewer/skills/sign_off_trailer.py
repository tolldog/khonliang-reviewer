"""Format an ``Agent-Reviewed-by`` trailer line from a ``ReviewResult``.

Closes the trailer-format drift surface (`fr_reviewer_b846a19c`):
subagents have been hand-assembling the trailer line every commit,
and the format drifts (punctuation, ordering, whether the verdict
includes a reason). This module is the single source of truth for
the format; the agent's ``sign_off_trailer`` skill is a thin wrapper
that calls into :func:`build_trailer`.

The trailer shape (locked):

    Agent-Reviewed-by: <role>/<backend>/<model> <verdict>[: <reason>]

Where:

- ``role`` defaults to ``khonliang-reviewer`` (configurable for
  future cross-vendor reviewer roles).
- ``backend`` and ``model`` come from the ``ReviewResult`` (they
  carry the canonical names the provider used).
- ``verdict`` is computed from finding severity counts and the
  result's error_category — see :func:`compute_verdict`.
- ``reason`` is required for ``approved-with-findings`` and
  ``concerns-raised``; omitted for ``approved`` and
  ``escalated-approved`` (those verdicts produce a clean two-
  segment trailer). Caller-supplied reasons are honored on every
  verdict — the auto-reason logic only fires when the caller
  doesn't supply one. Reason is capped at ~80 chars so ``git log
  --oneline`` stays readable.

- ``approved-with-findings`` reasons follow the spec's locked
  ``<histogram> filtered`` shape — e.g. ``2 nits filtered``,
  ``1 comment + 2 nits filtered``. The ``filtered`` suffix is a
  participial adjective on the histogram meaning "flagged but
  non-blocking"; the numeric counts are the surviving findings.
  ``UsageEvent.findings_filtered_count`` (the count
  severity_filter dropped) is telemetry, not trailer copy — the
  trailer talks about what survived, not what was dropped.

Errored review results (``disposition == "errored"``) raise
:class:`ValueError` from :func:`compute_verdict` /
:func:`build_trailer` rather than producing a trailer — committing
an ``Agent-Reviewed-by`` line for a review that didn't actually
run is misleading sign-off provenance. Agent handlers catch the
exception and surface ``{error: ...}`` to the caller. The one
exception is a *timeout* (``error_category`` in
:data:`_TIMEOUT_SKIP_SIGNALS`): it maps to the ``review-skipped``
verdict with an **empty** trailer line plus a ``note`` — a timed-out
hot-tier pass degrades to a non-blocking "skip with a note" so the
pre-push gate proceeds to the cross-vendor reviewer instead of either
blocking the commit or faking a sign-off. The ``claude_cli_escalation``
case is NOT an errored disposition (it's a successful cross-vendor
escalation result that happens to set ``error_category`` for routing
purposes); it still maps to ``escalated-approved`` per spec.

All trailer segments (role, backend, model, reason) are sanitized
against newline characters before interpolation so a malicious or
malformed provider output can't inject additional trailer lines
into the commit message.

Parses cleanly under ``git interpret-trailers`` (the trailer key
``Agent-Reviewed-by`` is a custom key, intentionally distinct from
the standard ``Reviewed-by:`` so a future human reviewer's sign-off
doesn't collide).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from khonliang_reviewer import ReviewFinding, ReviewResult, severity_rank

logger = logging.getLogger(__name__)


Verdict = Literal[
    "approved",
    "approved-with-findings",
    "concerns-raised",
    "escalated-approved",
    "review-skipped",
]


_REASON_MAX_CHARS = 80

#: ``(backend, error_category)`` pairs that degrade to a non-blocking
#: ``review-skipped`` verdict instead of a hard ``ValueError``. A review
#: that *timed out* never produced output — it's not a failed review with
#: findings to act on, and it's not a clean approval to sign off. For the
#: pre-push gate it should "skip with a note" so the workflow proceeds to
#: the cross-vendor gate, rather than either blocking the commit or
#: fabricating a sign-off.
#:
#: Scoped to the **local hot-tier** providers only: ollama's
#: ``backend_timeout`` and claude_cli's ``subprocess_timeout`` (the
#: hot-tier + escalation pass that the cross-vendor gate backstops). It
#: must NOT match ``codex_cli`` / ``gh_copilot``, which *also* emit
#: ``subprocess_timeout`` but ARE the cross-vendor gate — silently
#: skipping a timed-out authoritative review would let an unreviewed
#: commit through with no fallback. Keying on the (backend, category) pair
#: (not the shared category alone) keeps that distinction. Other errored
#: categories (malformed_envelope, auth_not_provisioned, …) stay hard
#: errors everywhere — real failures, not transient "couldn't run in time".
_TIMEOUT_SKIP_SIGNALS = frozenset(
    {
        ("ollama", "backend_timeout"),
        ("claude_cli", "subprocess_timeout"),
    }
)


def compute_verdict(result: ReviewResult) -> Verdict:
    """Map a ``ReviewResult`` to a four-value verdict.

    Mapping (per ``specs/MS-D/spec.md``):

    - ``error_category == "claude_cli_escalation"`` → ``escalated-approved``
      (the result came from the cross-vendor escalation path, so the
      sign-off records that vendor diversity).
    - At least one *actionable* concern-severity finding → ``concerns-raised``.
      A concern is actionable when it carries a concrete ``suggestion`` OR a
      diff-anchored location (``path`` + ``line``). A concern with NEITHER is
      a non-actionable paraphrase (the dogfood pattern where a local reviewer
      flips the verdict on a prose restatement) — it is reported in the
      findings but does NOT block; each is logged (fr_reviewer_fa2dd997).
    - Otherwise, ≥1 finding (comment / nit / a discounted concern)
      → ``approved-with-findings``.
    - Zero findings → ``approved``.

    Findings with non-string or unrecognized severity are treated as
    concerns and kept loud (always blocking) so a malformed-severity row
    never silently downgrades the verdict — only proper ``concern``-severity
    rows are subject to the actionability check.

    An errored disposition whose ``error_category`` is a timeout
    (:data:`_TIMEOUT_SKIP_SIGNALS`) maps to ``review-skipped``
    rather than raising — a timed-out review never ran to
    completion, so the pre-push gate should skip with a note (and
    no trailer) and let the cross-vendor gate carry the sign-off,
    instead of blocking the commit. Any *other* errored disposition
    still raises :class:`ValueError` — committing a sign-off trailer
    for a review that genuinely failed would be misleading
    provenance. The ``claude_cli_escalation`` case is checked first
    so a successful escalation result (which sets ``error_category``
    for routing but is NOT disposition=errored) still maps to
    ``escalated-approved``.
    """
    if result.error_category == "claude_cli_escalation":
        return "escalated-approved"
    if result.disposition == "errored":
        # A *local hot-tier* timeout degrades to a non-blocking skip (the
        # review never ran to completion); other errored categories — and
        # timeouts from the cross-vendor gate providers (codex/copilot, which
        # share the subprocess_timeout category) — stay hard errors.
        if (result.backend, result.error_category) in _TIMEOUT_SKIP_SIGNALS:
            return "review-skipped"
        message = result.error or "(no error message)"
        raise ValueError(
            f"cannot format sign-off trailer for errored review: {message}"
        )

    counts = _severity_counts(result)
    # Malformed severity stays loud — never silently downgraded.
    if counts["unknown"] > 0:
        return "concerns-raised"

    blocking, discounted = _partition_concerns(result)
    for f in discounted:
        logger.info(
            "sign_off_trailer: discounting non-actionable concern %r — no "
            "suggestion and no diff-anchored (path+line) location; reported "
            "but non-blocking (fr_reviewer_fa2dd997)",
            f.title or "(no title)",
        )
    if blocking:
        return "concerns-raised"
    if counts["comment"] > 0 or counts["nit"] > 0 or discounted:
        return "approved-with-findings"
    return "approved"


def _is_actionable_concern(f: ReviewFinding) -> bool:
    """A concern blocks the verdict only when it's *actionable*: it carries
    a concrete ``suggestion`` OR a diff-anchored location (``path`` + ``line``).

    A concern with neither is a non-actionable paraphrase — the dogfood
    pattern (fr_reviewer_fa2dd997) where local reviewers flip the verdict on
    a prose restatement of the diff. Such a concern is still reported in the
    findings; it just doesn't flip the verdict to ``concerns-raised``.
    """
    # Validate types at the bus boundary: a deserialized finding can carry
    # any JSON shape, so a malformed path (dict/list) or non-int line must
    # NOT count as a diff-anchored location (which would wrongly keep a
    # noise concern blocking). path must be a non-empty string; line a real
    # int (``bool`` is an int subclass — exclude it).
    has_suggestion = isinstance(f.suggestion, str) and bool(f.suggestion.strip())
    has_location = (
        isinstance(f.path, str)
        and bool(f.path.strip())
        and isinstance(f.line, int)
        and not isinstance(f.line, bool)
        # A real diff anchor is a 1-based line number; 0 / negative aren't
        # anchorable (GitHub rejects them), so they don't make a concern
        # actionable.
        and f.line > 0
    )
    return has_suggestion or has_location


def _partition_concerns(
    result: ReviewResult,
) -> tuple[list[ReviewFinding], list[ReviewFinding]]:
    """Split ``concern``-severity findings into ``(blocking, discounted)``.

    Only proper ``concern`` rows are partitioned; unknown / malformed
    severities are handled separately in :func:`compute_verdict` (always
    loud). ``blocking`` are the actionable concerns; ``discounted`` are the
    non-actionable ones (reported but non-blocking).
    """
    blocking: list[ReviewFinding] = []
    discounted: list[ReviewFinding] = []
    for f in result.findings:
        if f.severity != "concern":
            continue
        (blocking if _is_actionable_concern(f) else discounted).append(f)
    return blocking, discounted


def build_trailer(
    result: ReviewResult,
    *,
    role: str = "khonliang-reviewer",
    reason: str = "",
) -> dict[str, Any]:
    """Build a structured trailer record from a ``ReviewResult``.

    Returns ``{verdict, trailer_line}``. The caller stitches the
    trailer line into a commit message; the verdict is exposed as a
    structured field so subagents can branch on it.

    ``reason`` is auto-derived from the finding histogram when not
    supplied. An empty auto-reason for the ``approved`` and
    ``escalated-approved`` verdicts is intentional (no reason fits
    the trailer format for those cases). Caller-supplied ``reason``
    overrides the auto-derived one for both shapes.

    The ``review-skipped`` verdict (a timed-out review) is special:
    it returns ``{verdict, trailer_line: "", note}`` with an **empty**
    trailer line — committing an ``Agent-Reviewed-by`` line for a
    review that timed out would falsely claim a sign-off that never
    happened (CLAUDE.md: "omit rather than fake"). ``note`` carries
    the truthful skip reason (the caller-supplied ``reason`` if any,
    else the timeout message) so the caller can record *why* it
    skipped without minting a trailer.
    """
    verdict = compute_verdict(result)
    if verdict == "review-skipped":
        note = reason or _skip_note(result)
        return {
            "verdict": verdict,
            "trailer_line": "",
            "note": _truncate_reason(_sanitize_segment(note)),
        }
    # role/backend/model are joined by "/" in the trailer's locked
    # ``role/backend/model`` shape, so a "/" inside any of them would
    # add phantom path segments and break unambiguous parsing.
    # Model ids in this repo can carry "/" (see
    # tests/test_benchmark_sweep.py's "kimi-k2/5/cloud" example);
    # _sanitize_path_segment normalizes those to "-" alongside the
    # newline-strip every segment gets.
    #
    # Coerce backend/model to str via the placeholder fallback when
    # they're non-string. ReviewResult's type contract calls them
    # str, but a deserialized payload from the bus boundary can carry
    # any JSON-shape — passing a non-string into _sanitize_path_segment
    # would crash on ``.split()``. The placeholder fallback keeps the
    # trailer line shape intact for telemetry / auditing while
    # signaling the malformed input.
    backend = _sanitize_path_segment(_coerce_segment(result.backend, "unknown-backend"))
    model = _sanitize_path_segment(_coerce_segment(result.model, "unknown-model"))
    role = _sanitize_path_segment(_coerce_segment(role, "unknown-role"))

    if not reason:
        reason = _auto_reason(result, verdict)
    # ``reason`` keeps "/" — operator-supplied reasons often carry
    # path references like "reviewer/agent.py:42" that should
    # survive intact. Newlines / tabs are still collapsed (the
    # injection-prevention concern).
    reason = _sanitize_segment(reason)

    # The trailer emits the ``: <reason>`` segment whenever a
    # non-empty reason exists, regardless of verdict. Per the spec,
    # ``approved`` and ``escalated-approved`` get an empty
    # auto-reason (clean two-segment trailer), while
    # ``approved-with-findings`` and ``concerns-raised`` get a
    # populated auto-reason from the surviving-finding histogram.
    # Caller-supplied ``reason`` is honored on every verdict — a
    # caller can attach prose to a clean approval if they want, the
    # auto-reason logic just doesn't generate one.
    if reason:
        reason = _truncate_reason(reason)
        trailer_line = (
            f"Agent-Reviewed-by: {role}/{backend}/{model} {verdict}: {reason}"
        )
    else:
        trailer_line = f"Agent-Reviewed-by: {role}/{backend}/{model} {verdict}"

    return {"verdict": verdict, "trailer_line": trailer_line}


def _skip_note(result: ReviewResult) -> str:
    """Truthful note for a ``review-skipped`` verdict — the timeout message
    the provider recorded, or a generic fallback when it's empty.
    """
    message = result.error if isinstance(result.error, str) and result.error.strip() else ""
    if message:
        return f"review skipped — {message}"
    return "review skipped — hot-tier review timed out"


def _severity_counts(result: ReviewResult) -> dict[str, int]:
    """Count findings by severity. ``unknown`` covers non-string and
    unparseable severity strings.
    """
    counts = {"nit": 0, "comment": 0, "concern": 0, "unknown": 0}
    for f in result.findings:
        severity = f.severity
        if not isinstance(severity, str):
            counts["unknown"] += 1
            continue
        try:
            severity_rank(severity)
        except ValueError:
            counts["unknown"] += 1
            continue
        # Severity is a known str — bucket it.
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def _auto_reason(result: ReviewResult, verdict: Verdict) -> str:
    """Build a default reason from the finding histogram + first-concern.

    For ``concerns-raised`` the reason names the count and the first
    *contributing* finding's title (or category) so ``git log
    --oneline`` shows the headline. "Contributing" = severity is
    ``concern`` OR an unparseable / non-string severity (since
    unknown rows count toward concerns per ``compute_verdict``);
    leading nit/comment rows are skipped so the anchor doesn't
    accidentally point at a low-severity finding.

    For ``approved-with-findings`` the reason follows the spec's
    locked "<histogram> filtered" shape — e.g. ``2 nits filtered``,
    ``1 comment + 2 nits filtered``. The ``filtered`` suffix is a
    participial adjective on the histogram meaning "flagged but
    non-blocking". The numeric counts are the surviving findings
    in ``result.findings``; the spec deliberately doesn't reference
    ``UsageEvent.findings_filtered_count`` here (that field is
    telemetry, not trailer copy).

    For ``approved`` / ``escalated-approved`` the reason is empty —
    the trailer omits the segment so a clean approval reads
    cleanly.
    """
    counts = _severity_counts(result)

    if verdict == "concerns-raised":
        concern_count = counts["concern"] + counts["unknown"]
        anchor_finding = _first_concern_or_unknown(result)
        if anchor_finding is None:
            return (
                f"{concern_count} concern"
                + ("s" if concern_count != 1 else "")
            )
        anchor = (
            anchor_finding.category
            or anchor_finding.title
            or "(no title)"
        )
        plural = "s" if concern_count != 1 else ""
        return f"{concern_count} concern{plural}: {anchor}"

    if verdict == "approved-with-findings":
        parts = []
        if counts["comment"]:
            parts.append(
                f"{counts['comment']} comment"
                + ("s" if counts["comment"] != 1 else "")
            )
        if counts["nit"]:
            parts.append(
                f"{counts['nit']} nit" + ("s" if counts["nit"] != 1 else "")
            )
        if parts:
            # Spec-locked "<histogram> filtered" shape (MS-D). Discounted
            # concerns are surfaced via the INFO log, not appended here, so
            # the shape stays a single comment/nit histogram.
            return " + ".join(parts) + " filtered"
        # No surviving comment/nit, so the verdict came purely from
        # discounted concern(s). MS-D requires a reason for
        # approved-with-findings, so name the discount (this is the only
        # non-histogram approved-with-findings shape).
        _, discounted = _partition_concerns(result)
        if discounted:
            n = len(discounted)
            return f"{n} concern{'s' if n != 1 else ''} discounted"
        return ""

    # approved / escalated-approved.
    return ""


def _first_concern_or_unknown(result: ReviewResult):
    """Return the first finding whose severity counts toward
    ``concerns-raised`` — i.e. severity ``concern`` OR an
    unparseable / non-string severity. Skips leading nit/comment
    rows so the anchor reflects what actually triggered the verdict.
    Returns ``None`` when no contributing row exists (defensive;
    shouldn't fire for this verdict).
    """
    for f in result.findings:
        severity = f.severity
        if not isinstance(severity, str):
            return f  # unknown (non-string) — always blocking
        if severity == "concern":
            if _is_actionable_concern(f):
                return f  # actionable concern — blocking
            continue  # discounted (non-actionable) concern — skip
        try:
            severity_rank(severity)
        except ValueError:
            return f  # unknown (unparseable) — blocking
    return None


def _coerce_segment(value: Any, placeholder: str) -> str:
    """Coerce ``value`` to a non-empty string for trailer composition.

    ReviewResult fields are typed as ``str`` but the bus boundary can
    deliver any JSON shape (None, int, dict, list). Passing a non-
    string into the sanitizer would crash on ``.split()``; the
    placeholder fallback keeps the trailer's three-segment shape
    intact while signaling the malformed input. Empty strings also
    fall back to the placeholder so the trailer never carries a
    zero-length segment that would break tokenization.
    """
    if isinstance(value, str) and value:
        return value
    return placeholder


def _sanitize_segment(value: str) -> str:
    """Collapse whitespace + strip newlines from a trailer segment.

    The trailer format is single-line. A malicious or malformed
    provider output that injects ``\\n`` / ``\\r`` into a finding
    title (which feeds the auto-reason anchor) or a backend / model
    string could otherwise forge additional trailer keys in the
    commit message — e.g. an attacker-supplied finding titled
    ``hack\\nApproved-by: someone-else`` would produce a trailer
    block claiming a second sign-off. Replace any whitespace
    sequence (including CR/LF and tabs) with a single space and
    strip the result.

    Idempotent: re-sanitizing already-clean input is a no-op.
    """
    if not value:
        return value
    # Collapse every whitespace run (CR, LF, tab, multiple spaces)
    # into a single space, then strip leading/trailing space.
    return " ".join(value.split())


def _sanitize_path_segment(value: str) -> str:
    """Sanitize a segment that's joined into the locked
    ``role/backend/model`` path of the trailer line.

    Stricter than :func:`_sanitize_segment`: in addition to the
    newline / whitespace normalization, replaces ``/``, ``\\``,
    and the remaining single-space characters with ``-`` so the
    triple stays a single space-delimited token in the locked
    ``Agent-Reviewed-by: <role>/<backend>/<model> <verdict>...``
    format. Operator-supplied roles, provider-reported backend
    names, and model ids all run through this — model ids carry
    the most variability (e.g. ``kimi-k2/5/cloud``).
    """
    cleaned = _sanitize_segment(value)
    if not cleaned:
        return cleaned
    # Translate path-separator chars AND any remaining spaces to
    # ``-``. Preserves digits, dots, colons, underscores so
    # canonical model ids stay readable in the trailer.
    return (
        cleaned
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "-")
    )


def _truncate_reason(reason: str) -> str:
    """Cap the reason segment so the trailer stays readable in
    ``git log --oneline`` (which already truncates at terminal width).
    """
    if len(reason) <= _REASON_MAX_CHARS:
        return reason
    return reason[: _REASON_MAX_CHARS - 1].rstrip() + "…"


__all__ = ["Verdict", "build_trailer", "compute_verdict"]
