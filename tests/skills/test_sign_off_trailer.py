"""Tests for ``reviewer.skills.sign_off_trailer``.

Pins the trailer format + verdict mapping so subagents stop drifting
in trailer-line shape across commits. The format is locked to:

    Agent-Reviewed-by: <role>/<backend>/<model> <verdict>[: <reason>]

Verdict mapping (per ``specs/MS-D/spec.md``):
- ``approved``: zero findings.
- ``approved-with-findings``: ≥1 nit/comment, zero concerns.
- ``concerns-raised``: ≥1 concern (or unknown-severity).
- ``escalated-approved``: result.error_category == "claude_cli_escalation".

Trailer format integration: parses cleanly under ``git
interpret-trailers`` (test asserts the line passes the standard
trailer regex used by git).
"""

from __future__ import annotations

import re

from khonliang_reviewer import ReviewFinding, ReviewResult

from reviewer.skills.sign_off_trailer import (
    build_trailer,
    compute_verdict,
)


def _result(
    *,
    findings: list[ReviewFinding] | None = None,
    backend: str = "ollama",
    model: str = "qwen2.5-coder:14b",
    error_category: str = "",
) -> ReviewResult:
    return ReviewResult(
        request_id="req-test",
        summary="ok",
        findings=findings or [],
        backend=backend,
        model=model,
        error_category=error_category,  # type: ignore[arg-type]
    )


def _f(severity: str, title: str = "t", category: str = "") -> ReviewFinding:
    return ReviewFinding(severity=severity, title=title, body="b", category=category)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compute_verdict
# ---------------------------------------------------------------------------


def test_zero_findings_is_approved():
    assert compute_verdict(_result()) == "approved"


def test_only_nits_is_approved_with_findings():
    r = _result(findings=[_f("nit"), _f("nit")])
    assert compute_verdict(r) == "approved-with-findings"


def test_comments_only_is_approved_with_findings():
    r = _result(findings=[_f("comment", "minor wording")])
    assert compute_verdict(r) == "approved-with-findings"


def test_any_concern_is_concerns_raised():
    r = _result(findings=[_f("nit"), _f("concern", "race condition")])
    assert compute_verdict(r) == "concerns-raised"


def test_escalation_takes_precedence_over_findings():
    """A claude_cli escalation result is recorded as
    ``escalated-approved`` regardless of finding counts. The cross-
    vendor escalation is itself the sign-off signal; downgrading to
    'concerns-raised' just because Claude returned findings would
    lose the escalation provenance.
    """
    r = _result(
        findings=[_f("concern", "race")],
        error_category="claude_cli_escalation",
    )
    assert compute_verdict(r) == "escalated-approved"


def test_unknown_severity_treated_as_concern():
    """A finding with an unknown severity string isn't silently
    downgraded to nit/comment rank — it counts toward concerns.
    The trailer stays loud about malformed-severity rows so a
    provider bug doesn't accidentally produce an 'approved'
    sign-off.
    """
    weird = ReviewFinding(severity="bogus", title="weird", body="b")  # type: ignore[arg-type]
    r = _result(findings=[weird])
    assert compute_verdict(r) == "concerns-raised"


def test_non_string_severity_treated_as_concern():
    """Same convention for non-string severities — keep the trailer
    loud about malformed payloads.
    """
    weird = ReviewFinding(severity=None, title="weird", body="b")  # type: ignore[arg-type]
    r = _result(findings=[weird])
    assert compute_verdict(r) == "concerns-raised"


# ---------------------------------------------------------------------------
# build_trailer
# ---------------------------------------------------------------------------


def test_approved_trailer_format():
    out = build_trailer(_result())
    assert out["verdict"] == "approved"
    assert (
        out["trailer_line"]
        == "Agent-Reviewed-by: khonliang-reviewer/ollama/qwen2.5-coder:14b approved"
    )


def test_approved_with_findings_trailer_includes_count():
    r = _result(findings=[_f("nit"), _f("nit"), _f("comment")])
    out = build_trailer(r)
    assert out["verdict"] == "approved-with-findings"
    # Auto-reason names both buckets in count order; trailer carries
    # the full string.
    assert "1 comment" in out["trailer_line"]
    assert "2 nits" in out["trailer_line"]


def test_concerns_raised_trailer_uses_first_concern_anchor():
    """For ``concerns-raised`` the auto-reason names the count + the
    first concern's category (preferred) or title (fallback) — short
    and stable for ``git log --oneline`` scanning.
    """
    r = _result(
        findings=[
            _f("comment"),  # nit/comment first; should be ignored as anchor
            _f("concern", "Race condition in handler", category="race_condition"),
        ]
    )
    out = build_trailer(r)
    assert out["verdict"] == "concerns-raised"
    # Anchor is the first concern's category (preferred).
    assert "race_condition" in out["trailer_line"]
    assert "1 concern" in out["trailer_line"]


def test_escalated_approved_trailer_omits_reason():
    r = _result(
        findings=[_f("concern", "race")],
        error_category="claude_cli_escalation",
    )
    out = build_trailer(r)
    assert out["verdict"] == "escalated-approved"
    # Spec: no reason segment for approved / escalated-approved.
    assert ": " not in out["trailer_line"].split(" escalated-approved")[1]


def test_caller_supplied_reason_overrides_auto():
    r = _result(findings=[_f("concern", "anything")])
    out = build_trailer(r, reason="false positive: separate control-flow branches")
    assert "false positive" in out["trailer_line"]
    # Auto-derived 'concerns: anything' should NOT appear.
    assert "anything" not in out["trailer_line"]


def test_custom_role_applied():
    r = _result()
    out = build_trailer(r, role="claude-via-codex")
    assert out["trailer_line"].startswith(
        "Agent-Reviewed-by: claude-via-codex/ollama/qwen2.5-coder:14b"
    )


def test_long_reason_truncated_with_ellipsis():
    """Reason segment is capped so ``git log --oneline`` stays
    readable. The cap is enforced via :func:`_truncate_reason` —
    preserves head, drops tail, appends an ellipsis so readers see
    the truncation rather than mistaking the trailer for a
    too-long-but-complete reason.
    """
    very_long = "x" * 200
    r = _result(findings=[_f("concern", "race")])
    out = build_trailer(r, reason=very_long)
    # Trailer line must not contain the full 200-char reason.
    reason_in_line = out["trailer_line"].split(" concerns-raised: ")[1]
    assert len(reason_in_line) < 100
    assert reason_in_line.endswith("…")


def test_missing_backend_or_model_uses_placeholders():
    """ReviewResults without backend/model populated (rare; defensive
    path) get readable placeholders instead of empty path segments
    that would break the trailer's three-part role/backend/model
    layout.
    """
    r = ReviewResult(request_id="req-test", summary="")
    out = build_trailer(r)
    assert "unknown-backend" in out["trailer_line"]
    assert "unknown-model" in out["trailer_line"]


def test_trailer_parses_via_git_trailer_regex():
    """The trailer format must be valid for ``git interpret-trailers``.
    Standard git trailer shape is ``Token: Value`` where Token matches
    ``[A-Za-z0-9-]+``. Our token is ``Agent-Reviewed-by`` (custom
    key intentionally distinct from ``Reviewed-by:``).
    """
    r = _result(findings=[_f("concern", "race")])
    out = build_trailer(r)
    line = out["trailer_line"]
    m = re.match(r"^([A-Za-z0-9-]+): (.+)$", line)
    assert m is not None
    assert m.group(1) == "Agent-Reviewed-by"
