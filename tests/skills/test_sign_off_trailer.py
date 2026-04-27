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

from khonliang_reviewer import ReviewFinding, ReviewResult, UsageEvent

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
    findings_filtered_count: int = 0,
) -> ReviewResult:
    usage = None
    if findings_filtered_count:
        usage = UsageEvent(
            timestamp=0.0,
            backend=backend,
            model=model,
            input_tokens=0,
            output_tokens=0,
            duration_ms=0,
            findings_filtered_count=findings_filtered_count,
        )
    return ReviewResult(
        request_id="req-test",
        summary="ok",
        findings=findings or [],
        backend=backend,
        model=model,
        error_category=error_category,  # type: ignore[arg-type]
        usage=usage,
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


def test_approved_with_findings_trailer_matches_spec_filtered_suffix():
    """Spec acceptance example shape: ``<histogram> filtered`` —
    e.g. ``1 comment + 2 nits filtered``. The numeric counts are
    surviving findings; ``filtered`` is a participial adjective
    on the histogram meaning "flagged but non-blocking".
    """
    r = _result(findings=[_f("nit"), _f("nit"), _f("comment")])
    out = build_trailer(r)
    assert out["verdict"] == "approved-with-findings"
    # Auto-reason enumerates non-zero buckets in fixed severity
    # order (comments before nits) regardless of count — the
    # trailer reads with the higher-severity headline first.
    assert "1 comment" in out["trailer_line"]
    assert "2 nits" in out["trailer_line"]
    # And specifically: comments bucket appears before nits bucket.
    assert out["trailer_line"].index("comment") < out["trailer_line"].index("nit")
    # Spec-locked suffix.
    assert out["trailer_line"].endswith("filtered")
    # Spec-example shape for the simpler "2 nits" case.
    r2 = _result(findings=[_f("nit"), _f("nit")])
    out2 = build_trailer(r2)
    assert out2["trailer_line"].endswith("approved-with-findings: 2 nits filtered")


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


def test_concerns_anchor_skips_leading_nits_and_comments():
    """Anchor selection must point at the first finding that
    actually triggered the concerns verdict — a concern-severity
    or unknown-severity row — skipping leading nits/comments. A
    naive ``findings[0]`` fallback would mis-anchor the trailer
    at a low-severity row when the concern is later in the list.
    """
    weird = ReviewFinding(severity="bogus", title="malformed-row", body="b")  # type: ignore[arg-type]
    r = _result(
        findings=[
            ReviewFinding(severity="nit", title="leading-nit", body="b"),  # type: ignore[arg-type]
            ReviewFinding(severity="comment", title="leading-comment", body="b"),  # type: ignore[arg-type]
            weird,
        ]
    )
    out = build_trailer(r)
    assert out["verdict"] == "concerns-raised"
    # Anchor names the malformed row, not "leading-nit" / "leading-comment".
    assert "malformed-row" in out["trailer_line"]
    assert "leading-nit" not in out["trailer_line"]
    assert "leading-comment" not in out["trailer_line"]


def test_filtered_count_is_telemetry_not_trailer_copy():
    """``UsageEvent.findings_filtered_count`` counts how many
    findings severity_filter dropped — that's telemetry. The
    trailer talks about the SURVIVING histogram (which is what
    the reviewer is signing off on), so the trailer should NOT
    surface the dropped count as an extra segment. Spec example
    "2 nits filtered" is the surviving 2 nits, not 2 nits-
    that-were-dropped.

    Pinning this so a future revision doesn't mix telemetry
    into the trailer copy and break the spec-locked shape.
    """
    r = _result(
        findings=[ReviewFinding(severity="nit", title="t", body="b")],  # type: ignore[arg-type]
        findings_filtered_count=3,
    )
    out = build_trailer(r)
    assert out["verdict"] == "approved-with-findings"
    # Surviving histogram + " filtered" suffix; no separate "+ N filtered".
    assert out["trailer_line"].endswith("approved-with-findings: 1 nit filtered")
    assert "+ 3" not in out["trailer_line"]


def test_approved_with_zero_surviving_omits_reason():
    """When zero findings survive (verdict=approved), the trailer
    has no reason segment regardless of how many findings the
    floor dropped — the spec says reason is omitted for approved.
    A clean approval reads cleanly.
    """
    r = _result(findings=[], findings_filtered_count=4)
    out = build_trailer(r)
    assert out["verdict"] == "approved"
    # No reason segment.
    assert ": " not in out["trailer_line"].split("approved", 1)[1]


def test_space_in_role_sanitized():
    """Operator-supplied ``role`` (e.g. typo'd "my role") with
    embedded spaces would otherwise break the trailer's
    ``<role>/<backend>/<model> <verdict>`` tokenization — the
    triple is supposed to be a single space-delimited token.
    The path-segment sanitizer rewrites internal spaces to "-"
    so the locked shape holds even on careless override input.
    """
    r = _result()
    out = build_trailer(r, role="my role with spaces")
    line = out["trailer_line"]
    # The role/backend/model triple is the first space-delimited
    # token after "Agent-Reviewed-by:". A space inside any
    # segment would split the triple into multiple tokens.
    after_key = line.split(": ", 1)[1]
    triple = after_key.split(" ", 1)[0]
    # Sanitized role uses dashes; the triple stays single-token.
    assert triple.startswith("my-role-with-spaces/")
    # Original spaces don't leak through.
    assert "my role" not in line


def test_errored_disposition_raises():
    """Provider failure (disposition='errored') must NOT produce a
    sign-off trailer — the trailer would falsely advertise an
    approval against a review that didn't run. ValueError surfaces
    so the agent handler converts it to an error envelope rather
    than committing a misleading sign-off line.
    """
    import pytest

    r = ReviewResult(
        request_id="req-test",
        summary="",
        findings=[],
        backend="ollama",
        model="qwen2.5-coder:14b",
        disposition="errored",
        error="connection refused",
    )
    with pytest.raises(ValueError, match="errored"):
        build_trailer(r)


def test_escalated_approved_not_blocked_by_disposition_check():
    """A claude_cli escalation result is NOT disposition='errored'
    (it's a successful cross-vendor escalation that happens to set
    error_category for routing); the disposition check must run
    AFTER the escalation check so successful escalations still
    map to 'escalated-approved'.
    """
    r = _result(error_category="claude_cli_escalation")
    out = build_trailer(r)
    assert out["verdict"] == "escalated-approved"


def test_newline_in_finding_title_sanitized():
    """A finding title containing CR/LF would otherwise inject
    additional trailer keys into the commit message — e.g. a
    title like ``hack\\nApproved-by: someone-else`` would forge
    a second sign-off. The sanitizer collapses any whitespace
    sequence (CR, LF, tabs, multi-space) into a single space so
    the trailer stays single-line.
    """
    weird = ReviewFinding(
        severity="concern",
        title="legit\nApproved-by: forged-signer",
        body="b",
    )  # type: ignore[arg-type]
    r = _result(findings=[weird])
    out = build_trailer(r)
    line = out["trailer_line"]
    # Trailer must contain exactly one line.
    assert "\n" not in line
    assert "\r" not in line
    # The forged Approved-by trailer is now part of the reason
    # segment as a single-line string, not a separate trailer.
    assert "Approved-by: forged-signer" in line  # exists as text...
    # ...but on the same line as our trailer key.
    assert line.startswith("Agent-Reviewed-by:")
    assert line.count("Agent-Reviewed-by:") == 1


def test_newline_in_caller_reason_sanitized():
    """Same threat surface via caller-supplied ``reason``."""
    r = _result(findings=[ReviewFinding(severity="concern", title="t", body="b")])  # type: ignore[arg-type]
    out = build_trailer(r, reason="legit\nReviewed-by: forged")
    line = out["trailer_line"]
    assert "\n" not in line
    assert "\r" not in line
    assert line.startswith("Agent-Reviewed-by:")


def test_newline_in_role_sanitized():
    """Same threat surface via custom ``role``."""
    r = _result()
    out = build_trailer(r, role="role-with\nnewline")
    line = out["trailer_line"]
    assert "\n" not in line


def test_slash_in_model_id_sanitized():
    """Model ids in this ecosystem can carry "/" (e.g.
    ``kimi-k2/5/cloud`` per tests/test_benchmark_sweep.py). The
    trailer's locked ``role/backend/model`` shape needs the model
    id to be a single segment — the path-segment sanitizer
    rewrites "/" to "-" so the trailer keeps its three-part
    shape and remains unambiguously parseable.
    """
    r = _result(model="kimi-k2/5/cloud")
    out = build_trailer(r)
    # Trailer still has exactly three "/" — between role+backend
    # and backend+model. (No spurious extras from the model id.)
    # The trailer line is "Agent-Reviewed-by: <role>/<backend>/<model>..."
    # so we expect 2 "/" in the role/backend/model triple.
    after_key = out["trailer_line"].split(": ", 1)[1]
    triple = after_key.split(" ", 1)[0]
    assert triple.count("/") == 2
    # Model segment carries the rewritten id.
    assert "kimi-k2-5-cloud" in triple


def test_backslash_in_segment_sanitized():
    """Symmetric to the "/" case — backslash is also a path
    separator on the platforms this trailer might be parsed on,
    so it gets the same treatment.
    """
    r = _result(model="weird\\model\\id")
    out = build_trailer(r)
    assert "\\" not in out["trailer_line"]
    assert "weird-model-id" in out["trailer_line"]


def test_slash_in_reason_preserved():
    """The reason segment keeps "/" — operator-supplied reasons
    often carry path references like "reviewer/agent.py:42" that
    should survive intact for ``git log --oneline`` readability.
    The path-sanitizer is path-segment-only.
    """
    r = _result(findings=[_f("concern", "race")])
    out = build_trailer(r, reason="see reviewer/agent.py:42")
    assert "reviewer/agent.py:42" in out["trailer_line"]


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
