"""Tests for ``reviewer.distill.transforms.body_mode``.

Pins the contract:
- ``full`` is identity (default; misconfigured rule never silently
  trims).
- ``brief`` keeps the first sentence of every body and the summary;
  identity-preserving when content is already single-sentence.
- ``compact`` strips finding bodies entirely and reduces the
  summary to its first sentence; identity-preserving when nothing
  would actually change.
- Single-clause text without sentence terminators ("TODO comment
  lacks owner") is preserved intact under ``brief`` so titles
  without punctuation aren't truncated.
- An unknown mode raises ``ValueError``.
- Outlier preservation: a long-bodied concern surrounded by 20
  short nits keeps its first sentence under ``brief``; the
  sentence still names the bug ("Critical race condition...").
"""

from __future__ import annotations

import pytest
from khonliang_reviewer import ReviewFinding, ReviewResult

from reviewer.distill.transforms.body_mode import apply_body_mode
from reviewer.rules.distill import DistillConfig


def _result(summary: str, *findings: ReviewFinding) -> ReviewResult:
    return ReviewResult(request_id="req-test", summary=summary, findings=list(findings))


def _f(title: str, body: str = "", severity: str = "nit") -> ReviewFinding:
    return ReviewFinding(severity=severity, title=title, body=body)  # type: ignore[arg-type]


def test_full_mode_is_identity():
    """``full`` keeps the provider's output untouched, so the default
    DistillConfig() path stays identity through this transform.
    """
    result = _result(
        "Big summary. Multiple sentences. Final.",
        _f("t1", "Body sentence one. Body sentence two."),
    )
    out = apply_body_mode(result, DistillConfig())  # body_mode="full" default
    assert out is result


def test_brief_keeps_first_sentence_of_summary_and_bodies():
    result = _result(
        "Top-level finding. Some elaboration here.",
        _f("t1", "First. Second. Third."),
        _f("t2", "Only one."),
    )
    out = apply_body_mode(result, DistillConfig(body_mode="brief"))
    assert out.summary == "Top-level finding."
    assert [f.body for f in out.findings] == ["First.", "Only one."]


def test_brief_preserves_text_without_sentence_terminator():
    """Titles / bodies without periods (e.g. "TODO comment lacks
    owner") would lose all content under a naive split-on-period.
    The transform falls back to the whole text when no terminator
    is present.
    """
    result = _result(
        "TODO comment lacks owner",
        _f("t1", "Missing test for handler"),
    )
    out = apply_body_mode(result, DistillConfig(body_mode="brief"))
    assert out.summary == "TODO comment lacks owner"
    assert out.findings[0].body == "Missing test for handler"


def test_brief_handles_question_and_exclamation():
    """First-sentence detection covers ``?`` and ``!`` terminators
    too — code review findings sometimes use them in bodies
    ("Are these bounds correct? The off-by-one...").
    """
    result = _result(
        "Look here! There is a problem.",
        _f("t1", "Are these bounds correct? Likely off-by-one."),
    )
    out = apply_body_mode(result, DistillConfig(body_mode="brief"))
    assert out.summary == "Look here!"
    assert out.findings[0].body == "Are these bounds correct?"


def test_brief_is_identity_when_already_single_sentence():
    """Brief over content that's already single-sentence is a no-op;
    the transform returns the same object so the inert-config-style
    invariant extends to inert-content cases.
    """
    result = _result("One sentence.", _f("t", "Also one."))
    out = apply_body_mode(result, DistillConfig(body_mode="brief"))
    assert out is result


def test_compact_strips_bodies_and_briefs_summary():
    result = _result(
        "Summary first. Summary second.",
        _f("t1", "Body one. Body two."),
        _f("t2", "Whole body unchanged"),  # no terminator → still cleared
    )
    out = apply_body_mode(result, DistillConfig(body_mode="compact"))
    assert out.summary == "Summary first."
    assert all(f.body == "" for f in out.findings)


def test_compact_is_identity_when_already_compact():
    """Compact against bodies that are already empty AND a summary
    that's already a single sentence is a no-op.
    """
    result = _result("Single.", _f("t", ""), _f("t2", ""))
    out = apply_body_mode(result, DistillConfig(body_mode="compact"))
    assert out is result


def test_unknown_body_mode_raises():
    """Modes outside the Literal union should raise rather than
    silently degrading. Bus boundary may deliver wider payloads
    than the type system enforces.
    """
    result = _result("ok", _f("t"))
    with pytest.raises(ValueError, match="body_mode"):
        # type: ignore[arg-type]  -- intentional contract violation
        apply_body_mode(result, DistillConfig(body_mode="terse"))  # type: ignore[arg-type]


def test_outlier_concern_first_sentence_preserved_under_brief():
    """MS-B feature-preservation invariant adapted: an outlier
    concern body's *first sentence* still names the bug under
    ``brief`` mode, surrounded by 20 single-line nits. Brief
    shouldn't drop the headline of the most important finding.
    """
    findings = [_f(f"nit-{i}", "Short nit body.") for i in range(20)]
    findings.append(
        _f(
            "outlier",
            "Critical race condition in handler. Detailed analysis follows...",
            severity="concern",
        )
    )
    result = _result("Top summary. Detailed.", *findings)

    out = apply_body_mode(result, DistillConfig(body_mode="brief"))
    concern = next(f for f in out.findings if f.severity == "concern")
    assert concern.body == "Critical race condition in handler."


def test_empty_summary_and_bodies_stay_identity():
    """Edge case: an empty summary + empty bodies under any mode
    returns identity.
    """
    result = _result("", _f("t", ""))
    for mode in ("full", "brief", "compact"):
        out = apply_body_mode(result, DistillConfig(body_mode=mode))  # type: ignore[arg-type]
        assert out is result
