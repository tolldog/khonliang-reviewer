"""Tests for the audience-keyed distill rule table (fr_reviewer_de1694a8).

The rule table emits the ``(PolicyDecision, DistillConfig)`` pair from a
single ``evaluate`` call. Provider selection keys on diff size; distill
shaping keys on audience — the two are orthogonal, asserted here.
"""

from __future__ import annotations

from reviewer.rules import PolicyInput, decide_distill, evaluate
from reviewer.rules.distill import DistillConfig


# ---------------------------------------------------------------------------
# decide_distill (audience -> DistillConfig)
# ---------------------------------------------------------------------------


def test_decide_distill_default_audience_is_noop():
    cfg = decide_distill("agent_consumption")
    assert cfg == DistillConfig(audience="agent_consumption")
    # No shaping: identity over raw provider output.
    assert cfg.severity_floor == "nit"
    assert cfg.body_mode == "full"
    assert cfg.max_findings is None


def test_decide_distill_default_call_is_agent_consumption():
    assert decide_distill().audience == "agent_consumption"


def test_decide_distill_github_comment_is_aggressive():
    cfg = decide_distill("github_comment")
    assert cfg.severity_floor == "comment"
    assert cfg.body_mode == "compact"
    assert cfg.max_findings == 10
    assert cfg.audience == "github_comment"


def test_decide_distill_developer_handoff_trims_bodies_keeps_severities():
    cfg = decide_distill("developer_handoff")
    assert cfg.body_mode == "brief"
    assert cfg.severity_floor == "nit"  # keep every severity for the impl agent
    assert cfg.max_findings is None
    assert cfg.audience == "developer_handoff"


def test_decide_distill_audit_corpus_is_inert_marker():
    # run_pipeline short-circuits on audit_corpus; the config is the inert
    # default carrying only the audience marker.
    assert decide_distill("audit_corpus") == DistillConfig(audience="audit_corpus")


def test_decide_distill_unmapped_audience_carries_marker_only():
    assert decide_distill("human_review") == DistillConfig(audience="human_review")


# ---------------------------------------------------------------------------
# evaluate (single call -> provider + distill)
# ---------------------------------------------------------------------------


def test_policy_input_audience_defaults_to_agent_consumption():
    assert PolicyInput().audience == "agent_consumption"


def test_evaluate_returns_provider_and_distill_pair():
    pol, dis = evaluate(PolicyInput(kind="pr_diff", diff_line_count=50))
    assert pol.backend and pol.model            # provider half present
    assert dis.audience == "agent_consumption"  # distill half = default audience
    assert dis.max_findings is None             # no shaping for the default


def test_evaluate_provider_and_distill_are_orthogonal():
    # A large diff routes the PROVIDER to claude (size-driven), while the
    # github_comment audience shapes the DISTILL config (audience-driven) —
    # the two dimensions don't interfere.
    pol, dis = evaluate(
        PolicyInput(
            kind="pr_diff",
            diff_line_count=3000,
            diff_file_count=25,
            audience="github_comment",
        )
    )
    assert pol.backend == "claude_cli"   # >=2000 lines / >=20 files
    assert dis.body_mode == "compact"
    assert dis.max_findings == 10


def test_evaluate_audience_changes_only_distill_not_provider():
    base = PolicyInput(kind="pr_diff", diff_line_count=50)
    pol_a, dis_a = evaluate(base)
    pol_b, dis_b = evaluate(
        PolicyInput(kind="pr_diff", diff_line_count=50, audience="github_comment")
    )
    # Same provider decision regardless of audience...
    assert (pol_a.backend, pol_a.model) == (pol_b.backend, pol_b.model)
    # ...but different distill shaping.
    assert dis_a.max_findings is None
    assert dis_b.max_findings == 10
