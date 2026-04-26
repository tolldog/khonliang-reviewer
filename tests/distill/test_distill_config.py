"""Tests for ``reviewer.rules.distill.DistillConfig``.

The dataclass is the rule-table → distill-pipeline contract; the
tests pin the field set + defaults so future transform-PRs can
depend on a stable shape rather than chasing this file. (See
``specs/MS-B/spec.md`` for the design context.)
"""

from __future__ import annotations

import dataclasses

from reviewer.rules.distill import DistillConfig


def test_default_audience_is_agent_consumption():
    """``agent_consumption`` is the safe default — most reviews today
    are bot-to-bot, so the pipeline should keep findings as-emitted
    rather than trim aggressively. Aggressive shaping (``github_comment``,
    ``audit_corpus``) is opt-in via the rule-table row.
    """
    cfg = DistillConfig()
    assert cfg.audience == "agent_consumption"


def test_default_severity_floor_keeps_everything():
    """Default floor is ``nit`` (the lowest severity), i.e. no
    filtering. Filtering is opt-in via the rule-table row; the
    default keeps every finding the provider emits so a misconfigured
    rule never silently drops concerns.
    """
    assert DistillConfig().severity_floor == "nit"


def test_default_body_mode_is_full():
    """Same principle as severity_floor: default is no shaping, so
    a misconfigured rule never silently strips finding bodies.
    """
    assert DistillConfig().body_mode == "full"


def test_default_consensus_is_off():
    """Consensus runs at the selector layer (per Open Question #2
    resolution in MS-B); the default keeps it off so single-provider
    review stays the cheap path.
    """
    assert DistillConfig().consensus is False


def test_default_dedup_is_none():
    assert DistillConfig().dedup == "none"


def test_default_max_findings_is_unbounded():
    """``None`` means no cap. A misconfigured rule never silently
    truncates by hitting a non-zero default.
    """
    assert DistillConfig().max_findings is None


def test_dataclass_is_frozen():
    """A single rule-table evaluation can be safely shared across
    transforms; freezing the dataclass means callers can't mutate
    fields mid-run by accident.
    """
    cfg = DistillConfig()
    try:
        cfg.severity_floor = "concern"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("DistillConfig must be frozen")


def test_field_set_matches_spec():
    """Pins the exact field names so transform-PRs can rely on the
    contract. Adding a new field is fine (transform PRs may extend
    it); removing or renaming an existing field is a contract break
    that should fail this test loudly.
    """
    expected = {
        "severity_floor",
        "body_mode",
        "consensus",
        "dedup",
        "max_findings",
        "audience",
    }
    actual = {f.name for f in dataclasses.fields(DistillConfig)}
    assert actual == expected, f"DistillConfig fields drifted: {actual ^ expected}"
