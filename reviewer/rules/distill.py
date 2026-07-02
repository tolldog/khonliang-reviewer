"""``DistillConfig`` — rule-table-driven config for the distill pipeline.

Sibling to :mod:`reviewer.rules.policy`. The rule table grows from
emitting :class:`reviewer.rules.policy.PolicyDecision` alone to emitting
the pair ``(PolicyDecision, DistillConfig)`` so callers never assemble
the two halves from separate queries.

This module intentionally ships *only* the dataclass — the transforms
that consume it (dedup, severity_filter, body_mode, max_findings) and
the rule-table evolution that emits it land in follow-up FRs. Carving
the dataclass off first lets each transform-PR depend on a stable
shape rather than chasing a moving target.

See ``specs/MS-B/spec.md`` for the design context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


#: Output-shape audiences that the rule table can route to. The two
#: special-cased values are ``audit_corpus`` (which short-circuits the
#: distill pipeline so audit / benchmark callers always receive raw
#: findings) and ``github_comment`` (which gets the most aggressive
#: shaping; user-facing finding lists need to be terse). The remaining
#: values default to non-aggressive shaping appropriate for agent or
#: human consumption.
Audience = Literal[
    "github_comment",
    "developer_handoff",
    "human_review",
    "agent_consumption",
    "audit_corpus",
]

#: Severity floor values; matches the existing ``ReviewFinding.severity``
#: contract from ``khonliang-reviewer-lib``.
SeverityFloor = Literal["nit", "comment", "concern"]

#: Body-shaping mode; matches researcher's ``detail=`` vocabulary so
#: the audience-shaping vocabulary stays consistent across agents.
BodyMode = Literal["compact", "brief", "full"]

#: Dedup strategy. ``semantic`` is reserved for a future
#: embedding-similarity transform; ``apply_dedup`` raises on it until
#: that transform lands. ``exact`` is the default: hot-tier models
#: sometimes emit the same finding verbatim several times in one
#: response (dog_fa0e1a48 saw 5 byte-identical copies), and a
#: byte-identical repeat at the same location is never signal. The
#: drop is not silent — collapsed duplicates are recorded on
#: ``ReviewResult.dropped_findings``. Callers that need the raw
#: emission stream pass ``dedup="none"`` (or use ``audit_corpus``,
#: which short-circuits the whole pipeline).
DedupStrategy = Literal["none", "exact", "title_substring", "semantic"]


@dataclass(frozen=True)
class DistillConfig:
    """Rule-table-emitted config for the distill pipeline.

    Frozen so a single rule-table evaluation can be safely shared
    across the pipeline without callers mutating fields mid-run. The
    rule table emits a fresh config per ``(kind, profile, size,
    audience)`` row.
    """

    severity_floor: SeverityFloor = "nit"
    body_mode: BodyMode = "full"
    consensus: bool = False
    dedup: DedupStrategy = "exact"
    max_findings: int | None = None
    audience: Audience = "agent_consumption"


# ---------------------------------------------------------------------------
# Audience-keyed distill rule table (fr_reviewer_de1694a8)
# ---------------------------------------------------------------------------
#
# Output shaping keys on the *audience* (who consumes the findings),
# which is orthogonal to provider selection (``reviewer.rules.policy``,
# keyed on diff size). Keeping distill in its own small audience-indexed
# table avoids multiplying the provider rules by audience. ``decide_distill``
# is a pure lookup; an unmapped audience returns the non-aggressive default
# config (carrying only the audience marker). The default config is not a
# strict no-op — it collapses byte-identical duplicate findings
# (``dedup="exact"``, dog_fa0e1a48) — but it never drops unique content.
_DISTILL_BY_AUDIENCE: dict[Audience, DistillConfig] = {
    # User-facing GitHub comment lists must be terse: floor out nits,
    # compact the bodies, cap the count.
    "github_comment": DistillConfig(
        severity_floor="comment",
        body_mode="compact",
        max_findings=10,
        audience="github_comment",
    ),
    # Handoff to an implementing agent: keep every severity but trim
    # bodies to brief so the consuming agent isn't flooded.
    "developer_handoff": DistillConfig(
        body_mode="brief",
        audience="developer_handoff",
    ),
    # Audit / benchmark corpus: raw output. ``run_pipeline`` short-circuits
    # on this audience, so the config is the inert default plus the marker.
    "audit_corpus": DistillConfig(audience="audit_corpus"),
}


def decide_distill(
    audience: Audience = "agent_consumption",
    kind: str = "pr_diff",
) -> DistillConfig:
    """Return the :class:`DistillConfig` for an output ``audience``.

    Distill shaping keys on the *audience* (who consumes the findings),
    independent of the provider rule table (which keys on diff size).
    Unmapped audiences (``agent_consumption``, ``human_review``) get the
    non-aggressive default config carrying just the audience marker.
    Every audience inherits ``dedup="exact"`` from the dataclass default
    (byte-identical repeats are model-emission noise, never signal —
    dog_fa0e1a48); ``audit_corpus`` still sees raw output because
    ``run_pipeline`` short-circuits before any transform runs.

    ``kind`` is accepted for forward-compatibility (a future row may
    shape spec/doc reviews differently) but is not consulted yet.
    """
    mapped = _DISTILL_BY_AUDIENCE.get(audience)
    if mapped is not None:
        return mapped
    return DistillConfig(audience=audience)


__all__ = [
    "Audience",
    "BodyMode",
    "DedupStrategy",
    "DistillConfig",
    "SeverityFloor",
    "decide_distill",
]
