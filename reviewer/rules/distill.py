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

#: Dedup strategy. ``semantic`` is reserved (raises at pipeline run-time
#: until an embedding-similarity transform lands); the other three are
#: the targets of the first transforms-PR.
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
    dedup: DedupStrategy = "none"
    max_findings: int | None = None
    audience: Audience = "agent_consumption"


__all__ = [
    "Audience",
    "BodyMode",
    "DedupStrategy",
    "DistillConfig",
    "SeverityFloor",
]
