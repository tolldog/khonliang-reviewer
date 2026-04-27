"""Severity-filter transform — drop findings below the configured floor.

Lifts the existing severity-floor logic from
``reviewer.agent._filter_findings_by_floor`` (which operates on raw
``dict`` findings on the agent's post-call path) into the distill
pipeline (which operates on ``ReviewFinding`` dataclasses on the
post-provider path). Same noise-reduction contract:

- A finding is kept when its severity rank is ``>= rank(floor)``.
- Findings whose severity doesn't parse (corrupt provider output,
  unknown severity string) are **kept** — the filter's contract is
  noise reduction, not correctness enforcement. Discarding an
  unparseable finding would silently hide real signal; a malformed
  label is the provider's bug to fix, not the filter's data to drop.
- The default ``severity_floor="nit"`` is inert: every known severity
  rank is ``>= rank("nit")``, so no findings are dropped and the
  transform returns identity (preserving the pipeline's inert-config
  contract).

Once the rule-table → DistillConfig evolution lands, the agent's
post-call severity-floor application path — which uses
``reviewer.agent._filter_findings_by_floor`` before writing the
final result — collapses into this single pipeline step. Until
then both paths coexist; this transform runs against the
``DistillConfig.severity_floor`` field while the agent's path
runs against the legacy ``severity_floor`` skill arg.
"""

from __future__ import annotations

from dataclasses import replace

from khonliang_reviewer import ReviewResult, severity_rank

from reviewer.rules.distill import DistillConfig


def apply_severity_filter(
    result: ReviewResult, config: DistillConfig
) -> ReviewResult:
    """Drop findings below ``config.severity_floor`` from ``result``.

    Returns the same ``ReviewResult`` object when no findings would be
    dropped (so identity-equality holds for the inert default
    ``severity_floor="nit"``, and for any payload where every finding
    is already at or above the floor). Returns a new ``ReviewResult``
    with the kept findings only when the filter actually removes
    something.
    """
    floor_rank = severity_rank(config.severity_floor)
    kept = []
    dropped_any = False
    for finding in result.findings:
        severity = finding.severity
        if not isinstance(severity, str):
            # Non-string severity (impossible under the type contract,
            # but the bus boundary can deliver wider payloads) → keep.
            kept.append(finding)
            continue
        try:
            rank = severity_rank(severity)
        except ValueError:
            # Unknown severity string: keep. Same convention as the
            # agent's post-call filter.
            kept.append(finding)
            continue
        if rank >= floor_rank:
            kept.append(finding)
        else:
            dropped_any = True
    if not dropped_any:
        return result
    return replace(result, findings=kept)


__all__ = ["apply_severity_filter"]
