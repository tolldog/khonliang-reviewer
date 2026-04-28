"""Severity-filter transform — drop findings below the configured floor.

Lifts the legacy severity-floor logic that previously lived in
``reviewer.agent._apply_severity_floor`` into the distill pipeline.
Same noise-reduction contract plus the two side effects the legacy
path also performed:

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

When findings ARE dropped:

- ``result.summary`` is rewritten to strip references to dropped
  finding titles in three recognized line shapes (bullet list,
  ``title:`` prose lead, standalone title line). Other prose is
  preserved verbatim — the rewrite is collateral-damage-aware.
- ``result.usage.findings_filtered_count`` is bumped to the number
  of dropped findings so the SQLite usage row reflects how much
  noise was filtered. (The wire shape omits the field when zero;
  the in-process update only happens on the drop path so the inert
  identity invariant holds.)
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from khonliang_reviewer import ReviewFinding, ReviewResult, severity_rank

from reviewer.rules.distill import DistillConfig


def apply_severity_filter(
    result: ReviewResult, config: DistillConfig
) -> ReviewResult:
    """Drop findings below ``config.severity_floor`` from ``result``.

    Returns the same ``ReviewResult`` object when no findings would be
    dropped (so identity-equality holds for the inert default
    ``severity_floor="nit"``, and for any payload where every finding
    is already at or above the floor). Returns a new ``ReviewResult``
    with the kept findings + stripped summary + bumped
    ``usage.findings_filtered_count`` when the filter actually
    removes something.
    """
    floor_rank = severity_rank(config.severity_floor)
    kept: list[ReviewFinding] = []
    dropped: list[ReviewFinding] = []
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
            # agent's legacy post-call filter.
            kept.append(finding)
            continue
        if rank >= floor_rank:
            kept.append(finding)
        else:
            dropped.append(finding)
    if not dropped:
        return result

    new_summary = _strip_dropped_from_summary(
        result.summary,
        [f.to_dict() for f in dropped],
    )
    new_usage = result.usage
    if new_usage is not None:
        new_usage = replace(new_usage, findings_filtered_count=len(dropped))
    return replace(
        result,
        findings=kept,
        summary=new_summary,
        usage=new_usage,
    )


def _strip_dropped_from_summary(
    summary: str, dropped: list[dict[str, Any]]
) -> str:
    """Remove references to dropped findings from the ``summary`` prose.

    Some providers enumerate findings inside the summary itself
    (markdown bullet list, numbered section). If a dropped finding's
    title appears on a line **in one of three recognized shapes**,
    strip that line so the remaining prose reads cleanly. Lines that
    don't match any of the three shapes pass through untouched —
    even if the title happens to appear mid-sentence as prose.

    The three strip-eligible shapes (per line, after ``line.strip()``):

    1. **Bullet list item** — ``^[-*+]\\s+<title>(?=\\s|[:.,;)!?]|$)``
       (e.g. ``"- race condition: ..."`` or ``"- foo(): ..."``).
    2. **Title-colon prose** — ``^<title>\\s*:`` (e.g.
       ``"Missing docstring: ..."``).
    3. **Standalone title line** — ``^<title>\\s*$`` (just the title,
       nothing else).

    Mid-word collisions (dropped title ``"race"`` vs summary word
    ``"embrace"``) are prevented by start-of-line anchoring + exact
    escaped title, not by ``\\b``. The bullet shape's
    ``(?=\\s|[:.,;)!?]|$)`` lookahead catches titles ending in
    non-word chars (e.g. ``"foo()"``) that a bare ``\\b`` would skip.

    **Ultra-short titles (``len(title.strip()) < 3``) are not
    strip-eligible.** Single-letter / two-character titles (``"a"``,
    ``"if"``) would match every indefinite article or conjunction in
    prose. Skipping them preserves a slightly-noisier summary in
    exchange for not shredding unrelated lines.
    """
    if not summary or not dropped:
        return summary
    drop_titles = [
        str(f.get("title") or "").strip()
        for f in dropped
        if isinstance(f, dict)
    ]
    drop_titles = [t for t in drop_titles if len(t) >= 3]
    if not drop_titles:
        return summary
    drop_patterns = [
        re.compile(
            rf"(?:^[-*+]\s+{re.escape(title)}(?=\s|[:.,;)!?]|$))"
            rf"|(?:^{re.escape(title)}\s*:)"
            rf"|(?:^{re.escape(title)}\s*$)"
        )
        for title in drop_titles
    ]
    kept_lines: list[str] = []
    for line in summary.splitlines():
        stripped = line.strip()
        if stripped and any(p.search(stripped) for p in drop_patterns):
            continue
        kept_lines.append(line)
    # Collapse 2+ consecutive blank lines (from stripped content)
    # down to one so the output isn't visually gap-ridden.
    collapsed: list[str] = []
    blank_run = 0
    for line in kept_lines:
        if not line.strip():
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        collapsed.append(line)
    return "\n".join(collapsed).rstrip()


__all__ = ["apply_severity_filter"]
