"""Dedup transform — collapse near-duplicate findings.

Drops findings that duplicate an earlier finding in the result list,
preserving the highest severity among the merged set on the survivor.
The earlier finding's other fields (``title``, ``body``, ``path``,
``line``, ``suggestion``, ``category``) win — the later duplicate
contributes only its severity if higher.

Strategies, mapped from ``DistillConfig.dedup``:

- ``none``: pass-through. Used as the default so a misconfigured rule
  never silently drops findings.
- ``exact``: identical ``(title, body)`` tuple, case-sensitive. Catches
  literal repeats from a model that emitted the same finding twice.
- ``title_substring``: one finding's title appears as a substring of
  the other's title (case-insensitive, stripped). Catches the common
  case where a model emits "Missing test" and "Missing test for
  handler" as two findings for the same underlying concern.
- ``semantic``: reserved for a future embedding-similarity transform.
  Raises :class:`ValueError` until that transform lands so a
  misconfigured rule fails loudly instead of silently degrading to
  ``none``.

Feature-preservation invariant (from the MS-B distill principle):
a unique finding never merges into another, regardless of
strategy. The dedup decision is "is this the same finding as an
earlier one?" — never "is this similar enough to drop?". An outlier
concern surrounded by 20 nits survives every strategy.
"""

from __future__ import annotations

from dataclasses import replace

from khonliang_reviewer import ReviewFinding, ReviewResult, severity_rank

from reviewer.rules.distill import DistillConfig


def apply_dedup(result: ReviewResult, config: DistillConfig) -> ReviewResult:
    """Apply the configured dedup strategy to ``result.findings``.

    Returns the same ``ReviewResult`` object when no findings would be
    merged (so identity-equality holds — the pipeline shell relies on
    this for the ``dedup="none"`` and zero-duplicate cases). Returns a
    new ``ReviewResult`` with shaped findings only when an actual
    merge occurred.
    """
    strategy = config.dedup
    if strategy == "none":
        return result
    if strategy == "semantic":
        raise ValueError(
            "DistillConfig.dedup='semantic' is reserved for a future "
            "embedding-similarity transform; not implemented yet. "
            "Use 'exact', 'title_substring', or 'none'."
        )

    findings = result.findings
    if len(findings) < 2:
        # A solo finding can't duplicate anything; keep identity so the
        # transform composes cleanly with whatever runs next.
        return result

    if strategy == "exact":
        merged = _merge(findings, _is_exact_duplicate)
    elif strategy == "title_substring":
        merged = _merge(findings, _is_title_substring_duplicate)
    else:
        # Unknown strategy — typed as Literal so callers shouldn't get
        # here, but at the bus boundary this defends against a wider
        # config payload than the type system enforces.
        raise ValueError(
            f"DistillConfig.dedup={strategy!r} is not a recognized strategy; "
            "expected 'none' | 'exact' | 'title_substring' | 'semantic'."
        )

    if len(merged) == len(findings):
        # No actual merging happened (no pairs matched). Return the
        # original result so identity-equality holds.
        return result
    return replace(result, findings=merged)


def _merge(
    findings: list[ReviewFinding],
    is_duplicate: "callable[[ReviewFinding, ReviewFinding], bool]",  # type: ignore[name-defined]
) -> list[ReviewFinding]:
    """Walk findings in order; for each, either keep it (and let
    later duplicates merge into it) or merge it into the earliest
    surviving duplicate.

    O(n^2) on finding count, which is fine for typical review sizes
    (~1-50 findings); a hashed pre-pass would only matter for the
    ``exact`` strategy at scale, and the dedup transform is not on
    the hot path of any production review.
    """
    survivors: list[ReviewFinding] = []
    for f in findings:
        for i, kept in enumerate(survivors):
            if is_duplicate(kept, f):
                survivors[i] = _bumped(kept, f.severity)
                break
        else:
            survivors.append(f)
    return survivors


def _is_exact_duplicate(a: ReviewFinding, b: ReviewFinding) -> bool:
    return a.title == b.title and a.body == b.body


def _is_title_substring_duplicate(a: ReviewFinding, b: ReviewFinding) -> bool:
    """One title contains the other (case-insensitive, stripped).

    Asymmetric inputs collapse to the same answer: ``"X"`` is a
    duplicate of ``"X with extra"`` whether ``a`` or ``b`` carries
    the shorter title. Empty titles never match (an empty string is
    technically a substring of every string, but matching empties
    would aggressively merge every "summary-level" finding into the
    first empty-title row, which is the wrong behavior — empties
    are kept distinct).
    """
    a_title = a.title.strip().casefold()
    b_title = b.title.strip().casefold()
    if not a_title or not b_title:
        return False
    return a_title in b_title or b_title in a_title


def _bumped(kept: ReviewFinding, candidate_severity: str) -> ReviewFinding:
    """Return ``kept`` with its severity bumped if ``candidate_severity``
    outranks the existing one. Otherwise return ``kept`` unchanged
    (identity-preserving in the common case).

    Unknown severity strings on either side leave ``kept`` untouched
    rather than crashing the pipeline. Severity is a trust-boundary
    label (provider output, skill args); a malformed value is the
    provider's bug to fix, not the dedup transform's data to drop.
    The behavior matches the existing severity-floor filter in
    ``reviewer/agent.py`` which keeps findings with unparseable
    severities (see ``test_severity_floor_unknown_severity_in_finding_is_preserved``).
    """
    try:
        candidate_rank = severity_rank(candidate_severity)
        kept_rank = severity_rank(kept.severity)
    except ValueError:
        return kept
    if candidate_rank > kept_rank:
        return replace(kept, severity=candidate_severity)  # type: ignore[arg-type]
    return kept


__all__ = ["apply_dedup"]
