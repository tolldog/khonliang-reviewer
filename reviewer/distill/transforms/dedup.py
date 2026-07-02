"""Dedup transform — collapse near-duplicate findings.

Drops findings that duplicate an earlier finding in the result list,
preserving the highest severity among the merged set on the survivor.
The earlier finding's other fields (``title``, ``body``, ``path``,
``line``, ``suggestion``, ``category``) win — the later duplicate
contributes only its severity if higher.

Strategies, mapped from ``DistillConfig.dedup``:

- ``none``: pass-through. Opt-out for callers that need the raw
  emission stream (e.g. measuring a model's duplicate-emission rate).
- ``exact``: byte-identical repeat — every content field matches
  (``title``, ``body``, ``path``, ``line``, ``section``, ``category``,
  ``suggestion``; only ``severity`` is excluded, see ``_bumped``).
  Catches literal repeats from a model that emitted the same finding
  several times in one response (dog_fa0e1a48 saw 5 byte-identical
  copies of one nit). Two findings that differ in ANY content field —
  same text on different lines, same text with/without a concrete
  suggestion — are distinct observations, not repeats. This is the
  ``DistillConfig.dedup`` default — safe because the collapsed
  duplicates land on ``dropped_findings``, so nothing drops silently.
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
from typing import Callable

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
        merged, dropped = _merge(findings, _is_exact_duplicate)
    elif strategy == "title_substring":
        merged, dropped = _merge(findings, _is_title_substring_duplicate)
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
    return replace(
        result,
        findings=merged,
        # Record collapsed-away duplicates on the running audit trail
        # (fr_reviewer_de1694a8).
        dropped_findings=list(result.dropped_findings) + dropped,
    )


def _merge(
    findings: list[ReviewFinding],
    is_duplicate: Callable[[ReviewFinding, ReviewFinding], bool],
) -> tuple[list[ReviewFinding], list[ReviewFinding]]:
    """Walk findings in order; for each, either keep it (and let
    later duplicates merge into it) or merge it into the earliest
    surviving duplicate. Returns ``(survivors, dropped)`` where
    ``dropped`` are the findings collapsed into a survivor.

    O(n^2) on finding count, which is fine for typical review sizes
    (~1-50 findings); a hashed pre-pass would only matter for the
    ``exact`` strategy at scale, and the dedup transform is not on
    the hot path of any production review.
    """
    survivors: list[ReviewFinding] = []
    dropped: list[ReviewFinding] = []
    for f in findings:
        for i, kept in enumerate(survivors):
            if is_duplicate(kept, f):
                survivors[i] = _bumped(kept, f.severity)
                dropped.append(f)
                break
        else:
            survivors.append(f)
    return survivors, dropped


def _is_exact_duplicate(a: ReviewFinding, b: ReviewFinding) -> bool:
    """Byte-identical repeat: every content field matches.

    Location (``path``, ``line``, ``section``) is part of the key —
    identical terse text anchored to different files/lines is two
    distinct observations. So are ``category`` and ``suggestion``:
    a copy that carries a concrete suggestion block (or a different
    category label) is not a pure repeat, and merging it away would
    drop the suggested fix from GitHub-comment rendering and from
    ``sign_off_trailer``'s actionability check. ``severity`` is the
    one deliberate exclusion — the same text re-emitted at a
    different severity is still one finding, and ``_bumped`` keeps
    the highest severity on the survivor.

    Two findings that are byte-identical in EVERY content field —
    including all-``None`` locations — DO merge, by design. This is
    the dog_fa0e1a48 shape itself (hot-tier models emit unanchored
    descriptive nits, repeated verbatim), so exempting locationless
    findings would un-fix the bug this default exists for. Even when
    a model "meant" two different spots, a second copy with zero
    distinguishing content is not separately actionable by any
    consumer; the repeat count stays auditable on
    ``dropped_findings``.
    """
    return (
        a.title == b.title
        and a.body == b.body
        and a.path == b.path
        and a.line == b.line
        and a.section == b.section
        and a.category == b.category
        and a.suggestion == b.suggestion
    )


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

    Unknown severity strings are tolerated rather than crashing the
    pipeline; severity is a trust-boundary label (provider output,
    skill args), and a malformed value is the provider's bug to fix
    not the dedup transform's data to drop. Resolution per branch:

    - Candidate severity unparseable → keep ``kept`` unchanged. We
      can't reason about the candidate's rank so we don't disturb
      the survivor.
    - Candidate parses, kept's severity is unparseable → bump to
      candidate. The survivor's existing label is malformed; a
      parseable label from a duplicate is unambiguously better
      data, so the survivor inherits it (still the "highest known
      severity in the merged group" contract).
    - Both parse, candidate higher → bump.
    - Both parse, candidate not higher → keep.

    Convention matches the existing severity-floor filter in
    ``reviewer/agent.py:336-338`` which keeps findings with
    unparseable severities (see
    ``test_severity_floor_unknown_severity_in_finding_is_preserved``).
    """
    try:
        candidate_rank = severity_rank(candidate_severity)
    except ValueError:
        return kept
    try:
        kept_rank = severity_rank(kept.severity)
    except ValueError:
        # Survivor's label is malformed but the candidate parses;
        # promote to a known-good severity so the merged group's
        # survivor carries a usable label.
        return replace(kept, severity=candidate_severity)  # type: ignore[arg-type]
    if candidate_rank > kept_rank:
        return replace(kept, severity=candidate_severity)  # type: ignore[arg-type]
    return kept


__all__ = ["apply_dedup"]
