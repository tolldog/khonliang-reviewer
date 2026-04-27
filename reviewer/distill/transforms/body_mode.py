"""Body-mode transform — shape ``summary`` + finding ``body`` length.

Three modes from ``DistillConfig.body_mode``:

- ``full``: identity. Provider output passes through unchanged.
  This is the default — a misconfigured rule never silently
  trims content.
- ``brief``: keep only the first sentence of each finding's
  ``body`` and of the result's ``summary``. Useful for
  ``audience="developer_handoff"`` where the consuming agent
  only needs the headline.
- ``compact``: strip finding bodies entirely (empty string) and
  collapse the result's ``summary`` to a single line. Useful for
  ``audience="github_comment"`` where the user-facing comment
  body is the whole budget.

"First sentence" detection is a deliberately simple regex:
non-greedy match up through the first ``.`` / ``!`` / ``?``
followed by whitespace or end-of-string. Falls back to the
whole text when no terminator is present (so titles like
"TODO comment lacks owner" survive intact). DOTALL is on so
multi-paragraph bodies don't fool the non-greedy match.

Identity preservation: the transform returns the same
``ReviewResult`` object when no field would change (default
``full`` mode, ``compact`` against an already-empty payload,
``brief`` against a payload where every sentence is already
single). The pipeline shell's inert-config invariant relies on
this for the default ``DistillConfig()``.
"""

from __future__ import annotations

import re
from dataclasses import replace

from khonliang_reviewer import ReviewFinding, ReviewResult

from reviewer.rules.distill import DistillConfig


_SENTENCE_END_RE = re.compile(r"^(.+?[.!?])(?:\s|$)", re.DOTALL)


def apply_body_mode(result: ReviewResult, config: DistillConfig) -> ReviewResult:
    """Apply the configured body-mode shaping to ``result``.

    ``full`` short-circuits to identity. ``brief`` and ``compact``
    rebuild only when at least one field would actually change —
    a brief pass over already-brief content stays identity-preserving.
    """
    mode = config.body_mode
    if mode == "full":
        return result
    if mode == "brief":
        new_summary = _first_sentence(result.summary)
        new_findings = tuple(_brief_finding(f) for f in result.findings)
    elif mode == "compact":
        new_summary = _first_sentence(result.summary)
        new_findings = tuple(_compact_finding(f) for f in result.findings)
    else:
        # Unknown mode — typed as Literal so callers shouldn't get
        # here, but the bus boundary can deliver wider payloads
        # than the type system enforces.
        raise ValueError(
            f"DistillConfig.body_mode={mode!r} is not a recognized mode; "
            "expected 'full' | 'brief' | 'compact'."
        )

    if new_summary == result.summary and all(
        new is original for new, original in zip(new_findings, result.findings)
    ):
        # Nothing actually changed — preserve identity so the
        # transform composes cleanly with whatever runs next.
        return result
    return replace(result, summary=new_summary, findings=list(new_findings))


def _brief_finding(f: ReviewFinding) -> ReviewFinding:
    new_body = _first_sentence(f.body)
    if new_body == f.body:
        return f
    return replace(f, body=new_body)


def _compact_finding(f: ReviewFinding) -> ReviewFinding:
    if f.body == "":
        return f
    return replace(f, body="")


def _first_sentence(text: str) -> str:
    """Return the first sentence of ``text`` (terminator included).

    Falls back to the whole string when no sentence terminator is
    present, so single-clause findings without punctuation
    ("TODO comment lacks owner") aren't truncated. Returns ``text``
    unchanged if it's empty.
    """
    if not text:
        return text
    m = _SENTENCE_END_RE.match(text)
    if m:
        return m.group(1)
    return text


__all__ = ["apply_body_mode"]
