"""Rule-table policy for choosing ``(backend, model)`` per review.

The rule table is data, not code. Each :class:`Rule` pairs a predicate
over a :class:`PolicyInput` with a :class:`PolicyDecision`. ``decide()``
walks the rules in order and returns the first matching decision.

Inputs combine three signals:

- ``kind`` — review kind (``"pr_diff"`` initially, ``"spec"`` / ``"fr"`` /
  etc. later). Free-form string so new kinds slot in without lib changes.
- ``profile`` — cached repo profile (AST + distill summary) from the
  researcher's knowledge store, or ``None`` when unknown. Shape is
  deliberately loose so profile emitters can evolve without breaking
  the rule table.
- diff size — ``diff_line_count`` + ``diff_file_count``. Derived
  per-review. Together with the profile they drive model + context
  window decisions.

Caller-supplied ``(backend, model)`` always wins; that path is handled
one layer up, before ``decide()`` runs. Rule evaluation is pure — no
I/O, no logging side-effects — so tests can assert the whole decision
tree without setting up fakes.

Default rule ordering biases toward cheap + fast (Ollama/qwen) for the
common small-docs / small-code-diff case, escalating to Claude for large
or complex diffs where Claude's priors earn their keep. Rule table is a
starting point; accumulated usage + cost data (WU5 usage storage)
informs future iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


#: Floor context window sizes, in input tokens. Matches the documented
#: capability bands across ollama + claude at the time of writing.
CTX_SMALL = 16_000
CTX_MEDIUM = 64_000
CTX_LARGE = 200_000


@dataclass(frozen=True)
class PolicyInput:
    """Inputs the rule table consults.

    All fields default to sentinel values so partial inputs work — for
    example, a caller that has only the diff size but no profile can
    still get a reasonable decision.
    """

    kind: str = "pr_diff"
    diff_line_count: int = 0
    diff_file_count: int = 0
    profile: dict[str, Any] | None = None


@dataclass(frozen=True)
class PolicyDecision:
    """Recommended ``(backend, model)`` plus the context-window floor.

    ``reason`` is captured so observability (usage records, later
    dashboards) can explain *why* a particular model was chosen for a
    given review — useful when the rule table starts absorbing
    feedback from accumulated cost data.
    """

    backend: str
    model: str
    context_window_floor: int
    reason: str


@dataclass(frozen=True)
class Rule:
    """A predicate + decision pair evaluated in order."""

    predicate: Callable[[PolicyInput], bool]
    decision: PolicyDecision
    name: str = ""


# ---------------------------------------------------------------------------
# Default rule table
# ---------------------------------------------------------------------------


def _large_diff(inp: PolicyInput) -> bool:
    """≥2k diff lines or ≥20 files touched — architectural scope."""
    return inp.diff_line_count >= 2000 or inp.diff_file_count >= 20


def _long_context_diff(inp: PolicyInput) -> bool:
    """≥5k diff lines AND ≥10 files — very large refactors."""
    return inp.diff_line_count >= 5000 and inp.diff_file_count >= 10


def _docs_kind(inp: PolicyInput) -> bool:
    """Kinds that are likely short / free-form text (spec, doc, pr_description)."""
    return inp.kind in {"spec", "doc", "pr_description", "fr"}


DEFAULT_RULES: list[Rule] = [
    # Very large diffs want a long-context model and careful priors.
    Rule(
        name="long_context_diff_to_kimi",
        predicate=_long_context_diff,
        decision=PolicyDecision(
            backend="ollama",
            model="kimi-k2.5:cloud",
            context_window_floor=CTX_LARGE,
            reason=">=5000 diff lines across >=10 files — long-context model",
        ),
    ),
    # Architectural-scope diffs: lean on Claude's priors for correctness.
    Rule(
        name="large_diff_to_claude",
        predicate=_large_diff,
        decision=PolicyDecision(
            backend="claude_cli",
            model="claude",
            context_window_floor=CTX_MEDIUM,
            reason=">=2000 diff lines or >=20 files — architectural review",
        ),
    ),
    # Docs / spec / FR reviews: short, cheap, qwen is fine.
    Rule(
        name="docs_kind_to_qwen_small",
        predicate=_docs_kind,
        decision=PolicyDecision(
            backend="ollama",
            model="qwen3.5",
            context_window_floor=CTX_SMALL,
            reason="text-kind review (spec/doc/fr/pr_description) — qwen3.5 suffices",
        ),
    ),
]


#: Ultimate fallback when nothing in the rules matches — cheapest viable
#: option. Matches the Ollama Copilot-CLI doc's recommended default.
DEFAULT_FALLBACK = PolicyDecision(
    backend="ollama",
    model="qwen3.5",
    context_window_floor=CTX_SMALL,
    reason="default fallback — small code-diff review on local qwen3.5",
)


def decide(
    inp: PolicyInput,
    *,
    rules: list[Rule] | None = None,
    fallback: PolicyDecision = DEFAULT_FALLBACK,
) -> PolicyDecision:
    """Return the matched :class:`PolicyDecision` or the fallback.

    Rules are evaluated in order; first match wins. ``rules=None`` uses
    :data:`DEFAULT_RULES`. Pass an explicit ``[]`` to force the fallback
    for every input (useful in tests that isolate fallback behavior).
    """
    active_rules = DEFAULT_RULES if rules is None else rules
    for rule in active_rules:
        try:
            if rule.predicate(inp):
                return rule.decision
        except Exception:
            # A broken predicate must not crash the review pipeline; skip
            # and continue. The fallback still applies if nothing matches.
            continue
    return fallback


__all__ = [
    "CTX_LARGE",
    "CTX_MEDIUM",
    "CTX_SMALL",
    "DEFAULT_FALLBACK",
    "DEFAULT_RULES",
    "PolicyDecision",
    "PolicyInput",
    "Rule",
    "decide",
]
