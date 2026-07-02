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

# Imported from the dependency-free ``reviewer.defaults`` module rather
# than ``reviewer.selector`` to avoid a future circular import: the
# selector docstring notes it will eventually wire in the rule table
# (i.e. import from ``reviewer.rules.policy``), which would close a
# cycle if this file imported back from selector. Both modules now
# depend "downward" on ``reviewer.defaults``. PR #40 review pass-4
# finding 1.
from reviewer.defaults import DEFAULT_REVIEWER_BACKEND, DEFAULT_REVIEWER_MODEL

#: Model the latency-priority ("fast pre-push gate") rule routes to. Aliased
#: to :data:`DEFAULT_REVIEWER_MODEL` — since the fr_0e7ccff1 consolidation
#: that is the TabbyAPI-resident GPU model (Qwen3-14B exl3), i.e. the fast,
#: VRAM-resident tier by construction. Named separately so the fast-tier
#: choice can diverge from the global default later without touching the rule.
FAST_TIER_MODEL = DEFAULT_REVIEWER_MODEL

# One-way dependency: ``policy`` consults the audience-keyed distill table
# to emit the ``(PolicyDecision, DistillConfig)`` pair from a single call.
# ``reviewer.rules.distill`` never imports back from ``policy`` (its
# ``decide_distill`` keys on a bare ``audience`` string), so there's no
# cycle.
from reviewer.rules.distill import Audience, DistillConfig, decide_distill


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
    #: Output-shape audience (who consumes the findings). Drives the
    #: distill half of :func:`evaluate` via
    #: :func:`reviewer.rules.distill.decide_distill`; ignored by the
    #: provider rules, which key on diff size. Defaults to the
    #: non-aggressive ``agent_consumption``.
    audience: Audience = "agent_consumption"
    #: Caller hint: this is a quick pre-push *gate* where speed matters
    #: more than maximum review rigor. When set (and ``kind == "pr_diff"``)
    #: the :func:`_fast_tier_gate` rule pins the fast, VRAM-resident local
    #: tier (:data:`FAST_TIER_MODEL`) even for large diffs that would
    #: otherwise escalate to Claude. Scoped to ``pr_diff`` so it never
    #: short-circuits the spec/milestone design-rigor routing. Note this
    #: forces the *local* tier but does not by itself make a large review
    #: *fast* — a big diff still gets a large num_ctx window and can be
    #: slow under GPU contention; the timeout→skip degradation is the
    #: backstop there.
    latency_priority: bool = False


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


def _fast_tier_gate(inp: PolicyInput) -> bool:
    """Latency-priority pre-push gate over a code diff.

    Gated to ``kind == "pr_diff"`` deliberately: the pre-push gate is
    always a code diff, and scoping it here keeps ``latency_priority``
    from short-circuiting the spec/milestone design-rigor routing (an
    artifact review must still reach Claude even when a caller passes the
    fast hint).
    """
    return inp.latency_priority and inp.kind == "pr_diff"


def _large_diff(inp: PolicyInput) -> bool:
    """≥2k diff lines or ≥20 files touched — architectural scope."""
    return inp.diff_line_count >= 2000 or inp.diff_file_count >= 20


def _long_context_diff(inp: PolicyInput) -> bool:
    """≥5k diff lines AND ≥10 files — very large refactors."""
    return inp.diff_line_count >= 5000 and inp.diff_file_count >= 10


def _docs_kind(inp: PolicyInput) -> bool:
    """Light text kinds — cheap qwen suffices (doc, pr_description, fr).

    ``fr`` is a pre-design seed: a duplicate/target/dead-dep sanity pass, not
    full design rigor, so the cheap local model is the right tool. ``spec`` and
    ``milestone`` are handled by :func:`_artifact_design_kind` (claude) — full
    design documents that gate code work and reward Claude's priors.
    """
    return inp.kind in {"doc", "pr_description", "fr"}


def _artifact_design_kind(inp: PolicyInput) -> bool:
    """Design-rigor artifact kinds — specs and milestones route to Claude.

    These are full planning documents (a spec is 5K-25K+ tokens; a milestone's
    FR cluster needs holistic coherence judgement) that gate downstream code
    work. Claude's priors earn their keep here, and the CLI handles the large
    context transparently — no diff-size signal applies (artifact reviews carry
    no diff).
    """
    return inp.kind in {"spec", "milestone"}


DEFAULT_RULES: list[Rule] = [
    # Latency-priority pre-push gate: the caller wants a quick local pass,
    # not max rigor, so pin the fast resident tier even for large diffs that
    # would otherwise escalate to Claude/Kimi. First so it short-circuits the
    # diff-size escalations below; scoped to pr_diff inside the predicate so
    # spec/milestone artifact reviews still reach Claude.
    Rule(
        name="fast_tier_gate_to_resident",
        predicate=_fast_tier_gate,
        decision=PolicyDecision(
            backend=DEFAULT_REVIEWER_BACKEND,
            model=FAST_TIER_MODEL,
            context_window_floor=CTX_MEDIUM,
            reason="latency-priority pre-push gate — resident GPU tier",
        ),
    ),
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
    # Design artifacts (specs, milestones): full documents that gate code —
    # route to Claude for design-rigor reasoning over the whole artifact.
    # Placed before the docs/qwen rule so spec/milestone win; the diff-size
    # rules above don't fire for artifact reviews (no diff).
    Rule(
        name="artifact_design_kind_to_claude",
        predicate=_artifact_design_kind,
        decision=PolicyDecision(
            backend="claude_cli",
            model="claude",
            context_window_floor=CTX_LARGE,
            reason="design artifact (spec/milestone) — full-document review on claude",
        ),
    ),
    # Light text kinds (doc / pr_description / fr): short, cheap, qwen is fine.
    # Light text kinds ride the same resident model as the fast tier since
    # the fr_0e7ccff1 consolidation: the old qwen2.5-coder:14b pin was a
    # cheap-local-model carve-out, and the resident GPU model IS the cheap
    # local tier now (an evicted ollama model would be the slow path).
    Rule(
        name="docs_kind_to_resident_small",
        predicate=_docs_kind,
        decision=PolicyDecision(
            backend=DEFAULT_REVIEWER_BACKEND,
            model=FAST_TIER_MODEL,
            context_window_floor=CTX_SMALL,
            reason="text-kind review (doc/fr/pr_description) — resident GPU tier suffices",
        ),
    ),
]


#: Ultimate fallback when nothing in the rules matches. Sourced from the
#: ecosystem-wide ``DEFAULT_REVIEWER_MODEL`` constant in
#: ``reviewer/defaults.py`` (re-exported from ``reviewer/selector.py``
#: for backward compat) so a single edit there shifts both the
#: SelectorConfig fallback AND this rule-table fallback. The
#: ``docs_kind_to_resident_small`` rule above routes short-text reviews
#: to the same resident tier at a smaller context floor — a latency
#: carve-out, not a "default" in the sense this constant captures.
DEFAULT_FALLBACK = PolicyDecision(
    backend=DEFAULT_REVIEWER_BACKEND,
    model=DEFAULT_REVIEWER_MODEL,
    context_window_floor=CTX_SMALL,
    reason=f"default fallback — small code-diff review on {DEFAULT_REVIEWER_MODEL}",
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


def evaluate(
    inp: PolicyInput,
    *,
    rules: list[Rule] | None = None,
    fallback: PolicyDecision = DEFAULT_FALLBACK,
) -> tuple[PolicyDecision, DistillConfig]:
    """Single-call rule evaluation → ``(PolicyDecision, DistillConfig)``.

    Closes ``fr_reviewer_de1694a8``'s connective-tissue goal: callers get
    both halves of a review's config from one query instead of assembling
    the provider decision and the distill config from separate surfaces.
    The two halves key on orthogonal signals — provider on diff size
    (:func:`decide`), distill on audience
    (:func:`reviewer.rules.distill.decide_distill`) — so they're composed
    here rather than crammed into one combinatorial rule table.

    The returned :class:`DistillConfig` is the rule-table baseline for the
    input's audience; a caller that explicitly pins ``severity_floor`` (or
    other fields) layers that on top afterward via ``dataclasses.replace``.
    """
    return (
        decide(inp, rules=rules, fallback=fallback),
        decide_distill(inp.audience, inp.kind),
    )


__all__ = [
    "CTX_LARGE",
    "CTX_MEDIUM",
    "CTX_SMALL",
    "DEFAULT_FALLBACK",
    "DEFAULT_RULES",
    "FAST_TIER_MODEL",
    "PolicyDecision",
    "PolicyInput",
    "Rule",
    "decide",
    "evaluate",
]
