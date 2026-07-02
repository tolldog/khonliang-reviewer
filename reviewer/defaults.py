"""Dependency-free constants shared between reviewer modules.

Lives at the bottom of the reviewer package import graph so any module
(``reviewer.selector``, ``reviewer.rules.policy``, future agent helpers)
can depend on it without forcing a circular import. Add new constants
here only when at least two unrelated modules need them; per-module
defaults stay co-located with their consumer.

PR #40 review pass-4 introduced this split: ``DEFAULT_REVIEWER_MODEL``
previously lived in ``reviewer.selector`` and was imported by
``reviewer.rules.policy``. The selector module's docstring notes that
it will eventually wire in the rule table (i.e. import from
``reviewer.rules.policy``), which would have created a cycle. Hoisting
the constant here keeps both modules importing "downward" only.
"""

from __future__ import annotations

#: Ecosystem-wide default reviewer model. Sourced once from this module
#: so a swap of the constant flips every fallback path in lockstep:
#:
#: - ``reviewer.selector.SelectorConfig.default_model`` (the
#:   ``config.yaml``-omits-``default_model`` floor).
#: - ``reviewer.selector`` re-exports the name for callers that already
#:   import it from there (the selector is the public surface for
#:   provider/model resolution).
#: - ``reviewer.rules.policy.DEFAULT_FALLBACK.model`` (the rule-table
#:   fallback, used when ``decide()`` doesn't match a more specific
#:   rule — i.e. the typical no-override review path).
#:
#: fr_khonliang-reviewer_0e7ccff1 consolidation: the hot tier now rides
#: the box's resident GPU engine (TabbyAPI serving the Qwen3-14B exl3
#: quant) instead of ollama — with TabbyAPI holding the VRAM, the old
#: ollama hot-tier models (deepseek-coder-v2:16b / qwen2.5-coder:14b)
#: run CPU-bound into timeouts (dog_d6895752). The name is the model id
#: TabbyAPI reports for its loaded model; a box serving a different
#: quant overrides ``default_model`` / ``default_models`` in the local
#: ``config.yaml`` (gitignored) rather than editing this constant.
DEFAULT_REVIEWER_MODEL = "Qwen3-14B-exl3-6bpw"

#: Backend paired with :data:`DEFAULT_REVIEWER_MODEL`. Flipped together
#: (a model id only means something to its own backend); every fallback
#: path that reads the model constant reads this one alongside it —
#: ``SelectorConfig.default_backend``, the rule-table hot-tier rows, and
#: ``DEFAULT_FALLBACK``. Cloud-routed rules (e.g. the long-context kimi
#: row) deliberately keep their own backend pins: they never depended on
#: local VRAM, so consolidation doesn't move them.
DEFAULT_REVIEWER_BACKEND = "tabbyapi"


__all__ = ["DEFAULT_REVIEWER_BACKEND", "DEFAULT_REVIEWER_MODEL"]
