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
DEFAULT_REVIEWER_MODEL = "deepseek-coder-v2:16b"


__all__ = ["DEFAULT_REVIEWER_MODEL"]
