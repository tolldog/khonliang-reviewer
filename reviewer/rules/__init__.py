"""Rule-table policy + repo-profile cache for provider selection.

Two small modules:

- :mod:`reviewer.rules.policy` — pure data-driven rule table that maps
  ``(kind, profile, diff_size)`` to ``(backend, model)`` recommendations.
- :mod:`reviewer.rules.profile` — profile cache protocol + an in-memory
  implementation. Bus-backed implementation lives in the agent layer
  (WU5 wires ``researcher.knowledge_search`` for reads).
"""

from reviewer.rules.policy import (
    CTX_LARGE,
    CTX_MEDIUM,
    CTX_SMALL,
    DEFAULT_FALLBACK,
    DEFAULT_RULES,
    PolicyDecision,
    PolicyInput,
    Rule,
    decide,
)
from reviewer.rules.profile import (
    PROFILE_KEY_PREFIX,
    InMemoryProfileCache,
    ProfileCache,
    profile_key,
)


__all__ = [
    "CTX_LARGE",
    "CTX_MEDIUM",
    "CTX_SMALL",
    "DEFAULT_FALLBACK",
    "DEFAULT_RULES",
    "InMemoryProfileCache",
    "PROFILE_KEY_PREFIX",
    "PolicyDecision",
    "PolicyInput",
    "ProfileCache",
    "Rule",
    "decide",
    "profile_key",
]
