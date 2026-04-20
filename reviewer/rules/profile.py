"""Repo-profile cache client.

A profile is the researcher's AST + distill output for a repo,
serialized as a dict. The reviewer needs it to feed the rule table
(:mod:`reviewer.rules.policy`), which looks at shape signals like
language breakdown and repo size — not just per-review diff statistics.

Profiles live in researcher's knowledge store under a stable key
``repo_profile:<canonical_name>``. Reads are bus-mediated in production
(``researcher.knowledge_search``); this module defines the client-side
contract the provider layer calls through, plus a simple in-memory
implementation used by tests and agent-startup seeding.

The protocol is intentionally narrow: ``get_profile(repo)`` returns the
profile dict or ``None``. Invalidation (SHA drift / age) is a detail of
the backing implementation, not the contract. The helper
:func:`profile_key` canonicalizes names so cache lookups stay
deterministic when callers pass ``"tolldog/khonliang-reviewer"`` vs
``"khonliang-reviewer"``.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


PROFILE_KEY_PREFIX = "repo_profile:"


def profile_key(repo: str) -> str:
    """Return the canonical knowledge-store key for ``repo``.

    Accepts both ``"owner/name"`` and bare ``"name"`` forms and
    normalizes to lowercase. Anything outside a leading/trailing whitespace
    trim is left untouched so callers can pass arbitrary short names
    that researcher's ``register_repo`` uses internally.
    """
    normalized = repo.strip().lower()
    return f"{PROFILE_KEY_PREFIX}{normalized}"


@runtime_checkable
class ProfileCache(Protocol):
    """Read-only access to cached repo profiles.

    Concrete implementations live in the agent layer (WU5 wires a
    bus-backed cache that queries ``researcher.knowledge_search``).
    This module ships the in-memory implementation for tests and for
    agent-startup seeding before the bus client is available.
    """

    async def get_profile(self, repo: str) -> dict[str, Any] | None:
        ...


@dataclass
class InMemoryProfileCache:
    """Simple dict-backed :class:`ProfileCache` for tests + warm-up.

    Records an insertion timestamp on each entry so callers can
    observe age. SHA-based invalidation is the backing store's job —
    this implementation is intentionally naive; it never evicts unless
    :meth:`invalidate` is called explicitly.
    """

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    _inserted_at: dict[str, float] = field(default_factory=dict)

    async def get_profile(self, repo: str) -> dict[str, Any] | None:
        key = profile_key(repo)
        # Return a copy so callers can't mutate stored state.
        stored = self.entries.get(key)
        return deepcopy(stored) if stored is not None else None

    def put_profile(self, repo: str, profile: dict[str, Any]) -> None:
        key = profile_key(repo)
        self.entries[key] = deepcopy(profile)
        self._inserted_at[key] = time.time()

    def invalidate(self, repo: str) -> None:
        key = profile_key(repo)
        self.entries.pop(key, None)
        self._inserted_at.pop(key, None)

    def age_seconds(self, repo: str) -> float | None:
        """Seconds since ``repo``'s profile was last stored, or None."""
        key = profile_key(repo)
        inserted = self._inserted_at.get(key)
        if inserted is None:
            return None
        return max(0.0, time.time() - inserted)


__all__ = [
    "InMemoryProfileCache",
    "PROFILE_KEY_PREFIX",
    "ProfileCache",
    "profile_key",
]
