"""Reviewer configuration surface.

Two layers live under this namespace:

- ``reviewer.config.repo`` — repo-side ``.reviewer/`` directory loader
  (this FR). Read from base-branch HEAD via ``git show``; never from
  the PR branch tip. Callers feed resolved config into checks + the
  LLM prompt.
- The agent-side YAML config (bus URL, default provider, credentials)
  is loaded inline by :mod:`reviewer.agent`. Keep the two axes
  separate — they have different ownership, different trust models,
  and different failure semantics.
"""

from reviewer.config.repo import (
    BUILTIN_DEFAULTS,
    RepoConfig,
    RepoConfigUnreachableError,
    ResolvedConfig,
    load,
)

__all__ = [
    "BUILTIN_DEFAULTS",
    "RepoConfig",
    "RepoConfigUnreachableError",
    "ResolvedConfig",
    "load",
]
