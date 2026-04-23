"""Reviewer configuration surface.

Three layers live under this namespace:

- ``reviewer.config.repo`` — repo-side ``.reviewer/`` directory loader.
  Read from base-branch HEAD via ``git show``; never from the PR
  branch tip. Callers feed resolved config into checks + the LLM
  prompt.
- ``reviewer.config.prompts`` — repo-side ``.reviewer/prompts/``
  loader (severity rubric, few-shot examples, system preamble). Same
  trust boundary as ``reviewer.config.repo``; merged into the LLM
  prompt at assembly time.
- The agent-side YAML config (bus URL, default provider, credentials)
  is loaded inline by :mod:`reviewer.agent`. Keep the three axes
  separate — they have different ownership, different trust models,
  and different failure semantics.
"""

from reviewer.config.prompts import (
    RepoPrompts,
    load_repo_prompts,
)
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
    "RepoPrompts",
    "ResolvedConfig",
    "load",
    "load_repo_prompts",
]
