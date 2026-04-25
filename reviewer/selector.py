"""Provider selection for review skills.

Given a (backend, model) hint plus — eventually — review kind, repo
profile, and diff size, pick which :class:`ReviewProvider` instance
handles the request and what model to use.

This module ships with a **stubbed** selector: caller-supplied
backend/model always wins; otherwise the agent's config default
resolves. The rule-table-driven selection primitives live in
:mod:`reviewer.rules.policy` (lands with WU4) and wire in here as a
follow-up once they're available on ``main``. The stubbed shape keeps
the bus skill surface complete and dogfood-able before the full
policy is in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from khonliang_reviewer import ReviewProvider


class UnknownBackendError(ValueError):
    """Raised when the caller or config references an unregistered backend."""


@dataclass
class SelectorConfig:
    """Defaults used when no caller override is present."""

    default_backend: str = "ollama"
    default_model: str = "qwen2.5-coder:14b"


class ProviderSelector:
    """Resolve ``(backend, model)`` to a :class:`ReviewProvider` + model string.

    The selector does not construct providers — it receives an already-
    populated mapping from backend name to provider instance (the agent
    builds that on boot from its config). This keeps the selector
    transport-agnostic: tests inject fakes, agent boot injects real
    Claude / Ollama instances.

    Precedence:

    1. Caller-supplied ``backend`` + ``model`` — always win.
    2. Caller-supplied ``backend`` alone:
       - When the chosen backend matches ``config.default_backend`` the
         caller-omitted model resolves to ``config.default_model``.
       - When the chosen backend differs from the default backend, the
         model resolves to ``""`` so the provider applies its own
         default. ``config.default_model`` is treated as paired with
         ``config.default_backend`` and is not applied to a different
         backend, since the global default model would not fit (e.g.
         ``qwen2.5-coder:14b`` is not a Codex model).
    3. No caller input — both from config default.
    """

    def __init__(
        self,
        providers: Mapping[str, ReviewProvider],
        config: SelectorConfig | None = None,
    ):
        self._providers: dict[str, ReviewProvider] = dict(providers)
        self.config = config or SelectorConfig()

    @property
    def providers(self) -> Mapping[str, ReviewProvider]:
        return self._providers

    def select(
        self,
        *,
        backend: str | None = None,
        model: str | None = None,
    ) -> tuple[ReviewProvider, str]:
        chosen_backend = backend or self.config.default_backend
        # Distinguish ``None`` (caller didn't supply a value, fall
        # through to the default-resolution rules) from ``""``
        # (caller explicitly chose "no model", which means "let the
        # provider apply its own default"). The previous shape used
        # ``if model:`` which collapsed both cases — a caller passing
        # ``model=""`` would unexpectedly inherit ``config.default_model``
        # when the chosen backend matched ``config.default_backend``.
        if model is not None:
            chosen_model = model
        elif chosen_backend == self.config.default_backend:
            chosen_model = self.config.default_model
        else:
            # Caller switched backends without specifying a model; the
            # global ``default_model`` is paired with ``default_backend``
            # and would not fit here (e.g. an Ollama model spec leaking
            # into a Codex / Claude provider). Empty string lets the
            # provider apply its own default.
            chosen_model = ""
        provider = self._providers.get(chosen_backend)
        if provider is None:
            raise UnknownBackendError(
                f"unknown backend {chosen_backend!r}; "
                f"available: {sorted(self._providers)}"
            )
        return provider, chosen_model


__all__ = [
    "ProviderSelector",
    "SelectorConfig",
    "UnknownBackendError",
]
