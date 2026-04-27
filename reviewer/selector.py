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

from dataclasses import dataclass, field
from typing import Mapping

from khonliang_reviewer import ReviewProvider


class UnknownBackendError(ValueError):
    """Raised when the caller or config references an unregistered backend."""


@dataclass
class SelectorConfig:
    """Defaults used when no caller override is present.

    Two layers of default-model state coexist:

    - ``default_models`` (preferred): per-backend dict of model ids,
      so each backend ships its own sensible default. Resolution
      treats empty-string entries as **unset at the selector layer**,
      not as "select the empty string as the model id". That unset
      state can still fall through to the legacy ``default_model``
      when the chosen backend matches ``default_backend``; otherwise
      selection continues on to the empty-string sentinel that
      tells providers to apply their own
      ``ProviderConfig.default_model``. See
      :meth:`ProviderSelector._resolve_default_model` for the
      ordered branches.
    - ``default_model`` (legacy): single string treated as paired
      with ``default_backend``. Kept for backward-compat — operators
      with an existing ``config.yaml`` that only sets ``default_model``
      keep working without changes. The legacy field's value is
      consulted only when ``default_models`` does not carry a
      non-empty entry for the chosen backend AND that chosen backend
      matches ``default_backend``.

    See ``specs/MS-D/spec.md`` for the design context.
    """

    default_backend: str = "ollama"
    default_model: str = "qwen2.5-coder:14b"
    default_models: dict[str, str] = field(default_factory=dict)


class ProviderSelector:
    """Resolve ``(backend, model)`` to a :class:`ReviewProvider` + model string.

    The selector does not construct providers — it receives an already-
    populated mapping from backend name to provider instance (the agent
    builds that on boot from its config). This keeps the selector
    transport-agnostic: tests inject fakes, agent boot injects real
    Claude / Ollama / Codex / gh_copilot instances.

    Resolution order for ``model`` when the caller doesn't supply
    one (``model=None``):

    1. Caller-supplied non-empty ``model`` — always wins (handled
       above this list).
    2. Non-empty ``SelectorConfig.default_models[chosen_backend]`` —
       per-backend default. Empty-string entries are treated as
       "unset at the selector layer" and **do not** stop resolution;
       the lookup falls through to step 3.
    3. ``SelectorConfig.default_model`` (legacy global default) when
       and only when the chosen backend matches
       ``SelectorConfig.default_backend``. The legacy field is
       paired with the default backend; applying it to a different
       backend would send (e.g.) an Ollama-shaped model id into the
       Claude binary.
    4. Empty string — caller chose a non-default backend without
       specifying a model AND no per-backend default is registered.
       Empty string is the "let the provider apply its own default"
       sentinel that providers honor (see each provider's
       ``_resolve_model`` truthy check).

    Step 4's empty-string sentinel is the current bus contract;
    step 5 of MS-D's spec ("agent omits the model arg entirely so
    the binary picks") requires a downstream agent + provider
    change that has not landed yet. Once that lands, step 4 grows
    a fifth tier (``None`` instead of ``""``); for now the
    selector preserves the existing two-state contract.
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
        # ``model=""`` would unexpectedly inherit
        # ``config.default_model`` when the chosen backend matched
        # ``config.default_backend``.
        if model is not None:
            chosen_model = model
        else:
            chosen_model = self._resolve_default_model(chosen_backend)
        provider = self._providers.get(chosen_backend)
        if provider is None:
            raise UnknownBackendError(
                f"unknown backend {chosen_backend!r}; "
                f"available: {sorted(self._providers)}"
            )
        return provider, chosen_model

    def _resolve_default_model(self, chosen_backend: str) -> str:
        """Apply the default-model resolution order.

        Pulled out of :meth:`select` so the precedence reads as a
        single block: per-backend dict first (treating empty as
        unset), then legacy global default paired with default_backend,
        then empty-string sentinel for "let the provider decide".
        """
        per_backend = self.config.default_models.get(chosen_backend, "")
        if per_backend:
            return per_backend
        if chosen_backend == self.config.default_backend:
            return self.config.default_model
        # Caller switched backends without specifying a model AND
        # no per-backend default is registered; the global
        # ``default_model`` is paired with ``default_backend`` and
        # would not fit here (e.g. an Ollama model spec leaking
        # into a Codex / Claude provider). Empty string lets the
        # provider apply its own default.
        return ""


__all__ = [
    "ProviderSelector",
    "SelectorConfig",
    "UnknownBackendError",
]
