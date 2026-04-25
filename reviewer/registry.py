"""Provider registry — light formalization of the provider map.

Lifts the in-line ``dict[str, ReviewProvider]`` that
:meth:`ReviewerAgent._build_default_selector` used to assemble inline
into a structured object. The registry tracks per-backend metadata
(provider-default model, declared model list from pricing seed,
cheap-availability hints) so the ``list_models`` MCP skill can
enumerate the catalog without re-walking the pricing YAML or
reaching into provider internals.

The registry deliberately does not assume what's available at
provider-construction time. The caller (the agent) decides which
provider classes exist and what default + declared models each
should advertise; the registry just records that and exposes a
read-only view.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from khonliang_reviewer import ReviewProvider


@dataclass(frozen=True)
class ProviderRegistration:
    """Public-facing snapshot of one registered provider.

    ``available`` reflects a **cheap** static probe (binary on PATH,
    auth file present, env var set). It does NOT exercise the model
    or even hit the network. Use :meth:`ReviewProvider.healthcheck`
    when an actual liveness check matters; this field is for
    discovery and for callers that need an at-a-glance "can this
    backend respond at all on this machine right now?" hint.
    """

    backend: str
    default_model: str
    models: tuple[str, ...]
    available: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "default_model": self.default_model,
            "models": list(self.models),
            "available": self.available,
            "reason": self.reason,
        }


class ProviderRegistry:
    """Backend → :class:`ReviewProvider` registry with per-backend metadata.

    Construction is empty; callers register providers explicitly so
    the registry doesn't try to instantiate every possible backend
    eagerly (which would couple it to provider-specific config).

    The registry is **read-mostly**: register providers at agent boot,
    then ``providers`` / ``list`` reads are hot-path.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ReviewProvider] = {}
        self._default_models: dict[str, str] = {}
        self._declared_models: dict[str, list[str]] = {}

    def register(
        self,
        provider: ReviewProvider,
        *,
        default_model: str = "",
        declared_models: Iterable[str] = (),
    ) -> None:
        """Register ``provider`` under :attr:`ReviewProvider.name`.

        ``default_model`` is the model the provider falls back to when
        the caller's request doesn't specify one. ``declared_models``
        is the operator-curated list of models known to work with
        this backend (typically sourced from ``default_pricing.yaml``
        rows). Empty iterables are fine — ``list_models`` then shows
        only the default.
        """
        backend = provider.name
        if not backend:
            raise ValueError(
                "provider.name must be a non-empty string before registration"
            )
        self._providers[backend] = provider
        self._default_models[backend] = default_model
        # Preserve insertion order, dedupe so an operator-curated list
        # that happens to repeat the default doesn't double-count.
        seen: set[str] = set()
        deduped: list[str] = []
        for m in declared_models:
            if m and m not in seen:
                seen.add(m)
                deduped.append(m)
        self._declared_models[backend] = deduped

    @property
    def providers(self) -> Mapping[str, ReviewProvider]:
        """Read-only mapping suitable for :class:`ProviderSelector`."""
        return dict(self._providers)

    def list(
        self, backend: str | None = None
    ) -> list[ProviderRegistration]:
        """Return a snapshot of registered providers.

        When ``backend`` is given, returns the matching registration
        only (or empty list when unknown). Otherwise, every registered
        backend in registration order.
        """
        items: list[ProviderRegistration] = []
        for name, provider in self._providers.items():
            if backend is not None and name != backend:
                continue
            available, reason = _check_availability(name, provider)
            default_model = self._default_models.get(name, "")
            declared = self._declared_models.get(name, [])
            # If the provider's default isn't already in the declared
            # list, surface it so callers see at least one model per
            # backend even when the pricing YAML is silent. Order:
            # default first, then declared (deduped).
            models: list[str] = []
            if default_model:
                models.append(default_model)
            for m in declared:
                if m not in models:
                    models.append(m)
            items.append(
                ProviderRegistration(
                    backend=name,
                    default_model=default_model,
                    models=tuple(models),
                    available=available,
                    reason=reason,
                )
            )
        return items


def _check_availability(
    backend: str, provider: ReviewProvider
) -> tuple[bool, str]:
    """Cheap static probe — no network, no model invocation.

    Returns ``(available, reason)``. ``reason`` is empty when
    available is True; populated with an operator-actionable hint
    when False.

    Unknown backends always report ``True`` — the registry trusts
    that whoever registered the provider knows what they're doing.
    Known backends get tailored probes that match their auth /
    binary requirements.
    """
    config = getattr(provider, "config", None)

    if backend == "claude_cli":
        binary = _config_str(config, "binary", "claude")
        if shutil.which(binary) is None:
            return False, f"claude binary not found at {binary!r}"
        return True, ""

    if backend == "codex_cli":
        binary = _config_str(config, "binary", "codex")
        if shutil.which(binary) is None:
            return False, f"codex binary not found at {binary!r}"
        if not _codex_auth_present():
            return False, (
                "no codex auth (run `codex login` or set OPENAI_API_KEY)"
            )
        return True, ""

    if backend == "ollama":
        # Ollama runs as a long-lived HTTP server; reachability is
        # inherently a network probe and lives on
        # :meth:`OllamaProvider.healthcheck`. For the cheap-probe
        # contract here we assume registered = available.
        return True, ""

    if backend == "gh_copilot":
        # Mirror the codex pattern: binary on PATH plus either a
        # stored OAuth credential or one of the env-var fallbacks.
        # ``copilot login`` writes credentials to either the OS
        # keyring or ``~/.copilot/`` (per `copilot login --help`).
        binary = _config_str(config, "binary", "copilot")
        if shutil.which(binary) is None:
            return False, f"copilot binary not found at {binary!r}"
        if not _gh_copilot_auth_present():
            return False, (
                "no copilot auth (run `copilot login` or set "
                "COPILOT_GITHUB_TOKEN / GH_TOKEN / GITHUB_TOKEN)"
            )
        return True, ""

    return True, ""


def _config_str(config: Any, attr: str, default: str) -> str:
    if config is None:
        return default
    val = getattr(config, attr, default)
    return val if isinstance(val, str) and val else default


def _codex_auth_present() -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return True
    return os.path.exists(os.path.expanduser("~/.codex/auth.json"))


def _gh_copilot_auth_present() -> bool:
    for env in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(env):
            return True
    # ``copilot login`` may store creds in the OS keyring or in
    # ``~/.copilot/`` config files; presence of the directory is a
    # cheap upper bound (false positives possible if the operator
    # ran ``copilot`` interactively without logging in).
    return os.path.isdir(os.path.expanduser("~/.copilot"))


__all__ = ["ProviderRegistration", "ProviderRegistry"]
