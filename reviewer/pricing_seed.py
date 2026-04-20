"""Load the bundled default pricing YAML into :class:`ModelPricing` records.

The YAML is source-controlled at
``reviewer/data/default_pricing.yaml``. Operators can edit their live
``model_pricing`` table directly; this loader only touches the DB on
first startup (via :meth:`UsageStore.seed_pricing_if_empty`) so manual
edits survive.
"""

from __future__ import annotations

import logging
from importlib import resources
from typing import Any, Iterable

from khonliang_reviewer import ModelPricing


logger = logging.getLogger(__name__)


def load_default_pricing() -> list[ModelPricing]:
    """Parse the bundled YAML and return :class:`ModelPricing` instances.

    Returns an empty list (with a warning) if PyYAML is missing or the
    bundled resource is unreadable; callers treat that as "nothing to
    seed" and fall through to their normal operation.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("pyyaml missing; cannot seed default pricing")
        return []

    try:
        raw = (
            resources.files("reviewer.data")
            .joinpath("default_pricing.yaml")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        logger.warning("default pricing resource unreadable: %s", exc)
        return []

    try:
        doc = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning("default pricing YAML failed to parse: %s", exc)
        return []

    if not isinstance(doc, dict):
        logger.warning("default pricing YAML root must be a mapping")
        return []

    entries_raw = doc.get("pricing") or []
    if not isinstance(entries_raw, list):
        logger.warning("default pricing 'pricing' must be a list")
        return []

    entries: list[ModelPricing] = []
    for item in entries_raw:
        if not isinstance(item, dict):
            continue
        entries.append(
            ModelPricing(
                backend=str(item.get("backend", "")),
                model=str(item.get("model", "")),
                input_per_mtoken_usd=_as_float(item.get("input_per_mtoken_usd")),
                output_per_mtoken_usd=_as_float(item.get("output_per_mtoken_usd")),
                cache_read_per_mtoken_usd=_as_float(
                    item.get("cache_read_per_mtoken_usd")
                ),
                cache_creation_per_mtoken_usd=_as_float(
                    item.get("cache_creation_per_mtoken_usd")
                ),
                currency=str(item.get("currency", "USD") or "USD"),
                source_url=str(item.get("source_url", "")),
                as_of=str(item.get("as_of", "")),
            )
        )
    return [e for e in entries if e.backend and e.model]


def _as_float(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["load_default_pricing"]
