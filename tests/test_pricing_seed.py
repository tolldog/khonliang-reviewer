"""Tests for the bundled pricing-seed loader."""

from __future__ import annotations

from reviewer.pricing_seed import load_default_pricing


def test_load_default_pricing_returns_non_empty():
    """The bundled YAML must load cleanly and produce at least one entry."""
    entries = load_default_pricing()
    assert len(entries) > 0


def test_bundled_pricing_covers_both_first_class_backends():
    """At least one ollama and one claude_cli entry must be present."""
    entries = load_default_pricing()
    backends = {e.backend for e in entries}
    assert "ollama" in backends
    assert "claude_cli" in backends


def test_bundled_pricing_has_non_negative_rates():
    """Sanity: published rates are never negative."""
    for entry in load_default_pricing():
        assert entry.input_per_mtoken_usd >= 0.0
        assert entry.output_per_mtoken_usd >= 0.0
        assert entry.cache_read_per_mtoken_usd >= 0.0
        assert entry.cache_creation_per_mtoken_usd >= 0.0


def test_bundled_pricing_carries_source_and_as_of():
    """Every entry should name where it came from + when."""
    for entry in load_default_pricing():
        assert entry.source_url != ""
        assert entry.as_of != ""
