"""Tests for :mod:`reviewer.storage` (UsageStore + pricing table + back-fill)."""

from __future__ import annotations

from khonliang_reviewer import ModelPricing, UsageEvent

from reviewer.storage import UsageStore, open_usage_store


def _open_store() -> UsageStore:
    return open_usage_store(":memory:")


def _event(
    *,
    backend: str = "ollama",
    model: str = "qwen3.5",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    estimated_api_cost_usd: float = 0.0,
    timestamp: float = 1000.0,
    disposition: str = "posted",
    repo: str = "",
    pr_number: int | None = None,
) -> UsageEvent:
    return UsageEvent(
        timestamp=timestamp,
        backend=backend,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        duration_ms=42,
        disposition=disposition,  # type: ignore[arg-type]
        repo=repo,
        pr_number=pr_number,
        estimated_api_cost_usd=estimated_api_cost_usd,
    )


# ---------------------------------------------------------------------------
# schema + migrations
# ---------------------------------------------------------------------------


def test_open_is_idempotent_across_reopens(tmp_path):
    """Two successive opens against the same file must not fail or drop data."""
    db = tmp_path / "reviewer.db"
    store1 = open_usage_store(str(db))
    store1.write_usage(_event())
    store1.close()
    # Re-open the same file — schema-apply must be idempotent and
    # existing rows must survive.
    store2 = open_usage_store(str(db))
    try:
        summaries = store2.summarize()
        assert summaries[0].rows == 1
    finally:
        store2.close()


def test_open_creates_parent_dir(tmp_path):
    """open_usage_store should mkdir -p the parent so `data/reviewer.db` just works."""
    db = tmp_path / "nested" / "reviewer.db"
    store = open_usage_store(str(db))
    assert db.exists()
    store.close()


def test_open_expands_tilde_consistently(tmp_path, monkeypatch):
    """``~/reviewer.db`` must expand to the same path for mkdir and connect.

    Regression test: early code expanded ~ only for the parent mkdir
    and passed the unexpanded path to sqlite3.connect, producing a
    file literally named with ~ in the process cwd.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    db_path = "~/nested/reviewer.db"
    store = open_usage_store(db_path)
    try:
        expanded = tmp_path / "nested" / "reviewer.db"
        assert expanded.exists()
        # No literal ~ file created anywhere under tmp_path
        assert not list(tmp_path.glob("~*"))
    finally:
        store.close()


# ---------------------------------------------------------------------------
# usage writes / reads
# ---------------------------------------------------------------------------


def test_write_usage_round_trips_via_summary():
    store = _open_store()
    store.write_usage(_event(input_tokens=100, output_tokens=50))
    store.write_usage(_event(input_tokens=200, output_tokens=30))

    summaries = store.summarize()
    assert len(summaries) == 1
    s = summaries[0]
    assert s.backend == "ollama"
    assert s.model == "qwen3.5"
    assert s.rows == 2
    assert s.input_tokens == 300
    assert s.output_tokens == 80


def test_summarize_groups_by_backend_model():
    store = _open_store()
    store.write_usage(_event(backend="ollama", model="qwen3.5"))
    store.write_usage(_event(backend="claude_cli", model="claude-opus-4-7"))
    store.write_usage(_event(backend="claude_cli", model="claude-opus-4-7"))

    summaries = store.summarize()
    by_key = {(s.backend, s.model): s for s in summaries}
    assert by_key[("ollama", "qwen3.5")].rows == 1
    assert by_key[("claude_cli", "claude-opus-4-7")].rows == 2


def test_summarize_filters_by_backend():
    store = _open_store()
    store.write_usage(_event(backend="ollama"))
    store.write_usage(_event(backend="claude_cli", model="claude-opus-4-7"))

    summaries = store.summarize(backend="ollama")
    assert len(summaries) == 1
    assert summaries[0].backend == "ollama"


def test_summarize_filters_by_model():
    store = _open_store()
    store.write_usage(_event(model="qwen3.5"))
    store.write_usage(_event(model="kimi-k2.5:cloud"))

    summaries = store.summarize(model="kimi-k2.5:cloud")
    assert len(summaries) == 1
    assert summaries[0].model == "kimi-k2.5:cloud"


def test_summarize_filters_by_time_window():
    store = _open_store()
    store.write_usage(_event(timestamp=100.0))
    store.write_usage(_event(timestamp=200.0))
    store.write_usage(_event(timestamp=300.0))

    window = store.summarize(since=150.0, until=250.0)
    assert len(window) == 1
    assert window[0].rows == 1


def test_summarize_includes_errored_rows():
    """Errored reviews that still burned tokens must count in aggregates."""
    store = _open_store()
    store.write_usage(_event(disposition="posted", input_tokens=100))
    store.write_usage(_event(disposition="errored", input_tokens=50))

    summaries = store.summarize()
    assert summaries[0].rows == 2
    assert summaries[0].input_tokens == 150


# ---------------------------------------------------------------------------
# pricing reads / writes / seed
# ---------------------------------------------------------------------------


def test_get_pricing_returns_none_when_missing():
    store = _open_store()
    assert store.get_pricing("x", "y") is None


def test_put_and_get_pricing_round_trip():
    store = _open_store()
    entry = ModelPricing(
        backend="ollama",
        model="kimi-k2.5:cloud",
        input_per_mtoken_usd=1.5,
        output_per_mtoken_usd=6.0,
        source_url="https://example.com",
        as_of="2026-04-20",
    )
    store.put_pricing(entry)

    loaded = store.get_pricing("ollama", "kimi-k2.5:cloud")
    assert loaded == entry


def test_put_pricing_upserts_on_conflict():
    store = _open_store()
    store.put_pricing(
        ModelPricing(
            backend="ollama", model="qwen3.5", input_per_mtoken_usd=1.0
        )
    )
    store.put_pricing(
        ModelPricing(
            backend="ollama", model="qwen3.5", input_per_mtoken_usd=2.0
        )
    )
    assert store.pricing_count() == 1
    assert store.get_pricing("ollama", "qwen3.5").input_per_mtoken_usd == 2.0


def test_seed_pricing_if_empty_populates_on_empty():
    store = _open_store()
    entries = [
        ModelPricing(backend="ollama", model="qwen3.5"),
        ModelPricing(backend="claude_cli", model="claude-opus-4-7"),
    ]
    inserted = store.seed_pricing_if_empty(entries)
    assert inserted == 2
    assert store.pricing_count() == 2


def test_seed_pricing_dedups_entries_before_insert():
    """Duplicate (backend, model) keys in the seed list collapse to one row.

    Return count reflects distinct keys, not input length. Last-wins on
    duplicates so callers can layer "defaults then overrides" in a single
    list without bespoke merging.
    """
    store = _open_store()
    entries = [
        ModelPricing(backend="ollama", model="qwen3.5", input_per_mtoken_usd=1.0),
        ModelPricing(backend="ollama", model="qwen3.5", input_per_mtoken_usd=2.0),
        ModelPricing(backend="claude_cli", model="claude-opus-4-7"),
    ]
    inserted = store.seed_pricing_if_empty(entries)

    assert inserted == 2
    assert store.pricing_count() == 2
    # last-wins resolution for the duplicate
    assert (
        store.get_pricing("ollama", "qwen3.5").input_per_mtoken_usd == 2.0
    )


def test_seed_pricing_if_empty_is_idempotent_once_populated():
    """Subsequent seeds must not overwrite manual edits."""
    store = _open_store()
    store.put_pricing(
        ModelPricing(
            backend="ollama", model="qwen3.5", input_per_mtoken_usd=99.0
        )
    )
    # Pretend we restart: re-seed with different defaults
    inserted = store.seed_pricing_if_empty(
        [ModelPricing(backend="ollama", model="qwen3.5", input_per_mtoken_usd=1.0)]
    )
    assert inserted == 0
    assert store.get_pricing("ollama", "qwen3.5").input_per_mtoken_usd == 99.0


# ---------------------------------------------------------------------------
# back_fill_cost
# ---------------------------------------------------------------------------


def test_back_fill_cost_leaves_nonzero_cost_alone():
    """Claude envelopes already carry total_cost_usd; don't overwrite."""
    store = _open_store()
    store.put_pricing(
        ModelPricing(
            backend="claude_cli",
            model="claude-opus-4-7",
            input_per_mtoken_usd=15.0,
            output_per_mtoken_usd=75.0,
        )
    )
    event = _event(
        backend="claude_cli",
        model="claude-opus-4-7",
        estimated_api_cost_usd=0.12345,
    )
    filled = store.back_fill_cost(event)
    assert filled.estimated_api_cost_usd == 0.12345


def test_back_fill_cost_computes_for_zero_cost_event():
    """Ollama events land at cost=0; pricing table fills them in."""
    store = _open_store()
    store.put_pricing(
        ModelPricing(
            backend="ollama",
            model="kimi-k2.5:cloud",
            input_per_mtoken_usd=1.0,
            output_per_mtoken_usd=4.0,
        )
    )
    event = _event(
        backend="ollama",
        model="kimi-k2.5:cloud",
        input_tokens=1_000_000,
        output_tokens=250_000,
        estimated_api_cost_usd=0.0,
    )
    filled = store.back_fill_cost(event)
    # 1M * $1 + 250k * $4 = 1.0 + 1.0 = 2.0
    assert filled.estimated_api_cost_usd == 2.0


def test_back_fill_cost_returns_event_unchanged_when_no_pricing():
    """Missing pricing row => leave cost at 0, don't raise."""
    store = _open_store()
    event = _event(estimated_api_cost_usd=0.0)
    filled = store.back_fill_cost(event)
    assert filled.estimated_api_cost_usd == 0.0


def test_back_fill_cost_preserves_other_fields():
    store = _open_store()
    store.put_pricing(
        ModelPricing(
            backend="ollama",
            model="qwen3.5",
            input_per_mtoken_usd=1.0,
            output_per_mtoken_usd=1.0,
        )
    )
    event = _event(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        repo="tolldog/x",
        pr_number=42,
        disposition="posted",
    )
    filled = store.back_fill_cost(event)
    assert filled.repo == "tolldog/x"
    assert filled.pr_number == 42
    assert filled.input_tokens == 1_000_000
    assert filled.disposition == "posted"
    assert filled.estimated_api_cost_usd == 2.0
