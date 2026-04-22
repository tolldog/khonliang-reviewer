"""SQLite-backed usage + pricing store for the reviewer agent.

Two tables:

- ``reviewer_usage`` — one row per review call (posted, dry-run, or
  errored). Additive schema; new fields land via migrations here rather
  than in a separate tool. Indexed by ``(backend, model, timestamp)``
  for the common ``usage_summary`` query.
- ``model_pricing`` — per-model ``$/Mtoken`` rates keyed by
  ``(backend, model)``. Seeded from a bundled YAML on first startup so
  :data:`UsageEvent.estimated_api_cost_usd` can be back-filled for
  backends that don't return cost in-band (Ollama).

The store is process-local. ``check_same_thread=False`` is used only
so asyncio coroutines scheduled on different threads can call through
the same connection — it does NOT serialize concurrent access or make
the connection safe to use from multiple callers at once. The reviewer
agent drives reviews sequentially per-request from a single event
loop, and writes are infrequent + small, so actual contention is not
expected. If that invariant ever changes (parallel reviews, a
background writer, etc.), add explicit locking or move to a
per-coroutine connection before relying on the current shape.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from khonliang_reviewer import ModelPricing, UsageEvent, estimate_api_cost


logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1


_DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS reviewer_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        backend TEXT NOT NULL,
        model TEXT NOT NULL,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        cache_read_tokens INTEGER NOT NULL DEFAULT 0,
        cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
        duration_ms INTEGER NOT NULL DEFAULT 0,
        disposition TEXT NOT NULL DEFAULT 'posted',
        request_id TEXT NOT NULL DEFAULT '',
        repo TEXT NOT NULL DEFAULT '',
        pr_number INTEGER,
        estimated_api_cost_usd REAL NOT NULL DEFAULT 0.0,
        error TEXT NOT NULL DEFAULT '',
        error_category TEXT NOT NULL DEFAULT '',
        findings_filtered_count INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_reviewer_usage_backend_model_timestamp
    ON reviewer_usage(backend, model, timestamp)
    """,
    """
    CREATE TABLE IF NOT EXISTS model_pricing (
        backend TEXT NOT NULL,
        model TEXT NOT NULL,
        input_per_mtoken_usd REAL NOT NULL DEFAULT 0.0,
        output_per_mtoken_usd REAL NOT NULL DEFAULT 0.0,
        cache_read_per_mtoken_usd REAL NOT NULL DEFAULT 0.0,
        cache_creation_per_mtoken_usd REAL NOT NULL DEFAULT 0.0,
        currency TEXT NOT NULL DEFAULT 'USD',
        source_url TEXT NOT NULL DEFAULT '',
        as_of TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (backend, model)
    )
    """,
)


@dataclass
class UsageSummary:
    """Aggregated totals grouped by ``(backend, model)``."""

    backend: str
    model: str
    rows: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    total_cost_usd: float
    total_duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "rows": self.rows,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_duration_ms": self.total_duration_ms,
        }


class UsageStore:
    """SQLite-backed store for :class:`UsageEvent` rows + pricing rates.

    Open via :func:`open_usage_store`, which also runs migrations. For
    tests pass ``":memory:"`` as the path — every assertion is scoped
    to that connection's lifetime.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def close(self) -> None:
        self._conn.close()

    # -- usage writes / reads -----------------------------------------

    def write_usage(self, event: UsageEvent) -> int:
        """Insert one usage row; return its rowid."""
        cursor = self._conn.execute(
            """
            INSERT INTO reviewer_usage (
                timestamp, backend, model,
                input_tokens, output_tokens,
                cache_read_tokens, cache_creation_tokens,
                duration_ms, disposition,
                request_id, repo, pr_number,
                estimated_api_cost_usd,
                error, error_category,
                findings_filtered_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                float(event.timestamp),
                str(event.backend),
                str(event.model),
                int(event.input_tokens),
                int(event.output_tokens),
                int(event.cache_read_tokens),
                int(event.cache_creation_tokens),
                int(event.duration_ms),
                str(event.disposition),
                str(event.request_id),
                str(event.repo),
                event.pr_number,
                float(event.estimated_api_cost_usd),
                str(event.error),
                str(event.error_category),
                int(event.findings_filtered_count),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def summarize(
        self,
        *,
        backend: str | None = None,
        model: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> list[UsageSummary]:
        """Aggregate totals grouped by ``(backend, model)`` with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if backend:
            clauses.append("backend = ?")
            params.append(backend)
        if model:
            clauses.append("model = ?")
            params.append(model)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(float(since))
        if until is not None:
            clauses.append("timestamp < ?")
            params.append(float(until))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        rows = self._conn.execute(
            f"""
            SELECT backend, model,
                   COUNT(*) AS rows_,
                   COALESCE(SUM(input_tokens), 0),
                   COALESCE(SUM(output_tokens), 0),
                   COALESCE(SUM(cache_read_tokens), 0),
                   COALESCE(SUM(cache_creation_tokens), 0),
                   COALESCE(SUM(estimated_api_cost_usd), 0.0),
                   COALESCE(SUM(duration_ms), 0)
            FROM reviewer_usage
            {where}
            GROUP BY backend, model
            ORDER BY backend, model
            """,
            params,
        ).fetchall()

        return [
            UsageSummary(
                backend=row[0],
                model=row[1],
                rows=int(row[2]),
                input_tokens=int(row[3]),
                output_tokens=int(row[4]),
                cache_read_tokens=int(row[5]),
                cache_creation_tokens=int(row[6]),
                total_cost_usd=float(row[7]),
                total_duration_ms=int(row[8]),
            )
            for row in rows
        ]

    # -- pricing reads / writes ---------------------------------------

    def get_pricing(self, backend: str, model: str) -> ModelPricing | None:
        row = self._conn.execute(
            """
            SELECT backend, model,
                   input_per_mtoken_usd, output_per_mtoken_usd,
                   cache_read_per_mtoken_usd, cache_creation_per_mtoken_usd,
                   currency, source_url, as_of
            FROM model_pricing
            WHERE backend = ? AND model = ?
            """,
            (backend, model),
        ).fetchone()
        if row is None:
            return None
        return ModelPricing(
            backend=row[0],
            model=row[1],
            input_per_mtoken_usd=float(row[2]),
            output_per_mtoken_usd=float(row[3]),
            cache_read_per_mtoken_usd=float(row[4]),
            cache_creation_per_mtoken_usd=float(row[5]),
            currency=row[6],
            source_url=row[7],
            as_of=row[8],
        )

    def put_pricing(self, pricing: ModelPricing) -> None:
        self._conn.execute(
            """
            INSERT INTO model_pricing (
                backend, model,
                input_per_mtoken_usd, output_per_mtoken_usd,
                cache_read_per_mtoken_usd, cache_creation_per_mtoken_usd,
                currency, source_url, as_of
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(backend, model) DO UPDATE SET
                input_per_mtoken_usd = excluded.input_per_mtoken_usd,
                output_per_mtoken_usd = excluded.output_per_mtoken_usd,
                cache_read_per_mtoken_usd = excluded.cache_read_per_mtoken_usd,
                cache_creation_per_mtoken_usd = excluded.cache_creation_per_mtoken_usd,
                currency = excluded.currency,
                source_url = excluded.source_url,
                as_of = excluded.as_of
            """,
            (
                pricing.backend,
                pricing.model,
                float(pricing.input_per_mtoken_usd),
                float(pricing.output_per_mtoken_usd),
                float(pricing.cache_read_per_mtoken_usd),
                float(pricing.cache_creation_per_mtoken_usd),
                pricing.currency,
                pricing.source_url,
                pricing.as_of,
            ),
        )
        self._conn.commit()

    def pricing_count(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM model_pricing").fetchone()[0]
        )

    def seed_pricing_if_empty(self, entries: Iterable[ModelPricing]) -> int:
        """Populate ``model_pricing`` from ``entries`` iff the table is empty.

        Returns the number of distinct rows inserted. Idempotent across
        restarts — subsequent invocations see a non-empty table and
        no-op, so manual edits aren't overwritten on agent boot.

        Entries are de-duplicated by ``(backend, model)`` before insert
        so the return count reflects actual insertions rather than
        ``len(entries)`` (which would over-report when the caller
        supplies duplicate keys that would have collapsed under the
        ``ON CONFLICT`` path).

        All inserts land in a single transaction so agent startup pays
        one fsync regardless of how many rows the seed ships.
        """
        if self.pricing_count() > 0:
            return 0
        dedup: dict[tuple[str, str], ModelPricing] = {}
        for entry in entries:
            # Last-wins on duplicate keys, matching the ON CONFLICT
            # behavior if duplicates were allowed through.
            dedup[(entry.backend, entry.model)] = entry
        rows = [
            (
                entry.backend,
                entry.model,
                float(entry.input_per_mtoken_usd),
                float(entry.output_per_mtoken_usd),
                float(entry.cache_read_per_mtoken_usd),
                float(entry.cache_creation_per_mtoken_usd),
                entry.currency,
                entry.source_url,
                entry.as_of,
            )
            for entry in dedup.values()
        ]
        if not rows:
            return 0
        self._conn.executemany(
            """
            INSERT INTO model_pricing (
                backend, model,
                input_per_mtoken_usd, output_per_mtoken_usd,
                cache_read_per_mtoken_usd, cache_creation_per_mtoken_usd,
                currency, source_url, as_of
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(backend, model) DO UPDATE SET
                input_per_mtoken_usd = excluded.input_per_mtoken_usd,
                output_per_mtoken_usd = excluded.output_per_mtoken_usd,
                cache_read_per_mtoken_usd = excluded.cache_read_per_mtoken_usd,
                cache_creation_per_mtoken_usd = excluded.cache_creation_per_mtoken_usd,
                currency = excluded.currency,
                source_url = excluded.source_url,
                as_of = excluded.as_of
            """,
            rows,
        )
        self._conn.commit()
        return len(rows)

    # -- cost back-fill ------------------------------------------------

    def back_fill_cost(self, event: UsageEvent) -> UsageEvent:
        """Return a copy of ``event`` with ``estimated_api_cost_usd`` filled.

        If the event already carries a non-zero cost (Claude CLI reports
        one directly from its envelope), the event is returned unchanged.
        Otherwise we look up pricing for ``(backend, model)`` and compute
        from the token counts. Missing pricing leaves the cost at 0.0 —
        callers can detect via summarize() + pricing_count() if that
        becomes a monitoring concern.
        """
        if event.estimated_api_cost_usd > 0.0:
            return event
        pricing = self.get_pricing(event.backend, event.model)
        if pricing is None:
            return event
        cost = estimate_api_cost(
            pricing,
            input_tokens=event.input_tokens,
            output_tokens=event.output_tokens,
            cache_read_tokens=event.cache_read_tokens,
            cache_creation_tokens=event.cache_creation_tokens,
        )
        # UsageEvent isn't frozen — but returning a fresh instance keeps
        # callers free of accidental aliasing.
        return UsageEvent(
            timestamp=event.timestamp,
            backend=event.backend,
            model=event.model,
            input_tokens=event.input_tokens,
            output_tokens=event.output_tokens,
            cache_read_tokens=event.cache_read_tokens,
            cache_creation_tokens=event.cache_creation_tokens,
            duration_ms=event.duration_ms,
            disposition=event.disposition,
            request_id=event.request_id,
            repo=event.repo,
            pr_number=event.pr_number,
            estimated_api_cost_usd=cost,
            error=event.error,
            error_category=event.error_category,
            findings_filtered_count=event.findings_filtered_count,
        )


def open_usage_store(db_path: str) -> UsageStore:
    """Open (or create) the SQLite DB at ``db_path`` and run migrations.

    The database file is created on first call if it doesn't exist
    (this is sqlite3.connect() default behavior). The parent directory
    is auto-created so defaults like ``data/reviewer.db`` or
    ``~/reviewer.db`` work without a preparatory ``mkdir``.

    ``:memory:`` bypasses filesystem handling — useful for tests. Any
    other path is resolved through :meth:`~pathlib.Path.expanduser` so
    ``~`` shorthand reaches the same file on both the mkdir and the
    :func:`sqlite3.connect` call.
    """
    if db_path == ":memory:":
        resolved = ":memory:"
    else:
        resolved = str(Path(db_path).expanduser())
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    _apply_schema(conn)
    return UsageStore(conn)


#: Additive, best-effort migrations for databases created before a
#: given column existed. SQLite's ``ADD COLUMN`` raises
#: ``OperationalError`` when the column already exists; we swallow that
#: specific case so :func:`_apply_schema` stays idempotent across boots.
#: Anything else (syntax error, broken DB) re-raises so the operator
#: sees it during startup.
_ADDITIVE_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE reviewer_usage "
    "ADD COLUMN findings_filtered_count INTEGER NOT NULL DEFAULT 0",
)


def _apply_schema(conn: sqlite3.Connection) -> None:
    for stmt in _DDL_STATEMENTS:
        conn.execute(stmt)
    for stmt in _ADDITIVE_MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            # "duplicate column name" means the migration already ran;
            # anything else is a real failure and should surface.
            if "duplicate column name" not in str(exc).lower():
                raise
    conn.commit()


__all__ = [
    "SCHEMA_VERSION",
    "UsageStore",
    "UsageSummary",
    "open_usage_store",
]
