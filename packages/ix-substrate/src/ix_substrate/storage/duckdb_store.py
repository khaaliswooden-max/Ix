"""
ix_substrate.storage.duckdb_store
──────────────────────────────────
Local DuckDB sink. Append-only, batched writes. Designed to be swapped for
ClickHouse in production by reimplementing the same two write methods.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Iterable

import duckdb

from ..contracts.events import DisagreementEvent, TickEvent


DDL = """
CREATE TABLE IF NOT EXISTS ticks (
    schema_version VARCHAR,
    feed_kind      VARCHAR,
    venue          VARCHAR,
    symbol         VARCHAR,
    venue_ts       TIMESTAMPTZ,
    ingress_ts     TIMESTAMPTZ,
    bid_price      DECIMAL(38, 18),
    bid_size       DECIMAL(38, 18),
    ask_price      DECIMAL(38, 18),
    ask_size       DECIMAL(38, 18)
);

CREATE TABLE IF NOT EXISTS disagreements (
    schema_version    VARCHAR,
    kind              VARCHAR,
    symbol            VARCHAR,
    venues            VARCHAR[],
    window_start      TIMESTAMPTZ,
    window_end        TIMESTAMPTZ,
    magnitude         DECIMAL(38, 18),
    magnitude_unit    VARCHAR,
    leader            VARCHAR,
    half_life_ms      DOUBLE,
    raw_observations  BIGINT
);

CREATE INDEX IF NOT EXISTS ix_ticks_symbol_ts ON ticks (symbol, ingress_ts);
CREATE INDEX IF NOT EXISTS ix_dis_symbol_ts   ON disagreements (symbol, window_end);
"""


class DuckDBStore:
    """Single-process, single-writer sink. Open with context manager."""

    def __init__(self, path: str | Path = ":memory:"):
        self._path = str(path)
        self._con: duckdb.DuckDBPyConnection | None = None

    def __enter__(self) -> "DuckDBStore":
        self._con = duckdb.connect(self._path)
        self._con.execute(DDL)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise RuntimeError("DuckDBStore used outside context manager")
        return self._con

    # ---- writes ----

    def write_ticks(self, events: Iterable[TickEvent]) -> int:
        rows = [
            (
                e.schema_version, e.feed_kind.value, e.venue, e.symbol,
                e.venue_ts, e.ingress_ts,
                e.bid_price, e.bid_size, e.ask_price, e.ask_size,
            )
            for e in events
        ]
        if not rows:
            return 0
        self.con.executemany(
            "INSERT INTO ticks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        return len(rows)

    def write_disagreements(self, events: Iterable[DisagreementEvent]) -> int:
        rows = [
            (
                e.schema_version, e.kind.value, e.symbol, list(e.venues),
                e.window_start, e.window_end,
                e.magnitude, e.magnitude_unit, e.leader,
                e.half_life_ms, e.raw_observations,
            )
            for e in events
        ]
        if not rows:
            return 0
        self.con.executemany(
            "INSERT INTO disagreements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        return len(rows)

    # ---- reads ----

    def count(self, table: str) -> int:
        if table not in {"ticks", "disagreements"}:
            raise ValueError(f"unknown table {table!r}")
        return self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
