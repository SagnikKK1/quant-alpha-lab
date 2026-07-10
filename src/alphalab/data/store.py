"""Parquet/DuckDB point-in-time store + canonical content hashing.

Reproducibility is defined on CONTENT, not file bytes: Parquet encodings vary
across library versions, so the CI gate compares `canonical_hash()` values —
a SHA256 over a canonical serialization (rows sorted by key, datetimes as
ms-integer epochs). Two stores with identical data hash identically no matter
how or when the Parquet was written.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import polars as pl

from alphalab.config import CURATED_DIR

KEY_COLUMNS = {
    "klines": ["symbol", "open_time"],
    "funding": ["symbol", "calc_time"],
}


def _dataset_dir(kind: str) -> Path:
    return CURATED_DIR / kind


def write_symbol(df: pl.DataFrame, kind: str, symbol: str) -> Path:
    """Write one symbol's curated frame; full-file replace (idempotent)."""
    path = _dataset_dir(kind) / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = KEY_COLUMNS[kind]
    df.sort(keys).write_parquet(path)
    return path


def scan(kind: str, symbols: list[str] | None = None) -> pl.LazyFrame:
    """Lazy scan over the curated dataset (optionally a subset of symbols)."""
    base = _dataset_dir(kind)
    if symbols is None:
        return pl.scan_parquet(base / "*.parquet")
    frames = [pl.scan_parquet(base / f"{s}.parquet") for s in symbols]
    return pl.concat(frames)


def duckdb_connection() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with views over the curated datasets."""
    con = duckdb.connect()
    for kind in KEY_COLUMNS:
        glob = str(_dataset_dir(kind) / "*.parquet")
        if list(_dataset_dir(kind).glob("*.parquet")):
            con.execute(f"CREATE VIEW {kind} AS SELECT * FROM read_parquet('{glob}')")
    return con


def canonical_hash(df: pl.DataFrame, keys: list[str]) -> str:
    """Content hash: order-invariant in input, sensitive to any value change.

    Canonical form: sort by ``keys``, datetimes -> ms int64 epochs, serialize
    to CSV, SHA256 the bytes. Float formatting is polars' shortest-roundtrip
    repr — deterministic for a pinned polars major version (pyproject pins
    ``polars>=1.10,<2``).
    """
    canon = df.with_columns(
        pl.col(c).dt.epoch(time_unit="ms")
        for c, dt in zip(df.columns, df.dtypes, strict=True)
        if isinstance(dt, pl.Datetime)
    ).sort(keys)
    # Column order is part of the canonical form.
    canon = canon.select(sorted(canon.columns))
    return hashlib.sha256(canon.write_csv().encode()).hexdigest()


def dataset_manifest(kind: str) -> pl.DataFrame:
    """Per-symbol content hashes for the reproducibility gate."""
    keys = KEY_COLUMNS[kind]
    rows = []
    for path in sorted(_dataset_dir(kind).glob("*.parquet")):
        df = pl.read_parquet(path)
        rows.append(
            {
                "symbol": path.stem,
                "rows": df.height,
                "content_sha256": canonical_hash(df, keys),
            }
        )
    return pl.DataFrame(
        rows, schema={"symbol": pl.String, "rows": pl.Int64, "content_sha256": pl.String}
    )
