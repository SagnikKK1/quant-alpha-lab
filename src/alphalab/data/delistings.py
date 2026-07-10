"""Delisted-contract handling: settlement inference and history truncation.

The vendor keeps producing placeholder klines (frozen price, zero volume) for
years after a perp settles. Curation truncates each symbol at its settlement
so the placeholder era never reaches the store. The registry is data (CSV in
the repo, reviewed in PRs), inference is a helper for building/refreshing it —
an inferred date is a candidate until confirmed against the delisting
announcement.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

REGISTRY_SCHEMA = {
    "symbol": pl.String,
    "settlement_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "source": pl.String,  # announcement URL or "inferred"
}


def load_registry(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=REGISTRY_SCHEMA)
    raw = pl.read_csv(path, schema_overrides={"symbol": pl.String, "source": pl.String})
    return raw.with_columns(pl.col("settlement_time").str.to_datetime(time_zone="UTC"))


def infer_settlements(klines: pl.DataFrame, stale_days: int = 30) -> pl.DataFrame:
    """Candidate settlements: symbols whose real activity ended long ago.

    A symbol's inferred settlement is its last bar with actual trading
    (volume>0 or count>0). It's only a candidate if that bar is more than
    ``stale_days`` before the newest bar in the whole panel (i.e. the symbol
    went quiet while the world kept trading) — this distinguishes delistings
    from the panel simply ending.
    """
    panel_end = klines.select(pl.col("open_time").max()).item()
    last_real = (
        klines.filter((pl.col("volume") > 0) | (pl.col("count") > 0))
        .group_by("symbol")
        .agg(pl.col("open_time").max().alias("settlement_time"))
    )
    return (
        last_real.filter(
            pl.col("settlement_time")
            < panel_end - pl.duration(days=stale_days)
        )
        .with_columns(pl.lit("inferred").alias("source"))
        .sort("symbol")
    )


def truncate_at_settlement(
    df: pl.DataFrame, registry: pl.DataFrame, time_col: str = "open_time"
) -> pl.DataFrame:
    """Drop every row after a symbol's registered settlement time.

    Applies to any per-symbol time series — klines (open_time) AND funding
    (calc_time): the vendor keeps publishing placeholder funding records after
    settlement too, so both series must be cut. Symbols not in the registry
    pass through untouched. `audits.find_placeholder_runs` then asserts
    nothing placeholder-shaped survived curation.
    """
    if registry.is_empty():
        return df
    reg = registry.select("symbol", "settlement_time")
    return (
        df.join(reg, on="symbol", how="left")
        .filter(
            pl.col("settlement_time").is_null()
            | (pl.col(time_col) <= pl.col("settlement_time"))
        )
        .drop("settlement_time")
    )
