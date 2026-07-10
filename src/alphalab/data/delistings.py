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

# Delist-RELIST case (observed on ICPUSDT: perp delisted Jun 2022, relisted
# later): the placeholder era is an interior window, not a trailing one, so
# truncation can't express it. Exclusion windows drop [start, end] per symbol.
EXCLUSIONS_SCHEMA = {
    "symbol": pl.String,
    "start_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "end_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "source": pl.String,
}


def load_registry(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=REGISTRY_SCHEMA)
    raw = pl.read_csv(path, schema_overrides={"symbol": pl.String, "source": pl.String})
    return raw.with_columns(pl.col("settlement_time").str.to_datetime(time_zone="UTC"))


def load_exclusions(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=EXCLUSIONS_SCHEMA)
    raw = pl.read_csv(path, schema_overrides={"symbol": pl.String, "source": pl.String})
    return raw.with_columns(
        pl.col("start_time").str.to_datetime(time_zone="UTC"),
        pl.col("end_time").str.to_datetime(time_zone="UTC"),
    )


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


def infer_exclusions(
    klines: pl.DataFrame, min_run: int = 24, interval: str = "1h"
) -> pl.DataFrame:
    """Candidate exclusion windows: INTERIOR placeholder runs (real trading
    resumes after the run — the delist-relist signature). Trailing runs are
    settlement candidates, not exclusions.

    The window ends at the bar before trading actually RESUMES, not at the
    last placeholder file: on ICPUSDT the vendor stopped writing placeholder
    files weeks before the relist, so the halt outlives the placeholder run.
    """
    from alphalab.config import INTERVAL_MS
    from alphalab.data.audits import find_placeholder_runs

    runs = find_placeholder_runs(klines, min_run=min_run)
    if runs.is_empty():
        return pl.DataFrame(schema=EXCLUSIONS_SCHEMA)
    real = klines.filter((pl.col("volume") > 0) | (pl.col("count") > 0))
    last_real = real.group_by("symbol").agg(pl.col("open_time").max().alias("last_real"))
    interior = runs.join(last_real, on="symbol").filter(pl.col("run_end") < pl.col("last_real"))
    if interior.is_empty():
        return pl.DataFrame(schema=EXCLUSIONS_SCHEMA)

    # First real bar after each run -> window end = resume - one interval.
    resume = (
        interior.join(real.select("symbol", "open_time"), on="symbol")
        .filter(pl.col("open_time") > pl.col("run_end"))
        .group_by("symbol", "run_start")
        .agg(pl.col("open_time").min().alias("resume_time"))
    )
    return (
        interior.join(resume, on=["symbol", "run_start"])
        .select(
            "symbol",
            pl.col("run_start").alias("start_time"),
            (pl.col("resume_time") - pl.duration(milliseconds=INTERVAL_MS[interval])).alias(
                "end_time"
            ),
            pl.lit("inferred").alias("source"),
        )
        .sort("symbol", "start_time")
    )


def drop_exclusion_windows(
    df: pl.DataFrame, exclusions: pl.DataFrame, time_col: str = "open_time"
) -> pl.DataFrame:
    """Remove rows falling inside any [start_time, end_time] window for their
    symbol. Supports multiple windows per symbol."""
    if exclusions.is_empty():
        return df
    inside = (
        df.select("symbol", time_col)
        .join(exclusions.select("symbol", "start_time", "end_time"), on="symbol", how="inner")
        .filter(
            (pl.col(time_col) >= pl.col("start_time"))
            & (pl.col(time_col) <= pl.col("end_time"))
        )
        .select("symbol", time_col)
        .unique()
    )
    return df.join(inside, on=["symbol", time_col], how="anti")


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
