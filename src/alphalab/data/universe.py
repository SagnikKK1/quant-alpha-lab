"""Time-varying universe by rolling volume rank — strictly point-in-time.

The membership decision for date D uses dollar volume through the END OF
DATE D-1 and nothing later. That one-day shift is the whole no-lookahead
guarantee, and `tests/test_universe.py::test_no_lookahead` exists to make it
impossible to refactor away silently.
"""

from __future__ import annotations

import polars as pl


def daily_dollar_volume(klines: pl.DataFrame) -> pl.DataFrame:
    """Aggregate bar-level quote volume to (date, symbol) dollar volume."""
    return (
        klines.with_columns(pl.col("open_time").dt.date().alias("date"))
        .group_by("date", "symbol")
        .agg(pl.col("quote_volume").sum().alias("dollar_volume"))
        .sort("date", "symbol")
    )


def rolling_universe(
    klines: pl.DataFrame,
    top_n: int = 50,
    lookback_days: int = 30,
    min_days_live: int = 7,
) -> pl.DataFrame:
    """Universe membership per date: top_n by trailing dollar volume.

    - Rolling sum over ``lookback_days`` of daily dollar volume, per symbol,
      on a dense calendar (missing days count 0 — an untraded day is real
      information, not a hole to interpolate).
    - The rolling stat is SHIFTED one day before ranking: membership at D is
      decided with data through D-1.
    - ``min_days_live``: a symbol needs that many observed days before it is
      rankable, so a listing-day volume spike can't instantly buy membership.

    Returns (date, symbol, trailing_dollar_volume, rank) for members only.
    """
    dv = daily_dollar_volume(klines)

    # Dense (date, symbol) grid over each symbol's OWN [first_seen, last_seen]:
    # a settled contract must leave the universe with its last trading day —
    # otherwise its decaying trailing volume keeps a dead, untradeable name
    # ranked for up to `lookback_days` after settlement (observed on LUNAUSDT).
    # Whether a contract trades on date D is point-in-time knowledge.
    dates = dv.select(pl.col("date").unique().sort()).with_columns(pl.lit(1).alias("_j"))
    spans = dv.group_by("symbol").agg(
        pl.col("date").min().alias("first_seen"),
        pl.col("date").max().alias("last_seen"),
    ).with_columns(pl.lit(1).alias("_j"))
    grid = (
        dates.join(spans, on="_j")
        .filter((pl.col("date") >= pl.col("first_seen")) & (pl.col("date") <= pl.col("last_seen")))
        .select("date", "symbol", "first_seen")
    )
    dense = grid.join(dv, on=["date", "symbol"], how="left").with_columns(
        pl.col("dollar_volume").fill_null(0.0)
    )

    ranked = (
        dense.sort("symbol", "date")
        .with_columns(
            pl.col("dollar_volume")
            .rolling_sum(window_size=lookback_days, min_samples=1)
            .over("symbol")
            .alias("trailing_dollar_volume"),
            (pl.col("date") - pl.col("first_seen")).dt.total_days().alias("days_live"),
        )
        # THE point-in-time shift: stats through D-1 decide membership at D.
        .with_columns(
            pl.col("trailing_dollar_volume").shift(1).over("symbol"),
            pl.col("days_live").shift(1).over("symbol"),
        )
        .filter(
            pl.col("trailing_dollar_volume").is_not_null()
            & (pl.col("days_live") >= min_days_live)
        )
        .with_columns(
            pl.col("trailing_dollar_volume")
            .rank(method="ordinal", descending=True)
            .over("date")
            .alias("rank")
        )
        .filter(pl.col("rank") <= top_n)
        .select("date", "symbol", "trailing_dollar_volume", "rank")
        .sort("date", "rank")
    )
    return ranked
