"""Baseline: cross-sectional rank momentum on universe members.

No fitting, no hyperparameters beyond the lookback — this is the floor every
model in the ladder must beat, run through the SAME harness (walk-forward,
costs, funding, bootstrap). Signal at decision date D uses closes through D
(the daily_frame close = last hourly close of D) on names that are universe
members at D (membership itself decided with D-1 information).
"""

from __future__ import annotations

import polars as pl

from alphalab.backtest.frames import daily_frame


def rank_momentum_scores(
    klines: pl.DataFrame,
    universe: pl.DataFrame,  # (date, symbol, ...) from rolling_universe
    lookback_days: int = 30,
) -> pl.DataFrame:
    """(date, symbol, score): cross-sectional rank in [0,1] of the trailing
    ``lookback_days`` close-to-close return, members only."""
    daily = daily_frame(klines)
    mom = (
        daily.sort("symbol", "date")
        .with_columns(
            (pl.col("close") / pl.col("close").shift(lookback_days) - 1.0)
            .over("symbol")
            .alias("mom")
        )
        .drop_nulls("mom")
    )
    members = universe.select("date", "symbol")
    return (
        mom.join(members, on=["date", "symbol"], how="inner")
        .with_columns(
            ((pl.col("mom").rank("average").over("date") - 1)
             / (pl.len().over("date") - 1).clip(lower_bound=1)).alias("score")
        )
        .select("date", "symbol", "score")
    )
