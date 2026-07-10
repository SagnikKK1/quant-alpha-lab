"""Daily decision/execution frames from the hourly panel.

TIMING CONVENTION (tested in tests/test_timing.py — the classic bug):
- Decision date D: signals may use information through the close of D's last
  hourly bar (23:59:59.999 UTC).
- Execution: fills happen at the OPEN of the first hourly bar of D+1.
- Holding: from the D+1 open to the D+2 open (one rebalance period).

The engine never sees a close price for filling; the execution frame carries
only next-day opens, so "fill at signal-bar close" is unrepresentable.
"""

from __future__ import annotations

import polars as pl


def daily_frame(klines: pl.DataFrame) -> pl.DataFrame:
    """Per (symbol, date): decision-time info + next-day execution open.

    Columns:
    - close:      last hourly close of the date (decision-time price)
    - dollar_volume: sum of quote_volume over the date (for participation)
    - sigma:      realized daily vol from hourly log-returns (for impact)
    - exec_open:  open of the FIRST hourly bar of the NEXT calendar date with
                  data (the fill price for decisions made on this date)
    - exec_date:  the date that open belongs to
    """
    by_day = (
        klines.sort("symbol", "open_time")
        .with_columns(
            pl.col("open_time").dt.date().alias("date"),
            (pl.col("close") / pl.col("close").shift(1)).log().over("symbol").alias("lr"),
        )
        .group_by("symbol", "date", maintain_order=True)
        .agg(
            pl.col("open").first().alias("day_open"),
            pl.col("close").last().alias("close"),
            pl.col("quote_volume").sum().alias("dollar_volume"),
            (pl.col("lr").std() * (24**0.5)).alias("sigma"),
        )
    )
    return (
        by_day.sort("symbol", "date")
        .with_columns(
            pl.col("day_open").shift(-1).over("symbol").alias("exec_open"),
            pl.col("date").shift(-1).over("symbol").alias("exec_date"),
        )
        .drop("day_open")
    )


def execution_returns(daily: pl.DataFrame) -> pl.DataFrame:
    """Return earned by a decision made on date D: exec_open(D+1) ->
    exec_open(D+2), per symbol. Null on the last decision date (no exit
    price yet) — those rows can't be scored and are dropped."""
    return (
        daily.sort("symbol", "date")
        .with_columns(
            (pl.col("exec_open").shift(-1) / pl.col("exec_open") - 1.0)
            .over("symbol")
            .alias("exec_ret")
        )
        .drop_nulls(["exec_ret"])
    )
