"""Data-quality audits. Each returns a frame of FINDINGS (empty = clean).

These run as pytest assertions in CI and as an ingest-time report. They are
deliberately dumb and mechanical — an audit that needs judgment is a test that
will get skipped. Severity policy lives in the callers.
"""

from __future__ import annotations

import polars as pl

from alphalab.config import INTERVAL_MS


def find_duplicates(klines: pl.DataFrame) -> pl.DataFrame:
    """Rows sharing (symbol, open_time)."""
    return (
        klines.group_by("symbol", "open_time")
        .len()
        .filter(pl.col("len") > 1)
        .sort("symbol", "open_time")
    )


def find_gaps(
    klines: pl.DataFrame, interval: str, exclusions: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Missing bars between each symbol's first and last observed bar.

    Late listings are NOT gaps (the panel is legitimately unbalanced); only
    holes inside a symbol's own [first, last] range are findings. A gap whose
    missing window lies entirely inside a registered exclusion window (a
    delist-relist halt — the contract didn't exist) is waived.
    """
    step_ms = INTERVAL_MS[interval]
    gaps = (
        klines.sort("symbol", "open_time")
        .with_columns(
            pl.col("open_time").diff().over("symbol").dt.total_milliseconds().alias("delta_ms")
        )
        .filter(pl.col("delta_ms") > step_ms)
        .with_columns(((pl.col("delta_ms") // step_ms) - 1).alias("missing_bars"))
        .select("symbol", "open_time", "delta_ms", "missing_bars")
    )
    if exclusions is None or exclusions.is_empty() or gaps.is_empty():
        return gaps
    waived = (
        gaps.with_columns(
            (pl.col("open_time") - pl.duration(milliseconds=pl.col("delta_ms") - step_ms)).alias(
                "missing_start"
            ),
            (pl.col("open_time") - pl.duration(milliseconds=step_ms)).alias("missing_end"),
        )
        .join(exclusions.select("symbol", "start_time", "end_time"), on="symbol", how="inner")
        .filter(
            (pl.col("missing_start") >= pl.col("start_time"))
            & (pl.col("missing_end") <= pl.col("end_time"))
        )
        .select("symbol", "open_time")
        .unique()
    )
    return gaps.join(waived, on=["symbol", "open_time"], how="anti")


def check_grid_alignment(klines: pl.DataFrame, interval: str) -> pl.DataFrame:
    """Bars whose open_time is off the UTC interval grid."""
    step_ms = INTERVAL_MS[interval]
    return klines.filter(
        pl.col("open_time").dt.epoch(time_unit="ms") % step_ms != 0
    ).select("symbol", "open_time")


def find_placeholder_runs(klines: pl.DataFrame, min_run: int = 24) -> pl.DataFrame:
    """Consecutive runs of volume==0 AND count==0 bars, per symbol.

    This is the delisted-contract placeholder signature (frozen price, zero
    activity, generated for years after settlement — verified on FTTUSDT).
    Short zero-volume stretches can be genuine illiquidity, hence ``min_run``.
    Returns one row per run: symbol, run start/end, length.
    """
    flagged = (
        klines.sort("symbol", "open_time")
        .with_columns(
            ((pl.col("volume") == 0) & (pl.col("count") == 0)).alias("is_placeholder")
        )
        .with_columns(
            (pl.col("is_placeholder") != pl.col("is_placeholder").shift(1, fill_value=False))
            .cum_sum()
            .over("symbol")
            .alias("run_id")
        )
        .filter(pl.col("is_placeholder"))
    )
    return (
        flagged.group_by("symbol", "run_id")
        .agg(
            pl.col("open_time").min().alias("run_start"),
            pl.col("open_time").max().alias("run_end"),
            pl.len().alias("run_len"),
        )
        .filter(pl.col("run_len") >= min_run)
        .drop("run_id")
        .sort("symbol", "run_start")
    )


def check_funding_grid(funding: pl.DataFrame, tolerance_s: int = 60) -> pl.DataFrame:
    """Funding events off the grid implied by their own funding_interval_hours.

    No cadence is assumed: each row is checked against its declared interval
    (Binance um is 8h today; some contracts have switched — the column is the
    source of truth).
    """
    return funding.with_columns(
        (
            pl.col("calc_time").dt.epoch(time_unit="ms")
            % (pl.col("funding_interval_hours") * 3_600_000)
        ).alias("off_grid_ms")
    ).filter(
        pl.min_horizontal(
            pl.col("off_grid_ms"),
            pl.col("funding_interval_hours") * 3_600_000 - pl.col("off_grid_ms"),
        )
        > tolerance_s * 1_000
    ).select("symbol", "calc_time", "funding_interval_hours", "off_grid_ms")


def run_all(
    klines: pl.DataFrame,
    funding: pl.DataFrame,
    interval: str,
    exclusions: pl.DataFrame | None = None,
) -> dict[str, pl.DataFrame]:
    """Convenience bundle used by the ingest CLI report."""
    return {
        "duplicates": find_duplicates(klines),
        "gaps": find_gaps(klines, interval, exclusions),
        "grid_alignment": check_grid_alignment(klines, interval),
        "placeholder_runs": find_placeholder_runs(klines),
        "funding_grid": check_funding_grid(funding),
    }
