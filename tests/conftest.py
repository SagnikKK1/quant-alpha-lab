"""Shared synthetic fixtures. No network, no real data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest


def make_klines(
    symbol: str,
    start: datetime,
    n_bars: int,
    interval_h: int = 1,
    volume: float = 100.0,
    count: int = 50,
    price: float = 10.0,
) -> pl.DataFrame:
    """Synthetic typed klines matching schema.parse_klines output."""
    times = [start + timedelta(hours=i * interval_h) for i in range(n_bars)]
    return pl.DataFrame(
        {
            "symbol": [symbol] * n_bars,
            "open_time": times,
            "open": [price] * n_bars,
            "high": [price * 1.01] * n_bars,
            "low": [price * 0.99] * n_bars,
            "close": [price] * n_bars,
            "volume": [volume] * n_bars,
            "close_time": [
                t + timedelta(hours=interval_h) - timedelta(milliseconds=1) for t in times
            ],
            "quote_volume": [volume * price] * n_bars,
            "count": [count] * n_bars,
            "taker_buy_volume": [volume / 2] * n_bars,
            "taker_buy_quote_volume": [volume * price / 2] * n_bars,
        }
    ).with_columns(
        pl.col("open_time").dt.replace_time_zone("UTC"),
        pl.col("close_time").dt.replace_time_zone("UTC"),
    )


@pytest.fixture
def t0() -> datetime:
    return datetime(2023, 1, 1, tzinfo=UTC)
