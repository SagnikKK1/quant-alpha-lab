"""Typed parsing of Binance um-futures bulk CSVs.

Encodes two verified source quirks (HANDOFF §12.1):
- **Header seam**: files from 2022-01 onward carry a header row; earlier files
  don't. Detection is per-file (first field numeric => headerless), so the
  parser never depends on the filename's date.
- **Milliseconds**: um-futures timestamps are ms epochs (the 2025 microsecond
  migration applied to spot only). Parsed to UTC datetimes.
"""

from __future__ import annotations

import io

import polars as pl

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def _has_header(csv_bytes: bytes) -> bool:
    first_field = csv_bytes.split(b"\n", 1)[0].split(b",", 1)[0].strip()
    try:
        float(first_field)
        return False
    except ValueError:
        return True


def parse_klines(csv_bytes: bytes, symbol: str) -> pl.DataFrame:
    """Parse a klines CSV (either side of the header seam) to a typed frame."""
    df = pl.read_csv(
        io.BytesIO(csv_bytes),
        has_header=_has_header(csv_bytes),
        new_columns=KLINE_COLUMNS,
        schema_overrides={
            "open_time": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "close_time": pl.Int64,
            "quote_volume": pl.Float64,
            "count": pl.Int64,
            "taker_buy_volume": pl.Float64,
            "taker_buy_quote_volume": pl.Float64,
        },
    )
    return (
        df.drop("ignore")
        .with_columns(
            pl.lit(symbol).alias("symbol"),
            pl.from_epoch("open_time", time_unit="ms").dt.replace_time_zone("UTC"),
            pl.from_epoch("close_time", time_unit="ms").dt.replace_time_zone("UTC"),
        )
        .select(["symbol", *[c for c in KLINE_COLUMNS if c != "ignore"]])
    )


def parse_funding(csv_bytes: bytes, symbol: str) -> pl.DataFrame:
    """Parse a fundingRate CSV. Keeps funding_interval_hours — never assume 8h."""
    df = pl.read_csv(
        io.BytesIO(csv_bytes),
        has_header=_has_header(csv_bytes),
        new_columns=FUNDING_COLUMNS,
        schema_overrides={
            "calc_time": pl.Int64,
            "funding_interval_hours": pl.Int64,
            "last_funding_rate": pl.Float64,
        },
    )
    return df.with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.from_epoch("calc_time", time_unit="ms").dt.replace_time_zone("UTC"),
    ).select(["symbol", *FUNDING_COLUMNS])
