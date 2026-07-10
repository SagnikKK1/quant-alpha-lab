"""Ingest CLI: download -> parse -> curate -> audit -> manifest.

Usage:
    python -m alphalab.data.ingest --symbols BTCUSDT ETHUSDT \
        --start 2020-01 --end 2020-03 --interval 1h
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import polars as pl
import requests

from alphalab.config import CURATED_DIR
from alphalab.data import audits, binance_bulk, delistings, schema, store

log = logging.getLogger("alphalab.ingest")

REGISTRY_PATH = Path(__file__).resolve().parents[3] / "registry" / "delistings.csv"
EXCLUSIONS_PATH = Path(__file__).resolve().parents[3] / "registry" / "exclusions.csv"


def backfill_gaps_from_daily(
    klines: pl.DataFrame,
    symbol: str,
    interval: str,
    session: requests.Session,
) -> pl.DataFrame:
    """Fill holes in the monthly zips from the daily files.

    Verified vendor defect: several 2022 monthly um klines files are missing
    whole days (e.g. late Feb and Apr 1-2 2022 across many symbols) that ARE
    present as daily files. Only the missing dates are fetched; a date whose
    daily file also 404s stays missing and the gap audit reports it honestly.
    """
    gaps = audits.find_gaps(klines, interval)
    if gaps.is_empty():
        return klines
    step_ms = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}[interval]
    missing_days: set[str] = set()
    for row in gaps.iter_rows(named=True):
        end = row["open_time"]  # first bar AFTER the hole
        start = end - timedelta(milliseconds=row["delta_ms"] - step_ms)
        d = start.date()
        while d < end.date() or (d == end.date() and start < end):
            missing_days.add(d.isoformat())
            d += timedelta(days=1)
    frames = [klines]
    filled = 0
    for day in sorted(missing_days):
        path = binance_bulk.download_verified(
            binance_bulk.daily_kline_url(symbol, interval, day), session=session
        )
        if path is not None:
            frames.append(schema.parse_klines(binance_bulk.read_zip_csv(path), symbol))
            filled += 1
    if filled:
        log.info("%s: backfilled %d/%d missing day(s) from daily files",
                 symbol, filled, len(missing_days))
    return (
        pl.concat(frames)
        .unique(subset=["symbol", "open_time"], keep="first")
        .sort("symbol", "open_time")
    )


def ingest_symbol(
    symbol: str,
    months: list[str],
    interval: str,
    session: requests.Session,
    registry: pl.DataFrame,
    exclusions: pl.DataFrame,
) -> dict:
    kline_frames, funding_frames = [], []
    for month in months:
        kpath = binance_bulk.download_verified(
            binance_bulk.monthly_kline_url(symbol, interval, month), session=session
        )
        if kpath is not None:
            kline_frames.append(schema.parse_klines(binance_bulk.read_zip_csv(kpath), symbol))
        fpath = binance_bulk.download_verified(
            binance_bulk.monthly_funding_url(symbol, month), session=session
        )
        if fpath is not None:
            funding_frames.append(schema.parse_funding(binance_bulk.read_zip_csv(fpath), symbol))

    if not kline_frames:
        return {"symbol": symbol, "months": 0, "klines": 0, "funding": 0}

    merged = backfill_gaps_from_daily(pl.concat(kline_frames), symbol, interval, session)
    klines = delistings.truncate_at_settlement(merged, registry)
    klines = delistings.drop_exclusion_windows(klines, exclusions)
    store.write_symbol(klines, "klines", symbol)
    n_funding = 0
    if funding_frames:
        funding = delistings.truncate_at_settlement(
            pl.concat(funding_frames), registry, time_col="calc_time"
        )
        funding = delistings.drop_exclusion_windows(funding, exclusions, time_col="calc_time")
        store.write_symbol(funding, "funding", symbol)
        n_funding = funding.height
    return {
        "symbol": symbol,
        "months": len(kline_frames),
        "klines": klines.height,
        "funding": n_funding,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM")
    parser.add_argument("--end", required=True, help="YYYY-MM")
    parser.add_argument("--interval", default="1h")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    months = binance_bulk.month_range(args.start, args.end)
    registry = delistings.load_registry(REGISTRY_PATH)
    exclusions = delistings.load_exclusions(EXCLUSIONS_PATH)
    session = requests.Session()

    for symbol in args.symbols:
        result = ingest_symbol(symbol, months, args.interval, session, registry, exclusions)
        log.info("%s", result)

    # Post-ingest audit over everything just written.
    klines = store.scan("klines", args.symbols).collect()
    with_funding = [
        s for s in args.symbols if (CURATED_DIR / "funding" / f"{s}.parquet").exists()
    ]
    empty_funding_schema = {
        "symbol": pl.String,
        "calc_time": pl.Datetime(time_zone="UTC"),
        "funding_interval_hours": pl.Int64,
        "last_funding_rate": pl.Float64,
    }
    funding = (
        store.scan("funding", with_funding).collect()
        if with_funding
        else pl.DataFrame(schema=empty_funding_schema)
    )
    findings = audits.run_all(klines, funding, args.interval, exclusions)
    clean = True
    for name, frame in findings.items():
        if frame.is_empty():
            log.info("audit %-16s OK", name)
        else:
            clean = False
            log.warning("audit %-16s %d findings\n%s", name, frame.height, frame.head(10))

    manifest = store.dataset_manifest("klines")
    manifest_path = CURATED_DIR / "manifest_klines.csv"
    manifest.write_csv(manifest_path)
    log.info("manifest -> %s", manifest_path)
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
