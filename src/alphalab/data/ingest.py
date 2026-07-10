"""Ingest CLI: download -> parse -> curate -> audit -> manifest.

Usage:
    python -m alphalab.data.ingest --symbols BTCUSDT ETHUSDT \
        --start 2020-01 --end 2020-03 --interval 1h
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl
import requests

from alphalab.config import CURATED_DIR
from alphalab.data import audits, binance_bulk, delistings, schema, store

log = logging.getLogger("alphalab.ingest")

REGISTRY_PATH = Path(__file__).resolve().parents[3] / "registry" / "delistings.csv"


def ingest_symbol(
    symbol: str,
    months: list[str],
    interval: str,
    session: requests.Session,
    registry: pl.DataFrame,
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

    klines = delistings.truncate_at_settlement(pl.concat(kline_frames), registry)
    store.write_symbol(klines, "klines", symbol)
    n_funding = 0
    if funding_frames:
        funding = delistings.truncate_at_settlement(
            pl.concat(funding_frames), registry, time_col="calc_time"
        )
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
    session = requests.Session()

    for symbol in args.symbols:
        result = ingest_symbol(symbol, months, args.interval, session, registry)
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
    findings = audits.run_all(klines, funding, args.interval)
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
