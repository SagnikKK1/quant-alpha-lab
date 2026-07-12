"""Book-walk study: the impact curve checked against REAL displayed depth.

Bybit publishes historical 500-level order books (snapshot+delta, no login).
Reconstructing the book through the day and walking the ask side with
child-order sizes gives the INSTANTANEOUS taker cost of crossing visible
liquidity — a hard, empirical reference for the impact term at child-order
scale (it excludes book resilience/refill, so it upper-bounds the cost of a
single immediate child, while the sqrt law prices a whole worked metaorder;
the comparison is meaningful for child sizes, and that's what's compared).

Cross-venue caveat, stated: the books are Bybit's, while sigma/volume in the
model prediction come from our Binance panel — liquidity levels differ, but
the SHAPE of cost-vs-size and the order of magnitude are the deliverables.

Outputs per (symbol, day, size): median walk cost in bps across ~5-min book
samples, plus the sqrt-law band [alpha=0.5 (lit floor), 1.64 (tape ceiling)]
at matching participation. reports/book_walk_study.csv.

Usage: python scripts/book_walk_study.py [--max-mb 400]
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile

import polars as pl
import requests

from alphalab.backtest.frames import daily_frame
from alphalab.data import store

BYBIT_URL = "https://quote-saver.bycsi.com/orderbook/linear/{sym}/{day}_{sym}_ob500.data.zip"

SAMPLE = [
    ("BTCUSDT", "2023-09-15"),
    ("DOGEUSDT", "2023-09-15"),
    ("XTZUSDT", "2023-09-15"),
    ("DOGEUSDT", "2024-02-15"),
]
SIZES_USD = [1e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6]
SAMPLE_EVERY_MS = 5 * 60 * 1000
ALPHA_BAND = (0.5, 1.64)  # literature floor .. tape-calibrated ceiling


def walk_cost_bps(asks: dict[float, float], bids: dict[float, float], usd: float):
    """Cost of buying `usd` notional by walking the asks, vs mid."""
    if not asks or not bids:
        return None
    best_ask = min(asks)
    best_bid = max(bids)
    mid = (best_ask + best_bid) / 2
    remaining = usd
    paid = 0.0
    qty_total = 0.0
    for px in sorted(asks):
        level_usd = px * asks[px]
        take = min(remaining, level_usd)
        paid += take
        qty_total += take / px
        remaining -= take
        if remaining <= 0:
            break
    if remaining > 0:
        return None  # visible book exhausted
    vwap = paid / qty_total
    return (vwap / mid - 1.0) * 1e4


def process_day(symbol: str, day: str, session, max_mb: float) -> list[dict]:
    url = BYBIT_URL.format(sym=symbol, day=day)
    head = session.head(url, timeout=30)
    if head.status_code != 200:
        print(f"  {symbol} {day}: HTTP {head.status_code}, skipping")
        return []
    size_mb = int(head.headers.get("content-length", 0)) / 1e6
    if size_mb > max_mb:
        print(f"  skip {symbol} {day}: {size_mb:.0f}MB > cap")
        return []
    resp = session.get(url, timeout=600)
    resp.raise_for_status()

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    next_sample = None
    samples: dict[float, list[float]] = {s: [] for s in SIZES_USD}
    n_books = 0
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            for line in f:
                msg = json.loads(line)
                data = msg["data"]
                if msg["type"] == "snapshot":
                    bids = {float(p): float(q) for p, q in data["b"]}
                    asks = {float(p): float(q) for p, q in data["a"]}
                else:
                    for p, q in data["b"]:
                        p, q = float(p), float(q)
                        if q == 0:
                            bids.pop(p, None)
                        else:
                            bids[p] = q
                    for p, q in data["a"]:
                        p, q = float(p), float(q)
                        if q == 0:
                            asks.pop(p, None)
                        else:
                            asks[p] = q
                ts = msg["ts"]
                if next_sample is None:
                    next_sample = ts + SAMPLE_EVERY_MS
                if ts >= next_sample:
                    next_sample = ts + SAMPLE_EVERY_MS
                    n_books += 1
                    for s in SIZES_USD:
                        c = walk_cost_bps(asks, bids, s)
                        if c is not None:
                            samples[s].append(c)
    rows = []
    for s in SIZES_USD:
        if samples[s]:
            med = pl.Series(samples[s]).median()
            rows.append(
                {"symbol": symbol, "date": day, "size_usd": s,
                 "walk_cost_bps": round(float(med), 3),
                 "n_samples": len(samples[s])}
            )
    print(f"  {symbol} {day}: {n_books} book samples")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-mb", type=float, default=400.0)
    args = ap.parse_args()
    session = requests.Session()

    klines = store.scan("klines").collect()
    daily = daily_frame(klines).select("symbol", "date", "sigma", "dollar_volume")

    rows = []
    for symbol, day in SAMPLE:
        rows.extend(process_day(symbol, day, session, args.max_mb))
    out = pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())

    out = out.join(daily, on=["symbol", "date"], how="left").with_columns(
        (
            pl.col("sigma")
            * (pl.col("size_usd") / pl.col("dollar_volume")).sqrt()
            * 1e4
        ).alias("_unit")
    ).with_columns(
        (ALPHA_BAND[0] * pl.col("_unit")).round(3).alias("sqrtlaw_lo_bps"),
        (ALPHA_BAND[1] * pl.col("_unit")).round(3).alias("sqrtlaw_hi_bps"),
    ).drop("_unit", "sigma", "dollar_volume")

    pl.Config.set_tbl_rows(40)
    print(out)
    out.write_csv("reports/book_walk_study.csv")
    print("-> reports/book_walk_study.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
