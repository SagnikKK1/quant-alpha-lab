"""Predicted vs REAL half-spreads: validate the Abdi-Ranaldo estimator
against actually-quoted spreads from Binance bookTicker history.

bookTicker daily files (tick-level best bid/ask) exist from 2023-05-16 to
2024-03-30 (discontinued after) — inside our panel, so for that window the
"real" half-spread is knowable: median over ticks of (ask-bid)/2/mid. This
is the first predicted-vs-realized cost reconciliation of the project; the
live shadow feed later plays the same role for the post-2024 era.

Usage:
    python scripts/validate_spread_model.py [--max-mb 200]
"""

from __future__ import annotations

import argparse
import io
import zipfile
from datetime import date

import polars as pl
import requests

from alphalab.backtest.spreads import abdi_ranaldo_half_spread, tick_bound_half_spread
from alphalab.config import BINANCE_VISION_BASE, UM_FUTURES_PREFIX
from alphalab.data import store
from alphalab.data.binance_bulk import download_verified

# Mix of tight (BTC/ETH), mid (DOGE/SAND), and wide tick-constrained
# (XTZ/CRV) names; several days spread across the bookTicker era.
SAMPLE = [
    ("BTCUSDT", "2023-09-15"),
    ("ETHUSDT", "2023-09-15"),
    ("DOGEUSDT", "2023-09-15"),
    ("SANDUSDT", "2023-09-15"),
    ("XTZUSDT", "2023-09-15"),
    ("CRVUSDT", "2023-09-15"),
    ("BTCUSDT", "2024-02-15"),
    ("DOGEUSDT", "2024-02-15"),
    ("XTZUSDT", "2024-02-15"),
    ("CRVUSDT", "2024-02-15"),
    ("SANDUSDT", "2023-06-15"),
    ("XTZUSDT", "2023-06-15"),
]


def bookticker_url(symbol: str, day: str) -> str:
    return (
        f"{BINANCE_VISION_BASE}/{UM_FUTURES_PREFIX}/daily/bookTicker/"
        f"{symbol}/{symbol}-bookTicker-{day}.zip"
    )


def quoted_half_spread_bps(symbol: str, day: str, session, max_mb: float) -> float | None:
    url = bookticker_url(symbol, day)
    head = session.head(url, timeout=30)
    if head.status_code == 404:
        return None
    size_mb = int(head.headers.get("content-length", 0)) / 1e6
    if size_mb > max_mb:
        print(f"  skip {symbol} {day}: {size_mb:.0f}MB > {max_mb}MB cap")
        return None
    path = download_verified(url, session=session)
    if path is None:
        return None
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(zf.namelist()[0])
    df = pl.read_csv(
        io.BytesIO(raw),
        columns=["best_bid_price", "best_ask_price"],
    )
    rel = (
        (pl.col("best_ask_price") - pl.col("best_bid_price"))
        / ((pl.col("best_ask_price") + pl.col("best_bid_price")) / 2)
        / 2
        * 1e4
    )
    return float(df.select(rel.median()).item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-mb", type=float, default=200.0)
    args = ap.parse_args()

    session = requests.Session()
    klines = store.scan("klines").collect()
    ar = abdi_ranaldo_half_spread(klines)
    tick = tick_bound_half_spread(klines)

    rows = []
    for symbol, day in SAMPLE:
        real = quoted_half_spread_bps(symbol, day, session, args.max_mb)
        if real is None:
            continue
        d = date.fromisoformat(day)
        cond = (pl.col("symbol") == symbol) & (pl.col("date") == d)
        p_ar, p_tick = ar.filter(cond), tick.filter(cond)
        if p_ar.is_empty() or p_tick.is_empty():
            continue
        rows.append(
            {
                "symbol": symbol,
                "date": day,
                "real_bps": round(real, 3),
                "tick_model_bps": round(p_tick["half_spread_bps"][0], 3),
                "ar_model_bps": round(p_ar["half_spread_bps"][0], 3),
            }
        )
        print(f"  {symbol} {day}: real={real:.3f} tick={p_tick['half_spread_bps'][0]:.3f} "
              f"ar={p_ar['half_spread_bps'][0]:.3f} (bps)")

    out = pl.DataFrame(rows).with_columns(
        (pl.col("tick_model_bps") / pl.col("real_bps")).round(2).alias("tick_ratio"),
    )
    print(out)
    if out.height >= 4:
        for col in ("tick_model_bps", "ar_model_bps"):
            sp = out.select(pl.corr("real_bps", col, method="spearman")).item()
            err = out.select(
                ((pl.col(col) - pl.col("real_bps")).abs() / pl.col("real_bps")).median()
            ).item()
            print(f"{col}: spearman={sp:.3f} median_abs_rel_err={err:.1%}")
    out.write_csv("reports/spread_validation.csv")
    print("-> reports/spread_validation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
