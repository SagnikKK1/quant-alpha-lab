"""Tape calibration of the impact law — the no-client-data version of the
institutional TCA regression (HANDOFF §10.2).

With no client fills, the identifiable object is the MARKET's response to
observed aggressor flow. From tick-level aggTrades (aggressor-flagged):

  window w (5 min):  net signed dollar flow F_w, price return r_w
  regression:        sign(F_w) * r_w = alpha * sigma_d * sqrt(|F_w| / V_d) + eps
  log-log variant:   log E|r| ~ const + beta * log(|F_w| / V_d)   -> beta

alpha maps directly onto CostModel.impact_coeff and beta onto
size_sensitivity, because the regressor is built with the SAME normalization
(daily sigma, daily dollar volume) the backtest uses. Stated limitations:
public prints don't reveal parent orders (windows chop metaorders), and
flow is contemporaneous with returns (feedback trading inflates alpha vs a
desk's causal estimate) — so alpha here is an upper-bound-flavored venue
coefficient WITH A CI, not a causal constant. Both caveats are the honest
price of $0 data and are reported alongside the fit.

Usage: python scripts/calibrate_impact.py [--max-mb 250]
"""

from __future__ import annotations

import argparse
import io
import zipfile

import numpy as np
import polars as pl
import requests

from alphalab.backtest.frames import daily_frame
from alphalab.config import BINANCE_VISION_BASE, UM_FUTURES_PREFIX
from alphalab.data import store
from alphalab.data.binance_bulk import download_verified
from alphalab.data.schema import _has_header

AGG_COLUMNS = [
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
]

# Tight/mid/wide names across eras (avoid 2026-07-03/04 SOLUSDT vendor bug).
SAMPLE = [
    ("BTCUSDT", "2021-06-15"), ("BTCUSDT", "2023-09-15"), ("BTCUSDT", "2025-03-14"),
    ("ETHUSDT", "2021-06-15"), ("ETHUSDT", "2024-02-15"),
    ("DOGEUSDT", "2021-06-15"), ("DOGEUSDT", "2023-09-15"), ("DOGEUSDT", "2025-03-14"),
    ("SANDUSDT", "2022-03-15"), ("SANDUSDT", "2023-09-15"),
    ("XTZUSDT", "2021-06-15"), ("XTZUSDT", "2023-09-15"),
    ("CRVUSDT", "2022-03-15"), ("CRVUSDT", "2024-02-15"), ("CRVUSDT", "2025-03-14"),
    ("AVAXUSDT", "2022-03-15"), ("AVAXUSDT", "2024-02-15"),
]

WINDOW_MIN = 5


def aggtrades_url(symbol: str, day: str) -> str:
    return (
        f"{BINANCE_VISION_BASE}/{UM_FUTURES_PREFIX}/daily/aggTrades/"
        f"{symbol}/{symbol}-aggTrades-{day}.zip"
    )


def load_windows(symbol: str, day: str, session, max_mb: float) -> pl.DataFrame | None:
    url = aggtrades_url(symbol, day)
    head = session.head(url, timeout=30)
    if head.status_code == 404:
        return None
    if int(head.headers.get("content-length", 0)) / 1e6 > max_mb:
        print(f"  skip {symbol} {day}: file over {max_mb}MB cap")
        return None
    path = download_verified(url, session=session)
    if path is None:
        return None
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(zf.namelist()[0])
    df = pl.read_csv(
        io.BytesIO(raw),
        has_header=_has_header(raw),
        new_columns=AGG_COLUMNS,
        schema_overrides={"price": pl.Float64, "quantity": pl.Float64,
                          "transact_time": pl.Int64, "is_buyer_maker": pl.Boolean},
    )
    return (
        df.with_columns(
            pl.from_epoch("transact_time", time_unit="ms").alias("ts"),
            pl.when(pl.col("is_buyer_maker")).then(-1.0).otherwise(1.0).alias("side"),
            (pl.col("price") * pl.col("quantity")).alias("notional"),
        )
        .group_by_dynamic("ts", every=f"{WINDOW_MIN}m")
        .agg(
            (pl.col("side") * pl.col("notional")).sum().alias("net_flow"),
            pl.col("price").first().alias("p_open"),
            pl.col("price").last().alias("p_close"),
            pl.len().alias("n_trades"),
        )
        .filter(pl.col("n_trades") >= 10)
        .with_columns(
            pl.lit(symbol).alias("symbol"),
            pl.lit(day).str.to_date().alias("date"),
            (pl.col("p_close") / pl.col("p_open")).log().alias("ret"),
        )
        .select("symbol", "date", "net_flow", "ret")
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-mb", type=float, default=250.0)
    args = ap.parse_args()

    session = requests.Session()
    klines = store.scan("klines").collect()
    daily = daily_frame(klines).select("symbol", "date", "sigma", "dollar_volume")

    frames = []
    for symbol, day in SAMPLE:
        w = load_windows(symbol, day, session, args.max_mb)
        if w is not None:
            frames.append(w)
            print(f"  {symbol} {day}: {w.height} windows")
    obs = pl.concat(frames).join(daily, on=["symbol", "date"], how="inner").drop_nulls()

    # y = sign(F)*r ; x = sigma_d * sqrt(|F|/V_d)  -> alpha via OLS through origin
    obs = obs.with_columns(
        (pl.col("net_flow").abs() / pl.col("dollar_volume")).alias("part"),
        (pl.col("ret") * pl.col("net_flow").sign()).alias("y"),
    ).with_columns((pl.col("sigma") * pl.col("part").sqrt()).alias("x"))
    x = obs["x"].to_numpy()
    y = obs["y"].to_numpy()
    alpha = float((x @ y) / (x @ x))
    resid = y - alpha * x
    se = float(np.sqrt(resid**2 @ x**2) / (x @ x))  # HC-robust for origin OLS
    r2 = 1 - resid.var() / y.var()

    # beta: log-log slope among meaningful-flow windows
    big = obs.filter(pl.col("part") > 1e-6)
    lx = np.log(big["part"].to_numpy())
    ly = np.log(np.abs(big["y"].to_numpy()) / big["sigma"].to_numpy() + 1e-12)
    A = np.vstack([lx, np.ones_like(lx)]).T
    beta, _c = np.linalg.lstsq(A, ly, rcond=None)[0]
    br = ly - A @ np.array([beta, _c])
    beta_se = float(np.sqrt(br.var() / ((lx - lx.mean()) ** 2).sum()))

    print(f"\nobservations: {len(x)} 5-min windows, {obs['symbol'].n_unique()} symbols")
    print(f"alpha (impact_coeff): {alpha:.3f}  (robust se {se:.3f}, "
          f"95% CI [{alpha-1.96*se:.3f}, {alpha+1.96*se:.3f}])   R^2={r2:.3f}")
    print(f"beta (size_sensitivity): {beta:.3f}  (se {beta_se:.3f}, "
          f"95% CI [{beta-1.96*beta_se:.3f}, {beta+1.96*beta_se:.3f}])")
    print("caveats: contemporaneous flow (feedback inflates alpha); windows chop "
          "parent orders; sampled days, not the full tape.")

    per_sym = []
    for s in obs["symbol"].unique().sort():
        o = obs.filter(pl.col("symbol") == s)
        xs, ys = o["x"].to_numpy(), o["y"].to_numpy()
        per_sym.append({"symbol": s, "alpha": round(float((xs @ ys) / (xs @ xs)), 3),
                        "n": len(xs)})
    print(pl.DataFrame(per_sym))

    obs.select("symbol", "date", "part", "y", "x").write_csv(
        "reports/impact_calibration_obs.csv"
    )
    print("-> reports/impact_calibration_obs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
