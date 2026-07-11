"""Institutional implementation shortfall: the execution-horizon frontier.

For a parent order too large to cross at one print, IS(k, AUM) under a
TWAP-over-k-hours schedule decomposes as:

    IS = drift(k) + impact(k, AUM) + spread + fees

- drift(k): MEASURED from our own panel — the signed cost of filling the
  baseline's actual trades at the TWAP of the first k hourly opens instead
  of the arrival print (signal-correlated: momentum keeps moving while you
  work the order). This is yesterday's delay study generalized to schedules.
- impact(k, AUM): the square-root law per child order. With k equal children
  and hourly volume ~ V_d/24, per-unit impact = alpha * sigma_d *
  sqrt(|dw| * AUM / (V_d * k)) — impact decays like 1/sqrt(k).
- spread + fees: constant in k (taker child orders).

The trader's dilemma is drift rising in k vs impact falling in 1/sqrt(k);
the argmin over k per AUM is the (modeled) optimal execution horizon. Impact
uses the literature alpha=1 pending tape calibration — the frontier's SHAPE
is the deliverable; its absolute level inherits alpha's uncertainty.

Usage: python scripts/execution_frontier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from delay_cost_study import trade_list  # noqa: E402

from alphalab.backtest.engine import quantile_weights  # noqa: E402
from alphalab.backtest.frames import daily_frame  # noqa: E402
from alphalab.data import store  # noqa: E402
from alphalab.data.universe import rolling_universe  # noqa: E402
from alphalab.signals.baseline import rank_momentum_scores  # noqa: E402

HORIZONS = [1, 2, 4, 8, 24]
AUMS = [1e5, 1e6, 1e7, 5e7, 2e8]
ALPHA = 1.0  # sqrt-law coefficient; tape calibration replaces this (HANDOFF §10)
FEE_SPREAD_BPS = 5.5  # taker fee + typical tick half-spread, constant in k


def main() -> int:
    klines = store.scan("klines").collect()
    universe = rolling_universe(klines, top_n=30, lookback_days=30)
    weights = quantile_weights(rank_momentum_scores(klines, universe, 30), 0.2)
    trades = trade_list(weights)

    daily = daily_frame(klines).select("symbol", "date", "sigma", "dollar_volume")
    hourly = klines.select("symbol", "open_time", pl.col("open").alias("px"))

    base = (
        trades.with_columns(
            (
                pl.col("date").cast(pl.Datetime(time_unit="us")).dt.replace_time_zone("UTC")
                + pl.duration(days=1)
            ).alias("t0")
        )
        .join(hourly.rename({"open_time": "t0", "px": "px0"}), on=["symbol", "t0"])
        .join(daily, on=["symbol", "date"])  # decision-date sigma & volume (PIT)
    )
    # join opens for hours 1..23 once
    for h in range(1, 24):
        base = (
            base.with_columns((pl.col("t0") + pl.duration(hours=h)).alias("th"))
            .join(hourly.rename({"open_time": "th", "px": f"px{h}"}),
                  on=["symbol", "th"], how="left")
            .drop("th")
        )

    rows = []
    for k in HORIZONS:
        cols = [pl.col("px0")] + [pl.col(f"px{h}") for h in range(1, k)]
        twap = pl.mean_horizontal(cols)
        drift_expr = (pl.col("dw") * (twap / pl.col("px0") - 1.0)).alias("drift")
        frame = base.with_columns(drift_expr).drop_nulls("drift")
        turnover = frame["dw"].abs().sum()
        drift_bps = frame["drift"].sum() / turnover * 1e4
        for aum in AUMS:
            impact_expr = (
                pl.col("dw").abs()
                * ALPHA
                * pl.col("sigma").fill_null(0.0)
                * (
                    pl.col("dw").abs() * aum
                    / (pl.max_horizontal(pl.col("dollar_volume"), pl.lit(1.0)) * k)
                ).sqrt()
            ).alias("impact")
            imp = frame.with_columns(impact_expr)["impact"].sum() / turnover * 1e4
            rows.append(
                {
                    "aum": f"${aum/1e6:g}M",
                    "horizon_h": k,
                    "drift_bps": round(drift_bps, 2),
                    "impact_bps": round(imp, 2),
                    "total_is_bps": round(drift_bps + imp + FEE_SPREAD_BPS, 2),
                }
            )

    table = pl.DataFrame(rows)
    pl.Config.set_tbl_rows(40)
    print("=== modeled IS per unit traded (bps): drift(k) + impact(k,AUM) + 5.5 ===")
    print(table.pivot(values="total_is_bps", index="aum", on="horizon_h"))
    print("\ndrift component by horizon (bps):",
          dict(table.unique(subset=["horizon_h"], keep="first")
               .sort("horizon_h").select("horizon_h", "drift_bps").iter_rows()))
    best = (
        table.sort("total_is_bps")
        .group_by("aum", maintain_order=False)
        .first()
        .sort("total_is_bps")
        .select("aum", "horizon_h", "total_is_bps")
    )
    print("\n=== optimal execution horizon per AUM ===")
    print(best)
    table.write_csv("reports/execution_frontier.csv")
    print("-> reports/execution_frontier.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
