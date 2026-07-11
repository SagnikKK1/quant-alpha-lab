"""Arrival-vs-execution study: what does each hour of latency cost, and for
which names is DRIFT (not spread) the dominant implicit cost?

Setup: the backtest's convention fills at the D+1 00:00 open (arrival lag
~0 from the decision print, since crypto is 24/7). Real systems compute,
risk-check and route — so we sweep hypothetical execution delays of
h in {1, 2, 4, 8, 24} hours and reprice the baseline's ACTUAL trades:

    delay_cost(h) = dw * (open(00:00 + h) / open(00:00) - 1)

Positive = the delay hurt (buys filled higher / sells lower). Because the
trades are signal-correlated (momentum buys recent winners), E[cost] > 0 is
expected — this is alpha decay expressed as an execution cost, invisible to
any spread/impact model.

Outputs: portfolio-level drag per delay; per-symbol decomposition of
drift cost vs tick-spread cost at h=1 (who pays via drift, who via spread);
reports/delay_cost_study.csv.
"""

from __future__ import annotations

import polars as pl

from alphalab.backtest.engine import quantile_weights
from alphalab.backtest.spreads import tick_bound_half_spread
from alphalab.data import store
from alphalab.data.universe import rolling_universe
from alphalab.signals.baseline import rank_momentum_scores

DELAYS_H = [1, 2, 4, 8, 24]


def trade_list(weights: pl.DataFrame) -> pl.DataFrame:
    """(date, symbol, dw) with the engine's calendar alignment: dw vs the
    previous decision date's weight, exits included."""
    next_map = weights.select(pl.col("date").unique().sort()).with_columns(
        pl.col("date").shift(-1).alias("next_date")
    )
    prev = (
        weights.rename({"weight": "w_prev"})
        .join(next_map, on="date", how="left")
        .filter(pl.col("next_date").is_not_null())
        .drop("date")
        .rename({"next_date": "date"})
    )
    return (
        weights.join(prev, on=["date", "symbol"], how="full", coalesce=True)
        .filter(pl.col("date").is_not_null())
        .with_columns(
            pl.col("weight").fill_null(0.0), pl.col("w_prev").fill_null(0.0)
        )
        .with_columns((pl.col("weight") - pl.col("w_prev")).alias("dw"))
        .filter(pl.col("dw") != 0.0)
        .select("date", "symbol", "dw")
    )


def main() -> int:
    klines = store.scan("klines").collect()
    universe = rolling_universe(klines, top_n=30, lookback_days=30)
    scores = rank_momentum_scores(klines, universe, lookback_days=30)
    weights = quantile_weights(scores, quantile=0.2)
    trades = trade_list(weights)

    # Hourly opens keyed by (symbol, exec base ts = D+1 00:00 shifted h hours)
    hourly = klines.select(
        "symbol", pl.col("open_time"), pl.col("open").alias("px")
    )
    base = trades.with_columns(
        (pl.col("date").cast(pl.Datetime(time_unit="us"))
         .dt.replace_time_zone("UTC") + pl.duration(days=1)).alias("t0")
    ).join(
        hourly.rename({"open_time": "t0", "px": "px0"}), on=["symbol", "t0"], how="inner"
    )

    per_delay = []
    joined = base
    for h in DELAYS_H:
        joined = joined.with_columns(
            (pl.col("t0") + pl.duration(hours=h)).alias("th")
        ).join(
            hourly.rename({"open_time": "th", "px": f"px{h}"}),
            on=["symbol", "th"],
            how="left",
        ).drop("th")
        joined = joined.with_columns(
            (pl.col("dw") * (pl.col(f"px{h}") / pl.col("px0") - 1.0) * 1e4).alias(
                f"cost{h}_bps_notional"
            )
        )

    total_turnover = joined["dw"].abs().sum()
    for h in DELAYS_H:
        c = joined.drop_nulls(f"cost{h}_bps_notional")
        signed = c[f"cost{h}_bps_notional"].sum()
        per_unit = signed / c["dw"].abs().sum()
        daily = c.group_by("date").agg(
            pl.col(f"cost{h}_bps_notional").sum().alias("d")
        )["d"]
        per_delay.append(
            {
                "delay_h": h,
                "cost_bps_per_unit_traded": round(per_unit, 3),
                "ann_drag_pct": round(daily.mean() / 1e4 * 365 * 100, 2),
                "hit_rate_positive_cost": round((daily > 0).mean(), 3),
            }
        )
    summary = pl.DataFrame(per_delay)
    print("=== portfolio delay cost (signed; + = delay hurts) ===")
    print(summary)

    # Per-symbol at h=1: drift cost vs tick-spread cost
    spreads = tick_bound_half_spread(klines)
    per_sym = (
        joined.drop_nulls("cost1_bps_notional")
        .group_by("symbol")
        .agg(
            (pl.col("cost1_bps_notional").sum() / pl.col("dw").abs().sum()).alias(
                "drift_1h_bps"
            ),
            pl.col("dw").abs().sum().alias("turnover_share"),
            pl.len().alias("n_trades"),
        )
        .join(
            spreads.group_by("symbol").agg(
                pl.col("half_spread_bps").median().alias("spread_bps")
            ),
            on="symbol",
        )
        .with_columns(
            (pl.col("drift_1h_bps") / pl.col("spread_bps")).round(2).alias(
                "drift_over_spread"
            ),
            (pl.col("turnover_share") / total_turnover * 100).round(2).alias(
                "turnover_pct"
            ),
        )
        .sort("drift_1h_bps", descending=True)
        .select("symbol", "drift_1h_bps", "spread_bps", "drift_over_spread",
                "turnover_pct", "n_trades")
        .with_columns(pl.col("drift_1h_bps").round(2))
    )
    pl.Config.set_tbl_rows(50)
    print("\n=== per-symbol: 1h drift cost vs tick spread (traded names) ===")
    print(per_sym.filter(pl.col("n_trades") > 100))

    per_sym.write_csv("reports/delay_cost_study.csv")
    print("-> reports/delay_cost_study.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
