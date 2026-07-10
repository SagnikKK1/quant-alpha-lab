"""Daily-rebalance long-short backtest engine.

Pipeline per decision date D:
  scores(D) -> weights(D)  [rank quantile long-short, dollar neutral]
  fills at exec_open(D+1); position earns exec_open(D+1) -> exec_open(D+2)
  costs on |Δweight| via CostModel; funding accrues on held weight

Simplifications, stated: turnover is |w_D - w_{D-1}| without intraperiod
drift adjustment (equal-weight daily portfolios; second-order), and fills
assume full execution at the open print (the shadow-trading lane exists to
price that assumption).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from alphalab.backtest.costs import CostModel, funding_pnl
from alphalab.backtest.frames import daily_frame, execution_returns


def quantile_weights(scores: pl.DataFrame, quantile: float = 0.2) -> pl.DataFrame:
    """(date, symbol, score) -> dollar-neutral weights: equal-weight long the
    top ``quantile`` of scores, short the bottom, 0.5 gross per side."""
    n_expr = pl.len().over("date")
    k_expr = pl.max_horizontal((n_expr * quantile).floor().cast(pl.Int64), pl.lit(1))
    ranked = scores.with_columns(
        pl.col("score").rank(method="ordinal", descending=True).over("date").alias("_r"),
        k_expr.alias("_k"),
        n_expr.alias("_n"),
    )
    return (
        ranked.with_columns(
            pl.when(pl.col("_r") <= pl.col("_k"))
            .then(0.5 / pl.col("_k"))
            .when(pl.col("_r") > pl.col("_n") - pl.col("_k"))
            .then(-0.5 / pl.col("_k"))
            .otherwise(0.0)
            .alias("weight")
        )
        .filter(pl.col("weight") != 0.0)
        .select("date", "symbol", "weight")
    )


@dataclass(frozen=True)
class BacktestResult:
    pnl: pl.DataFrame  # per decision date: gross, costs, funding, net
    positions: pl.DataFrame  # (date, symbol, weight, exec_date, fill price)

    def summary(self, ann_factor: float = 365.0) -> dict:
        net = self.pnl["net_ret"]
        mu, sd = net.mean(), net.std()
        sharpe = (mu / sd) * ann_factor**0.5 if sd and sd > 0 else float("nan")
        equity = (1.0 + self.pnl["net_ret"]).cum_prod()
        peak = equity.cum_max()
        max_dd = ((equity - peak) / peak).min()
        return {
            "days": self.pnl.height,
            "ann_return": float(mu * ann_factor) if mu is not None else float("nan"),
            "ann_sharpe_net": float(sharpe),
            "max_drawdown": float(max_dd),
            "avg_daily_turnover": float(self.pnl["turnover"].mean()),
            "hit_rate": float((net > 0).mean()),
            "total_cost_drag_ann": float(self.pnl["cost"].mean() * ann_factor),
            "total_funding_ann": float(self.pnl["funding_pnl"].mean() * ann_factor),
        }


def run_backtest(
    klines: pl.DataFrame,
    weights: pl.DataFrame,
    funding: pl.DataFrame,
    cost_model: CostModel | None = None,
) -> BacktestResult:
    """weights: (date, symbol, weight) decided at date's close."""
    cost_model = cost_model or CostModel()
    daily = daily_frame(klines)
    rets = execution_returns(daily)

    pos = weights.join(
        rets.select("symbol", "date", "exec_date", "exec_open", "exec_ret", "sigma",
                    "dollar_volume"),
        on=["date", "symbol"],
        how="inner",  # a decision with no next-day fill can't be executed
    )

    # Turnover vs previous day's weight, per symbol (0 if newly entered/exited):
    # re-index yesterday's weights onto the next CALENDAR decision date. The
    # calendar comes from the panel, not from `weights` — a rebalance to fully
    # flat has no weight rows, but its closing trades must still be charged.
    next_map = rets.select(pl.col("date").unique().sort()).with_columns(
        pl.col("date").shift(-1).alias("next_date")
    )
    prev = (
        pos.select("date", "symbol", "weight")
        .rename({"weight": "w_prev"})
        .join(next_map, on="date", how="left")
        .filter(pl.col("next_date").is_not_null())
        .drop("date")
        .rename({"next_date": "date"})
    )
    pos = (
        pos.join(prev, on=["date", "symbol"], how="full", coalesce=True)
        .filter(pl.col("date").is_not_null())  # last date's w has no successor
        .with_columns(
            pl.col("weight").fill_null(0.0),
            pl.col("w_prev").fill_null(0.0),
        )
        .with_columns((pl.col("weight") - pl.col("w_prev")).alias("dw"))
    )

    pos = pos.with_columns(
        cost_model.trade_cost_return(
            pl.col("dw"), pl.col("sigma"), pl.col("dollar_volume")
        ).alias("cost"),
        (pl.col("weight") * pl.col("exec_ret").fill_null(0.0)).alias("gross_pnl"),
    )

    fpnl = funding_pnl(
        pos.filter(pl.col("weight") != 0.0).select("date", "symbol", "weight", "exec_date"),
        funding,
    )
    pos = pos.join(fpnl, on=["date", "symbol"], how="left").with_columns(
        pl.col("funding_pnl").fill_null(0.0)
    )

    pnl = (
        pos.group_by("date")
        .agg(
            pl.col("gross_pnl").sum().alias("gross_ret"),
            pl.col("cost").sum().alias("cost"),
            pl.col("funding_pnl").sum().alias("funding_pnl"),
            pl.col("dw").abs().sum().alias("turnover"),
            pl.col("weight").abs().sum().alias("gross_leverage"),
        )
        .with_columns(
            (pl.col("gross_ret") - pl.col("cost") + pl.col("funding_pnl")).alias("net_ret")
        )
        .sort("date")
    )
    positions = pos.filter(pl.col("weight") != 0.0).select(
        "date", "symbol", "weight", "exec_date", "exec_open"
    )
    return BacktestResult(pnl=pnl, positions=positions)
