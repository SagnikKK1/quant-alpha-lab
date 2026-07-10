"""Execution cost model: taker fee + half-spread + square-root impact,
plus funding accrual by position leg.

All components are per-unit-of-traded-notional except funding, which accrues
on HELD notional at each funding timestamp. Longs pay positive funding,
shorts receive it — first-order P&L for a perp long-short and the single most
common omission in retail crypto backtests (HANDOFF §2).

The impact coefficient defaults to the literature square-root law; §10 of the
handoff replaces it with a tape-calibrated venue estimate + CI later. The
model is deliberately a pure function of PIT-known quantities: sigma and
dollar_volume are decision-date values, never future ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


def load_fee_schedule(path: Path | None = None) -> pl.DataFrame:
    """Dated fee schedule from registry/fees.csv -> (effective_from,
    taker_bps). Applied by as-of join on decision date, so a future
    documented fee change is one CSV row, not a code change."""
    if path is None:
        path = Path(__file__).resolve().parents[3] / "registry" / "fees.csv"
    return (
        pl.read_csv(path)
        .with_columns(pl.col("effective_from").str.to_date())
        .select("effective_from", "taker_bps")
        .sort("effective_from")
    )


@dataclass(frozen=True)
class CostModel:
    """Scalar fields are FALLBACKS, not facts — production paths feed
    per-(symbol, date) columns: fees from registry/fees.csv (sourced, dated),
    half-spreads from the Corwin-Schultz estimator (spreads.py). The scalars
    exist for unit tests and for scenarios where data-driven inputs are
    deliberately overridden. cost_multiplier is the "costs ±50%" sensitivity
    axis from the handoff — it scales fee+spread+impact together.
    """

    taker_fee_bps: float = 5.0  # fallback; see registry/fees.csv for source
    half_spread_bps: float = 1.0  # fallback; see spreads.corwin_schultz_half_spread
    impact_coeff: float = 1.0  # Y in Y * sigma_daily * sqrt(participation)
    aum_usd: float = 100_000.0
    cost_multiplier: float = 1.0

    def trade_cost_return(
        self,
        traded_weight: pl.Expr,
        sigma: pl.Expr,
        dollar_volume: pl.Expr,
        taker_fee_bps: pl.Expr | None = None,
        half_spread_bps: pl.Expr | None = None,
    ) -> pl.Expr:
        """Cost of trading |Δw| of AUM in one name, as a portfolio return.

        fee+spread are linear in traded notional; impact is
        Y * sigma * sqrt(Q/V) with Q = |Δw|*AUM and V = the name's daily
        dollar volume (decision date). Result = cost fraction of AUM.
        """
        fee = (
            taker_fee_bps.fill_null(self.taker_fee_bps)
            if taker_fee_bps is not None
            else pl.lit(self.taker_fee_bps)
        )
        spread = (
            half_spread_bps.fill_null(self.half_spread_bps)
            if half_spread_bps is not None
            else pl.lit(self.half_spread_bps)
        )
        traded = traded_weight.abs()
        participation = (traded * self.aum_usd) / pl.max_horizontal(
            dollar_volume, pl.lit(1.0)
        )
        impact = self.impact_coeff * sigma.fill_null(0.0) * participation.sqrt()
        return self.cost_multiplier * traded * ((fee + spread) / 1e4 + impact)


def funding_pnl(
    positions: pl.DataFrame,
    funding: pl.DataFrame,
) -> pl.DataFrame:
    """Funding P&L per (decision date, symbol) for the holding window.

    positions: (date, symbol, weight, exec_date) — weight held from
      exec_date's open until the next exec date's open (one day for daily
      rebalance).
    funding:   (symbol, calc_time, last_funding_rate).

    A funding event hits the position if it falls inside the holding window
    (exec_date 00:00, next day 00:00]; with daily rebalance that's simply the
    events whose calc_time date == exec_date (00:00 events belong to the
    PREVIOUS day's holder — the position entered at that same instant pays
    from the NEXT event onward; convention documented and tested).
    P&L per event = -weight * funding_rate (long pays positive funding).
    """
    events = funding.with_columns(
        # shift by 1us so a calc_time of exactly 00:00 lands on the prior date
        (pl.col("calc_time") - pl.duration(microseconds=1)).dt.date().alias("exec_date")
    )
    per_day = events.group_by("symbol", "exec_date").agg(
        pl.col("last_funding_rate").sum().alias("funding_sum")
    )
    return (
        positions.join(per_day, on=["symbol", "exec_date"], how="left")
        .with_columns(
            (-pl.col("weight") * pl.col("funding_sum").fill_null(0.0)).alias("funding_pnl")
        )
        .select("date", "symbol", "funding_pnl")
    )
