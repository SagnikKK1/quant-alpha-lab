"""The FULL pipe, baseline edition (HANDOFF Week 2):

rank-momentum signal -> universe-gated weights -> purged walk-forward OOS
periods -> funding- and cost-aware backtest -> Politis-White block-bootstrap
Sharpe CI -> trial registry entry -> tear sheet.

The baseline fits nothing, but it flows through the same OOS discipline the
ML ladder will use: reported P&L is the concatenation of walk-forward TEST
blocks only (the burn-in before the first test block is discarded), and the
run is logged to the append-only trial registry like every other experiment.

Usage:
    python scripts/run_baseline.py [--lookback 30] [--quantile 0.2]
        [--top-n 30] [--aum 100000]
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import polars as pl

from alphalab.backtest.costs import CostModel
from alphalab.backtest.engine import quantile_weights, run_backtest
from alphalab.data import store
from alphalab.data.universe import rolling_universe
from alphalab.registry.trials import log_trial
from alphalab.report.tearsheet import render_tearsheet
from alphalab.signals.baseline import rank_momentum_scores
from alphalab.validation.bootstrap import sharpe_ci
from alphalab.validation.walkforward import purged_walk_forward

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--quantile", type=float, default=0.2)
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--aum", type=float, default=100_000.0)
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()

    klines = store.scan("klines").collect()
    funding = store.scan("funding").collect()

    universe = rolling_universe(klines, top_n=args.top_n, lookback_days=30)
    scores = rank_momentum_scores(klines, universe, lookback_days=args.lookback)
    weights = quantile_weights(scores, quantile=args.quantile)

    cost_model = CostModel(aum_usd=args.aum)
    result = run_backtest(klines, weights, funding, cost_model)

    # OOS = union of walk-forward test blocks (baseline fits nothing, but the
    # reporting convention must match what the ML arms will be held to).
    decision_dates = result.pnl["date"].unique().sort().to_list()
    folds = purged_walk_forward(
        decision_dates, n_folds=args.n_folds, label_horizon_days=2, min_train_days=180
    )
    oos_start = folds[0].test[0]
    oos = result.pnl.filter(pl.col("date") >= oos_start)

    ci = sharpe_ci(oos["net_ret"].to_numpy(), ann_factor=365.0)
    from alphalab.backtest.engine import BacktestResult

    oos_summary = BacktestResult(pnl=oos, positions=result.positions).summary()
    stats = {**oos_summary, **ci}

    config = {
        "strategy": "rank_momentum_baseline",
        "lookback_days": args.lookback,
        "quantile": args.quantile,
        "universe_top_n": args.top_n,
        "cost_model": {
            "taker_fee_bps": cost_model.taker_fee_bps,
            "half_spread_bps": cost_model.half_spread_bps,
            "impact_coeff": cost_model.impact_coeff,
            "aum_usd": cost_model.aum_usd,
        },
        "n_folds": args.n_folds,
        "label_horizon_days": 2,
        "oos_start": str(oos_start),
        "timing": "signal@D close -> fill@D+1 open -> exit@D+2 open",
    }
    tid = log_trial(config, oos.select("date", "net_ret"), stats)

    png = render_tearsheet(
        oos,
        f"Rank momentum {args.lookback}d — OOS {oos_start} → {oos['date'].max()}"
        f"  (trial {tid})",
        stats,
        REPORTS / f"baseline_mom{args.lookback}_tearsheet.png",
    )

    print(json.dumps({"trial_id": tid, **stats}, indent=2, default=str))
    print(f"tearsheet: {png}")
    print(f"folds: {[(str(f.test[0]), str(f.test[1])) for f in folds]}")
    # Purge sanity, printed for the record:
    for f in folds:
        gap = (f.test[0] - f.train[1]) - timedelta(days=0)
        assert gap.days > 2, "purge violated"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
