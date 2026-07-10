"""Abdi-Ranaldo (hourly) half-spread estimation + data-driven cost wiring."""

from datetime import date, timedelta

import numpy as np
import polars as pl

from alphalab.backtest.costs import CostModel, load_fee_schedule
from alphalab.backtest.engine import run_backtest
from alphalab.backtest.spreads import abdi_ranaldo_half_spread
from tests.conftest import make_klines

EMPTY_FUNDING = pl.DataFrame(
    schema={
        "symbol": pl.String,
        "calc_time": pl.Datetime(time_unit="us", time_zone="UTC"),
        "funding_interval_hours": pl.Int64,
        "last_funding_rate": pl.Float64,
    }
)


def _bounce_panel(symbol, t0, n_days, spread_frac, daily_vol=0.02, seed=5):
    """Hourly bars from a mid random walk observed with bid-ask bounce:
    trades alternate bid/ask, so highs sit ~ask and lows ~bid."""
    rng = np.random.default_rng(seed)
    n = n_days * 24
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, daily_vol / np.sqrt(24), n)))
    side = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    half = spread_frac / 2.0
    close = mid * (1 + side * half)
    high = mid * (1 + half)
    low = mid * (1 - half)
    kl = make_klines(symbol, t0, n)
    return kl.with_columns(
        pl.Series("close", close), pl.Series("high", high), pl.Series("low", low)
    )


def test_estimator_recovers_known_spread(t0):
    """40 bps full spread (20 bps half), crypto-level vol: the hourly AR
    estimator should land within ~25% of truth."""
    panel = _bounce_panel("AAAUSDT", t0, 120, spread_frac=0.004)
    est = abdi_ranaldo_half_spread(panel, floor_bps=0.0, cap_bps=1000.0)
    med = est["half_spread_bps"].median()
    assert 15.0 <= med <= 25.0


def test_estimator_resolves_single_digit_bps(t0):
    """The failure mode that killed Corwin-Schultz here: a tight-spread name
    must NOT read ~20bps of vol-noise. True half = 1bp -> estimate < 3bps."""
    panel = _bounce_panel("BTCUSDT", t0, 120, spread_frac=0.0002)
    est = abdi_ranaldo_half_spread(panel, floor_bps=0.0, cap_bps=1000.0)
    assert est["half_spread_bps"].median() < 3.0


def test_estimator_ranks_liquid_vs_illiquid(t0):
    """The point of measuring: a wide-spread alt must cost more than a
    tight-spread major, same vol."""
    tight = _bounce_panel("BTCUSDT", t0, 120, spread_frac=0.0002, seed=1)
    wide = _bounce_panel("ALTUSDT", t0, 120, spread_frac=0.006, seed=2)
    est = abdi_ranaldo_half_spread(pl.concat([tight, wide]), floor_bps=0.0,
                                     cap_bps=1000.0)
    med = est.group_by("symbol").agg(pl.col("half_spread_bps").median())
    btc = med.filter(pl.col("symbol") == "BTCUSDT")["half_spread_bps"][0]
    alt = med.filter(pl.col("symbol") == "ALTUSDT")["half_spread_bps"][0]
    assert alt > 3 * btc


def test_estimator_is_point_in_time(t0):
    """Truncating the future must not change past estimates (same guarantee
    the leakage validator enforces for features)."""
    panel = _bounce_panel("AAAUSDT", t0, 100, spread_frac=0.004)
    cutoff = t0 + timedelta(days=60)
    full = abdi_ranaldo_half_spread(panel)
    trunc = abdi_ranaldo_half_spread(panel.filter(pl.col("open_time") <= cutoff))
    joined = full.filter(pl.col("date") <= cutoff.date() - timedelta(days=1)).join(
        trunc, on=["symbol", "date"], suffix="_t"
    )
    assert (
        (joined["half_spread_bps"] - joined["half_spread_bps_t"]).abs().max() < 1e-12
    )


def test_floor_and_cap_applied(t0):
    panel = _bounce_panel("BTCUSDT", t0, 60, spread_frac=0.00001)  # ~zero spread
    est = abdi_ranaldo_half_spread(panel, floor_bps=0.5)
    assert est["half_spread_bps"].min() >= 0.5


def test_engine_uses_per_name_spreads(t0):
    """Same trade, wide-vs-tight spread inputs -> different charged costs."""
    panel = pl.concat([make_klines("AAAUSDT", t0, 24 * 5),
                       make_klines("BBBUSDT", t0, 24 * 5)])
    weights = pl.DataFrame(
        {"date": [date(2023, 1, 2)] * 2, "symbol": ["AAAUSDT", "BBBUSDT"],
         "weight": [0.5, -0.5]}
    )
    spreads = pl.DataFrame(
        {
            "symbol": ["AAAUSDT", "BBBUSDT"],
            "date": [date(2023, 1, 2)] * 2,
            "half_spread_bps": [1.0, 20.0],
        }
    )
    cm = CostModel(taker_fee_bps=0.0, impact_coeff=0.0)
    r = run_backtest(panel, weights, EMPTY_FUNDING, cm, spreads=spreads)
    day = r.pnl.filter(pl.col("date") == date(2023, 1, 2))
    expected = 0.5 * 1.0 / 1e4 + 0.5 * 20.0 / 1e4
    assert abs(day["cost"][0] - expected) < 1e-12


def test_fee_schedule_applied_by_date(t0):
    panel = pl.concat([make_klines("AAAUSDT", t0, 24 * 5),
                       make_klines("BBBUSDT", t0, 24 * 5)])
    weights = pl.DataFrame(
        {"date": [date(2023, 1, 2)] * 2, "symbol": ["AAAUSDT", "BBBUSDT"],
         "weight": [0.5, -0.5]}
    )
    schedule = pl.DataFrame(
        {"effective_from": [date(2020, 1, 1), date(2023, 1, 2)],
         "taker_bps": [4.0, 5.0]}
    )
    cm = CostModel(half_spread_bps=0.0, impact_coeff=0.0)
    r = run_backtest(panel, weights, EMPTY_FUNDING, cm, fee_schedule=schedule)
    day = r.pnl.filter(pl.col("date") == date(2023, 1, 2))
    assert abs(day["cost"][0] - 1.0 * 5.0 / 1e4) < 1e-12  # post-change rate


def test_cost_multiplier_scales_everything(t0):
    panel = pl.concat([make_klines("AAAUSDT", t0, 24 * 5),
                       make_klines("BBBUSDT", t0, 24 * 5)])
    weights = pl.DataFrame(
        {"date": [date(2023, 1, 2)] * 2, "symbol": ["AAAUSDT", "BBBUSDT"],
         "weight": [0.5, -0.5]}
    )
    base = run_backtest(panel, weights, EMPTY_FUNDING,
                        CostModel(5.0, 1.0, 0.0)).pnl["cost"][0]
    scaled = run_backtest(panel, weights, EMPTY_FUNDING,
                          CostModel(5.0, 1.0, 0.0, cost_multiplier=1.5)).pnl["cost"][0]
    assert abs(scaled / base - 1.5) < 1e-12


def test_repo_fee_registry_loads():
    sched = load_fee_schedule()
    assert sched.height >= 1
    assert sched["taker_bps"][0] == 5.0
