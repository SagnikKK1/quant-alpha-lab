"""Cost arithmetic, funding accrual by leg, and engine bookkeeping."""

from datetime import UTC, date, datetime

import polars as pl

from alphalab.backtest.costs import CostModel, funding_pnl
from alphalab.backtest.engine import quantile_weights, run_backtest
from tests.conftest import make_klines

UTC = UTC

EMPTY_FUNDING = pl.DataFrame(
    schema={
        "symbol": pl.String,
        "calc_time": pl.Datetime(time_unit="us", time_zone="UTC"),
        "funding_interval_hours": pl.Int64,
        "last_funding_rate": pl.Float64,
    }
)


def test_quantile_weights_dollar_neutral_and_unit_gross():
    scores = pl.DataFrame(
        {
            "date": [date(2023, 1, 1)] * 10,
            "symbol": [f"S{i}" for i in range(10)],
            "score": [float(i) for i in range(10)],
        }
    )
    w = quantile_weights(scores, quantile=0.2)
    assert abs(w["weight"].sum()) < 1e-12  # dollar neutral
    assert abs(w["weight"].abs().sum() - 1.0) < 1e-12  # gross leverage 1
    top = w.filter(pl.col("weight") > 0)["symbol"].to_list()
    assert set(top) == {"S9", "S8"}  # highest scores long


def test_linear_cost_arithmetic(t0):
    """One symbol long 0.5 from flat: cost = 0.5 * (fee+spread)."""
    panel = pl.concat([make_klines("AAAUSDT", t0, 24 * 5),
                       make_klines("BBBUSDT", t0, 24 * 5)])
    weights = pl.DataFrame(
        {"date": [date(2023, 1, 2)] * 2, "symbol": ["AAAUSDT", "BBBUSDT"],
         "weight": [0.5, -0.5]}
    )
    cm = CostModel(taker_fee_bps=5.0, half_spread_bps=1.0, impact_coeff=0.0)
    result = run_backtest(panel, weights, EMPTY_FUNDING, cm)
    day = result.pnl.filter(pl.col("date") == date(2023, 1, 2))
    assert abs(day["cost"][0] - 1.0 * 6.0 / 1e4) < 1e-12  # turnover 1.0 * 6bps
    assert abs(day["turnover"][0] - 1.0) < 1e-12


def test_funding_sign_long_pays_short_receives():
    positions = pl.DataFrame(
        {
            "date": [date(2023, 1, 2)] * 2,
            "symbol": ["LONGY", "SHORTY"],
            "weight": [0.5, -0.5],
            "exec_date": [date(2023, 1, 3)] * 2,
        }
    )
    funding = pl.DataFrame(
        {
            "symbol": ["LONGY", "SHORTY"],
            "calc_time": [datetime(2023, 1, 3, 8, tzinfo=UTC)] * 2,
            "funding_interval_hours": [8, 8],
            "last_funding_rate": [0.0001, 0.0001],  # positive funding
        }
    )
    out = funding_pnl(positions, funding).sort("symbol")
    long_pnl = out.filter(pl.col("symbol") == "LONGY")["funding_pnl"][0]
    short_pnl = out.filter(pl.col("symbol") == "SHORTY")["funding_pnl"][0]
    assert long_pnl == -0.5 * 0.0001  # long pays
    assert short_pnl == +0.5 * 0.0001  # short receives


def test_funding_event_at_midnight_belongs_to_previous_holder():
    """A 00:00 funding event settles the position held INTO midnight, not the
    one entered at that instant."""
    positions = pl.DataFrame(
        {
            "date": [date(2023, 1, 2)],
            "symbol": ["AAAUSDT"],
            "weight": [1.0],
            "exec_date": [date(2023, 1, 3)],
        }
    )
    fund_midnight_next = pl.DataFrame(
        {
            "symbol": ["AAAUSDT"],
            # midnight at the END of the holding day -> belongs to Jan 3 holder
            "calc_time": [datetime(2023, 1, 4, 0, tzinfo=UTC)],
            "funding_interval_hours": [8],
            "last_funding_rate": [0.001],
        }
    )
    out = funding_pnl(positions, fund_midnight_next)
    assert out["funding_pnl"][0] == -0.001  # counted for the Jan-3 holder

    fund_midnight_entry = fund_midnight_next.with_columns(
        pl.Series("calc_time", [datetime(2023, 1, 3, 0, tzinfo=UTC)]).dt.replace_time_zone(
            None
        ).dt.replace_time_zone("UTC")
    )
    out2 = funding_pnl(positions, fund_midnight_entry)
    assert out2["funding_pnl"][0] == 0.0  # entry-instant event not ours


def test_impact_scales_with_sqrt_participation(t0):
    """Doubling AUM must raise impact cost by sqrt(2) exactly (linear part off)."""
    wiggle = (1 + 0.01 * (pl.int_range(pl.len()) % 2)).alias("_w")
    panel = pl.concat([make_klines("AAAUSDT", t0, 24 * 5),
                       make_klines("BBBUSDT", t0, 24 * 5)]).with_columns(
        (pl.col("close") * wiggle).alias("close")  # nonzero sigma for impact
    )
    weights = pl.DataFrame(
        {"date": [date(2023, 1, 2)] * 2, "symbol": ["AAAUSDT", "BBBUSDT"],
         "weight": [0.5, -0.5]}
    )
    costs = []
    for aum in (100_000.0, 200_000.0):
        cm = CostModel(taker_fee_bps=0.0, half_spread_bps=0.0, impact_coeff=1.0,
                       aum_usd=aum)
        r = run_backtest(panel, weights, EMPTY_FUNDING, cm)
        costs.append(r.pnl.filter(pl.col("date") == date(2023, 1, 2))["cost"][0])
    assert costs[0] > 0
    assert abs(costs[1] / costs[0] - 2**0.5) < 1e-9


def test_exit_turnover_is_charged(t0):
    """Holding then going flat: the closing trade pays linear costs too."""
    panel = pl.concat([make_klines("AAAUSDT", t0, 24 * 6),
                       make_klines("BBBUSDT", t0, 24 * 6)])
    weights = pl.DataFrame(
        {
            "date": [date(2023, 1, 2)] * 2 + [date(2023, 1, 3)] * 2,
            "symbol": ["AAAUSDT", "BBBUSDT"] * 2,
            "weight": [0.5, -0.5, 0.0, 0.0],
        }
    ).filter(pl.col("weight") != 0.0)  # day-2 only: day-3 weights absent = flat
    cm = CostModel(taker_fee_bps=10.0, half_spread_bps=0.0, impact_coeff=0.0)
    result = run_backtest(panel, weights, EMPTY_FUNDING, cm)
    exit_day = result.pnl.filter(pl.col("date") == date(2023, 1, 3))
    assert exit_day.height == 1
    assert abs(exit_day["cost"][0] - 1.0 * 10.0 / 1e4) < 1e-12


def test_gross_leverage_reported(t0):
    panel = pl.concat([make_klines("AAAUSDT", t0, 24 * 5),
                       make_klines("BBBUSDT", t0, 24 * 5)])
    weights = pl.DataFrame(
        {"date": [date(2023, 1, 2)] * 2, "symbol": ["AAAUSDT", "BBBUSDT"],
         "weight": [0.5, -0.5]}
    )
    r = run_backtest(panel, weights, EMPTY_FUNDING,
                     CostModel(0.0, 0.0, 0.0))
    assert abs(r.pnl["gross_leverage"][0] - 1.0) < 1e-12
