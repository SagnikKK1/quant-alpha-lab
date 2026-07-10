"""THE timing-convention test (HANDOFF §2): signals from bar-t close trade at
bar-(t+1) open — the engine must be UNABLE to fill at the signal bar's close.
Construction: decision-day close and next-day open differ hugely; if any
fill/P&L reflects the decision close, the engine is broken.
"""

from datetime import date, timedelta

import polars as pl

from alphalab.backtest.costs import CostModel
from alphalab.backtest.engine import run_backtest
from alphalab.backtest.frames import daily_frame, execution_returns
from tests.conftest import make_klines

EMPTY_FUNDING = pl.DataFrame(
    schema={
        "symbol": pl.String,
        "calc_time": pl.Datetime(time_unit="us", time_zone="UTC"),
        "funding_interval_hours": pl.Int64,
        "last_funding_rate": pl.Float64,
    }
)

FREE = CostModel(taker_fee_bps=0.0, half_spread_bps=0.0, impact_coeff=0.0)


def _two_symbol_panel(t0):
    """AAA jumps: day-2 close 10 -> day-3 open 20 -> day-4 open 30.
    BBB is flat at 10 throughout."""
    a = make_klines("AAAUSDT", t0, 24 * 6, price=10.0)
    day = pl.col("open_time").dt.date()
    a = a.with_columns(
        pl.when(day >= date(2023, 1, 3)).then(20.0).otherwise(pl.col("open")).alias("open"),
        pl.when(day >= date(2023, 1, 4))
        .then(30.0)
        .when(day >= date(2023, 1, 3))
        .then(20.0)
        .otherwise(pl.col("close"))
        .alias("close"),
    )
    # make opens follow closes properly per day: day4+ open 30
    a = a.with_columns(
        pl.when(day >= date(2023, 1, 4)).then(30.0).otherwise(pl.col("open")).alias("open")
    )
    b = make_klines("BBBUSDT", t0, 24 * 6, price=10.0)
    return pl.concat([a, b])


def test_fill_is_next_day_open_not_signal_close(t0):
    panel = _two_symbol_panel(t0)
    daily = daily_frame(panel)
    row = daily.filter(
        (pl.col("symbol") == "AAAUSDT") & (pl.col("date") == date(2023, 1, 2))
    )
    # decision info on Jan 2: close still 10; the fill for that decision: 20
    assert row["close"][0] == 10.0
    assert row["exec_open"][0] == 20.0
    assert row["exec_date"][0] == date(2023, 1, 3)


def test_pnl_accrues_from_exec_open_not_decision_close(t0):
    """Long AAA decided Jan 2. If the engine filled at the Jan-2 close (10),
    the day's return would be +100% (10->20). Correct: filled at 20, earns
    20->30 = +50%."""
    panel = _two_symbol_panel(t0)
    weights = pl.DataFrame(
        {"date": [date(2023, 1, 2)] * 2, "symbol": ["AAAUSDT", "BBBUSDT"],
         "weight": [0.5, -0.5]}
    )
    result = run_backtest(panel, weights, EMPTY_FUNDING, FREE)
    day = result.pnl.filter(pl.col("date") == date(2023, 1, 2))
    assert abs(day["gross_ret"][0] - 0.5 * 0.5) < 1e-12  # 0.5 weight * 50%
    # and the recorded fill price is the next-day open, definitionally
    fills = result.positions.filter(pl.col("symbol") == "AAAUSDT")
    assert fills["exec_open"][0] == 20.0


def test_execution_returns_never_use_decision_day_prices(t0):
    panel = _two_symbol_panel(t0)
    rets = execution_returns(daily_frame(panel))
    r = rets.filter((pl.col("symbol") == "AAAUSDT") & (pl.col("date") == date(2023, 1, 2)))
    assert abs(r["exec_ret"][0] - 0.5) < 1e-12  # 20 -> 30, not 10 -> anything


def test_last_decision_date_cannot_execute(t0):
    """No next-day open exists for the final date — those decisions must be
    dropped, not silently filled at the close."""
    panel = make_klines("AAAUSDT", t0, 24 * 3)
    rets = execution_returns(daily_frame(panel))
    assert rets["date"].max() < date(2023, 1, 3) - timedelta(days=1) + timedelta(days=1)
    assert date(2023, 1, 3) not in rets["date"].to_list()
