"""Universe builder: the no-lookahead property is THE test here."""

from datetime import date, timedelta

import polars as pl

from alphalab.data.universe import daily_dollar_volume, rolling_universe
from tests.conftest import make_klines


def test_daily_aggregation(t0):
    kl = make_klines("AAAUSDT", t0, 48, volume=100.0, price=10.0)
    dv = daily_dollar_volume(kl)
    assert dv.height == 2
    assert dv["dollar_volume"].to_list() == [24 * 1000.0, 24 * 1000.0]


def test_no_lookahead(t0):
    """A symbol whose volume explodes on day D must not enter the universe
    until D+1 — membership at D is decided with data through D-1 only."""
    n_days = 40
    quiet = make_klines("QUIETUSDT", t0, 24 * n_days, volume=1.0)
    loud = make_klines("LOUDUSDT", t0, 24 * n_days, volume=1000.0)
    # SPIKE trades tiny volume until day 30, then 100x everyone.
    spike_before = make_klines("SPIKEUSDT", t0, 24 * 30, volume=0.5)
    spike_after = make_klines(
        "SPIKEUSDT", t0 + timedelta(days=30), 24 * (n_days - 30), volume=100_000.0
    )
    panel = pl.concat([quiet, loud, spike_before, spike_after])

    members = rolling_universe(panel, top_n=2, lookback_days=5, min_days_live=2)
    spike_day = date(2023, 1, 31)  # t0 is 2023-01-01, so day-30 bars land on Jan 31

    at_spike = members.filter(pl.col("date") == spike_day)["symbol"].to_list()
    day_after = spike_day + timedelta(days=1)
    next_day = members.filter(pl.col("date") == day_after)["symbol"].to_list()

    assert "SPIKEUSDT" not in at_spike, "lookahead: spike-day volume decided spike-day membership"
    assert "SPIKEUSDT" in next_day


def test_min_days_live_blocks_new_listings(t0):
    old = make_klines("OLDUSDT", t0, 24 * 20, volume=10.0)
    new = make_klines("NEWUSDT", t0 + timedelta(days=15), 24 * 5, volume=1_000_000.0)
    members = rolling_universe(pl.concat([old, new]), top_n=2, lookback_days=5, min_days_live=7)
    newest_new = members.filter(pl.col("symbol") == "NEWUSDT")
    if not newest_new.is_empty():
        # NEWUSDT listed on day 15; must not appear before day 15+7
        assert newest_new["date"].min() >= date(2023, 1, 16) + timedelta(days=7)


def test_rank_is_dense_and_capped(t0):
    panel = pl.concat(
        [make_klines(f"S{i}USDT", t0, 24 * 20, volume=float(10 * (i + 1))) for i in range(6)]
    )
    members = rolling_universe(panel, top_n=3, lookback_days=5, min_days_live=2)
    per_day = members.group_by("date").len()
    assert (per_day["len"] <= 3).all()
    assert (members["rank"] >= 1).all() and (members["rank"] <= 3).all()
