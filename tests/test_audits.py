"""Audit tests on synthetic panels — every audit must catch its target defect
and stay silent on clean data."""

from datetime import UTC, datetime, timedelta

import polars as pl

from alphalab.data import audits
from tests.conftest import make_klines


def test_clean_panel_has_no_findings(t0):
    kl = make_klines("AAAUSDT", t0, 100)
    assert audits.find_duplicates(kl).is_empty()
    assert audits.find_gaps(kl, "1h").is_empty()
    assert audits.check_grid_alignment(kl, "1h").is_empty()
    assert audits.find_placeholder_runs(kl).is_empty()


def test_duplicates_detected(t0):
    kl = make_klines("AAAUSDT", t0, 10)
    dup = pl.concat([kl, kl.slice(3, 1)])
    findings = audits.find_duplicates(dup)
    assert findings.height == 1
    assert findings["open_time"][0] == t0 + timedelta(hours=3)


def test_gap_detected_with_missing_bar_count(t0):
    kl = make_klines("AAAUSDT", t0, 50)
    holey = kl.filter(~pl.col("open_time").is_in([t0 + timedelta(hours=h) for h in (10, 11, 12)]))
    findings = audits.find_gaps(holey, "1h")
    assert findings.height == 1
    assert findings["missing_bars"][0] == 3


def test_late_listing_is_not_a_gap(t0):
    early = make_klines("AAAUSDT", t0, 100)
    late = make_klines("BBBUSDT", t0 + timedelta(hours=60), 40)
    findings = audits.find_gaps(pl.concat([early, late]), "1h")
    assert findings.is_empty()


def test_off_grid_bar_detected(t0):
    kl = make_klines("AAAUSDT", t0, 5)
    shifted = kl.with_columns(pl.col("open_time") + pl.duration(minutes=7))
    assert audits.check_grid_alignment(shifted, "1h").height == 5


def test_placeholder_run_detected_ftt_pattern(t0):
    """The verified FTTUSDT signature: real bars, then years of frozen-price
    zero-volume bars."""
    real = make_klines("FTTUSDT", t0, 100)
    fake = make_klines("FTTUSDT", t0 + timedelta(hours=100), 500, volume=0.0, count=0)
    findings = audits.find_placeholder_runs(pl.concat([real, fake]))
    assert findings.height == 1
    assert findings["run_len"][0] == 500
    assert findings["run_start"][0] == t0 + timedelta(hours=100)


def test_short_illiquid_stretch_not_flagged(t0):
    """A few quiet bars on a thin contract are market reality, not vendor rot."""
    real = make_klines("THINUSDT", t0, 50)
    quiet = make_klines("THINUSDT", t0 + timedelta(hours=50), 5, volume=0.0, count=0)
    more_real = make_klines("THINUSDT", t0 + timedelta(hours=55), 50)
    findings = audits.find_placeholder_runs(pl.concat([real, quiet, more_real]))
    assert findings.is_empty()


def _funding(symbol: str, times: list[datetime], interval_h: int = 8) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(times),
            "calc_time": times,
            "funding_interval_hours": [interval_h] * len(times),
            "last_funding_rate": [0.0001] * len(times),
        }
    ).with_columns(pl.col("calc_time").dt.replace_time_zone("UTC"))


def test_funding_on_grid_ok():
    times = [datetime(2023, 1, 1, h, tzinfo=UTC) for h in (0, 8, 16)]
    assert audits.check_funding_grid(_funding("AAAUSDT", times)).is_empty()


def test_funding_off_grid_flagged():
    times = [datetime(2023, 1, 1, 0, tzinfo=UTC), datetime(2023, 1, 1, 9, 30, tzinfo=UTC)]
    findings = audits.check_funding_grid(_funding("AAAUSDT", times))
    assert findings.height == 1


def test_funding_respects_declared_interval_change():
    """A contract that switches to 4h funding must be judged by its own column,
    not by an assumed 8h grid."""
    times = [datetime(2023, 1, 1, h, tzinfo=UTC) for h in (0, 4, 8, 12)]
    assert audits.check_funding_grid(_funding("AAAUSDT", times, interval_h=4)).is_empty()


def test_funding_small_jitter_tolerated():
    times = [datetime(2023, 1, 1, 8, 0, 30, tzinfo=UTC)]  # 30s late
    assert audits.check_funding_grid(_funding("AAAUSDT", times)).is_empty()
