"""Settlement inference + truncation: the placeholder-bar kill switch."""

from datetime import timedelta

import polars as pl

from alphalab.data import audits, delistings
from tests.conftest import make_klines


def test_infer_settlement_from_ftt_pattern(t0):
    real = make_klines("FTTUSDT", t0, 100)
    fake = make_klines("FTTUSDT", t0 + timedelta(hours=100), 24 * 400, volume=0.0, count=0)
    alive = make_klines("BTCUSDT", t0, 100 + 24 * 400)  # keeps panel_end current
    panel = pl.concat([real, fake, alive])

    inferred = delistings.infer_settlements(panel)
    assert inferred.height == 1
    assert inferred["symbol"][0] == "FTTUSDT"
    assert inferred["settlement_time"][0] == t0 + timedelta(hours=99)


def test_live_symbol_not_inferred_as_delisted(t0):
    panel = make_klines("BTCUSDT", t0, 24 * 60)
    assert delistings.infer_settlements(panel).is_empty()


def test_truncation_removes_placeholders_and_audit_confirms(t0):
    real = make_klines("FTTUSDT", t0, 100)
    fake = make_klines("FTTUSDT", t0 + timedelta(hours=100), 1000, volume=0.0, count=0)
    panel = pl.concat([real, fake])

    registry = pl.DataFrame(
        {
            "symbol": ["FTTUSDT"],
            "settlement_time": [t0 + timedelta(hours=99)],
            "source": ["test"],
        }
    ).with_columns(pl.col("settlement_time").dt.replace_time_zone("UTC"))

    curated = delistings.truncate_at_settlement(panel, registry)
    assert curated.height == 100
    # The CI invariant: no placeholder run survives curation.
    assert audits.find_placeholder_runs(curated).is_empty()


def test_funding_truncated_at_settlement_too(t0):
    """Binance publishes placeholder FUNDING records after settlement as well
    (observed on SRMUSDT: identical funding row count to a live contract).
    Both series must be cut at the same settlement time."""
    times = [t0 + timedelta(hours=8 * i) for i in range(30)]
    funding = pl.DataFrame(
        {
            "symbol": ["SRMUSDT"] * 30,
            "calc_time": times,
            "funding_interval_hours": [8] * 30,
            "last_funding_rate": [0.0001] * 30,
        }
    ).with_columns(pl.col("calc_time").dt.replace_time_zone("UTC"))
    registry = pl.DataFrame(
        {
            "symbol": ["SRMUSDT"],
            "settlement_time": [t0 + timedelta(hours=80)],
            "source": ["test"],
        }
    ).with_columns(pl.col("settlement_time").dt.replace_time_zone("UTC"))

    cut = delistings.truncate_at_settlement(funding, registry, time_col="calc_time")
    assert cut.height == 11  # events at h0..h80 inclusive
    assert cut["calc_time"].max() <= t0 + timedelta(hours=80)


def test_interior_run_inferred_as_exclusion_not_settlement(t0):
    """The ICPUSDT case: delisted mid-2022, RELISTED later. The placeholder
    era is interior, so it must become an exclusion window — and must NOT be
    inferred as a settlement (the symbol is alive today)."""
    real1 = make_klines("ICPUSDT", t0, 100)
    halt = make_klines("ICPUSDT", t0 + timedelta(hours=100), 200, volume=0.0, count=0)
    real2 = make_klines("ICPUSDT", t0 + timedelta(hours=300), 100)
    panel = pl.concat([real1, halt, real2])

    exclusions = delistings.infer_exclusions(panel)
    assert exclusions.height == 1
    assert exclusions["start_time"][0] == t0 + timedelta(hours=100)
    assert exclusions["end_time"][0] == t0 + timedelta(hours=299)
    assert delistings.infer_settlements(panel).is_empty()


def test_drop_exclusion_windows_removes_only_the_window(t0):
    real1 = make_klines("ICPUSDT", t0, 100)
    halt = make_klines("ICPUSDT", t0 + timedelta(hours=100), 200, volume=0.0, count=0)
    real2 = make_klines("ICPUSDT", t0 + timedelta(hours=300), 100)
    other = make_klines("BTCUSDT", t0, 400)
    panel = pl.concat([real1, halt, real2, other])

    exclusions = delistings.infer_exclusions(panel)
    curated = delistings.drop_exclusion_windows(panel, exclusions)
    assert curated.filter(pl.col("symbol") == "ICPUSDT").height == 200
    assert curated.filter(pl.col("symbol") == "BTCUSDT").height == 400
    assert audits.find_placeholder_runs(curated).is_empty()


def test_exclusion_window_extends_to_relist_not_last_placeholder(t0):
    """ICPUSDT reality: placeholder files STOP weeks before trading resumes,
    so the inferred window must run to the relist bar, not the last fake bar."""
    real1 = make_klines("ICPUSDT", t0, 100)
    fake = make_klines("ICPUSDT", t0 + timedelta(hours=100), 200, volume=0.0, count=0)
    # nothing at all for hours 300..399 (no files), then trading resumes
    real2 = make_klines("ICPUSDT", t0 + timedelta(hours=400), 100)
    panel = pl.concat([real1, fake, real2])

    exclusions = delistings.infer_exclusions(panel)
    assert exclusions.height == 1
    assert exclusions["end_time"][0] == t0 + timedelta(hours=399)


def test_gap_audit_waives_registered_exclusion_window(t0):
    """After curation, an exclusion window looks like a hole in the panel;
    the gap audit must waive exactly that hole and nothing else."""
    real1 = make_klines("ICPUSDT", t0, 100)
    real2 = make_klines("ICPUSDT", t0 + timedelta(hours=400), 100)
    curated = pl.concat([real1, real2])
    exclusions = pl.DataFrame(
        {
            "symbol": ["ICPUSDT"],
            "start_time": [t0 + timedelta(hours=100)],
            "end_time": [t0 + timedelta(hours=399)],
            "source": ["test"],
        }
    ).with_columns(
        pl.col("start_time").dt.replace_time_zone("UTC"),
        pl.col("end_time").dt.replace_time_zone("UTC"),
    )
    assert audits.find_gaps(curated, "1h").height == 1
    assert audits.find_gaps(curated, "1h", exclusions).is_empty()

    # a DIFFERENT hole is still reported even with the exclusion registered
    holey = curated.filter(pl.col("open_time") != t0 + timedelta(hours=50))
    assert audits.find_gaps(holey, "1h", exclusions).height == 1


def test_unregistered_symbols_pass_through(t0):
    panel = make_klines("BTCUSDT", t0, 50)
    registry = pl.DataFrame(schema=delistings.REGISTRY_SCHEMA)
    assert delistings.truncate_at_settlement(panel, registry).equals(panel)
