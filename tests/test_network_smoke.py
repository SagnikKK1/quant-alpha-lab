"""Live smoke tests against data.binance.vision. Run explicitly:

    pytest -m network

Excluded from the default run and from CI's unit job — these verify the
vendor contract (URLs, checksums, schemas), not our logic.
"""

import pytest

from alphalab.data import binance_bulk, schema

pytestmark = pytest.mark.network


def test_funding_month_downloads_verifies_and_parses(tmp_path):
    url = binance_bulk.monthly_funding_url("BTCUSDT", "2024-01")
    path = binance_bulk.download_verified(url, dest=tmp_path / "f.zip")
    assert path is not None
    df = schema.parse_funding(binance_bulk.read_zip_csv(path), "BTCUSDT")
    assert df.height >= 80  # ~3/day * 31 days, minus jitter
    assert set(df["funding_interval_hours"].unique().to_list()) <= {4, 8}


def test_pre_seam_klines_month_parses_headerless(tmp_path):
    """2021-06 predates the 2022-01 header seam — exercises the headerless path."""
    url = binance_bulk.monthly_kline_url("BTCUSDT", "1h", "2021-06")
    path = binance_bulk.download_verified(url, dest=tmp_path / "k.zip")
    assert path is not None
    df = schema.parse_klines(binance_bulk.read_zip_csv(path), "BTCUSDT")
    assert df.height == 30 * 24
    assert df["open_time"].dt.month().unique().to_list() == [6]


def test_missing_month_returns_none(tmp_path):
    url = binance_bulk.monthly_kline_url("BTCUSDT", "1h", "2019-01")  # pre-listing
    assert binance_bulk.download_verified(url, dest=tmp_path / "x.zip") is None
