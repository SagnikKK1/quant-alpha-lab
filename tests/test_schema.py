"""Header-seam and type parsing tests (the 2022-01 seam, ms timestamps)."""

from datetime import UTC, datetime

from alphalab.data.schema import parse_funding, parse_klines

KLINE_ROW = (
    "1609459200000,28923.63,29031.34,28690.17,28995.13,2311.812,"
    "1609462799999,66768142.34,31421,1156.101,33402108.43,0"
)
KLINE_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,"
    "count,taker_buy_volume,taker_buy_quote_volume,ignore"
)


def test_headerless_pre_2022_file_parses():
    df = parse_klines(f"{KLINE_ROW}\n".encode(), "BTCUSDT")
    assert df.height == 1
    assert df["symbol"][0] == "BTCUSDT"
    # 1609459200000 ms == 2021-01-01 00:00:00 UTC
    assert df["open_time"][0] == datetime(2021, 1, 1, tzinfo=UTC)
    assert df["count"][0] == 31421


def test_headered_post_2022_file_parses_identically():
    headerless = parse_klines(f"{KLINE_ROW}\n".encode(), "BTCUSDT")
    headered = parse_klines(f"{KLINE_HEADER}\n{KLINE_ROW}\n".encode(), "BTCUSDT")
    assert headerless.equals(headered)


def test_funding_keeps_interval_hours_column():
    csv = b"calc_time,funding_interval_hours,last_funding_rate\n1735689600000,8,0.0001\n"
    df = parse_funding(csv, "BTCUSDT")
    assert df["funding_interval_hours"][0] == 8
    assert df["last_funding_rate"][0] == 0.0001
    assert df["calc_time"][0] == datetime(2025, 1, 1, tzinfo=UTC)


def test_funding_headerless_variant():
    csv = b"1735689600000,4,-0.0002\n"
    df = parse_funding(csv, "XUSDT")
    assert df["funding_interval_hours"][0] == 4
    assert df["last_funding_rate"][0] == -0.0002
