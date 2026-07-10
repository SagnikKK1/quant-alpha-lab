"""Canonical content hash: order-invariant, value-sensitive, byte-stable."""

import polars as pl

from alphalab.data.store import canonical_hash
from tests.conftest import make_klines


def test_hash_invariant_to_row_order(t0):
    kl = make_klines("AAAUSDT", t0, 50)
    shuffled = kl.sample(fraction=1.0, shuffle=True, seed=7)
    keys = ["symbol", "open_time"]
    assert canonical_hash(kl, keys) == canonical_hash(shuffled, keys)


def test_hash_invariant_to_column_order(t0):
    kl = make_klines("AAAUSDT", t0, 10)
    reordered = kl.select(sorted(kl.columns, reverse=True))
    keys = ["symbol", "open_time"]
    assert canonical_hash(kl, keys) == canonical_hash(reordered, keys)


def test_hash_sensitive_to_any_value_change(t0):
    kl = make_klines("AAAUSDT", t0, 50)
    tweaked = kl.with_columns(
        pl.when(pl.arange(0, pl.len()) == 25)
        .then(pl.col("close") + 1e-9)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    keys = ["symbol", "open_time"]
    assert canonical_hash(kl, keys) != canonical_hash(tweaked, keys)


def test_hash_stable_across_parquet_roundtrip(tmp_path, t0):
    """The reproducibility gate's core claim: content hash survives a
    write/read cycle even though parquet bytes may differ."""
    kl = make_klines("AAAUSDT", t0, 100)
    p = tmp_path / "x.parquet"
    kl.write_parquet(p, compression="zstd")
    back = pl.read_parquet(p)
    keys = ["symbol", "open_time"]
    assert canonical_hash(kl, keys) == canonical_hash(back, keys)
