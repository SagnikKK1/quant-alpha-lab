"""The leakage-detector pair, data-layer half (HANDOFF §2):

(a) DETECTOR test — a deliberately leaky feature (the forward return) must be
    REJECTED by the causal validator. If this test ever passes silently, the
    canary is dead and nothing downstream can be trusted.
(b) Honest features must pass, including ones that merely LOOK suspicious
    (long lookbacks), so the detector isn't rejecting everything.

The model-side half ("a model fed the leak lights it up") arrives with the
model ladder.
"""

import numpy as np
import polars as pl
import pytest

from alphalab.features.base import FeatureSpec
from alphalab.features.library import LIBRARY
from alphalab.features.validate import (
    LeakageError,
    assert_causal,
    causality_violations,
    default_cutoffs,
)
from tests.conftest import make_klines


@pytest.fixture
def panel(t0):
    """Two symbols, 300 bars, with non-trivial price paths so returns vary."""
    rng = np.random.default_rng(11)
    frames = []
    for sym in ("AAAUSDT", "BBBUSDT"):
        kl = make_klines(sym, t0, 300)
        walk = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
        kl = kl.with_columns(pl.Series("close", walk))
        frames.append(kl)
    return pl.concat(frames)


FORWARD_RETURN = FeatureSpec(
    name="fwd_ret_24h_CANARY",
    family="canary",
    hypothesis="none — this is the deliberately leaky control",
    timestamp_rule="CLAIMS close of bar t but actually uses bar t+24",
    lookback_bars=0,
    fn=lambda p: (
        p.sort("symbol", "open_time")
        .with_columns(
            (pl.col("close").shift(-24) / pl.col("close") - 1.0).over("symbol").alias("value")
        )
        .drop_nulls("value")
        .select("symbol", pl.col("open_time").alias("ts"), "value")
    ),
)

CENTERED_ZSCORE = FeatureSpec(
    name="centered_zscore_CANARY",
    family="canary",
    hypothesis="none — subtle leak: normalization uses the full sample",
    timestamp_rule="CLAIMS close of bar t but normalizes with full-sample stats",
    lookback_bars=0,
    fn=lambda p: (
        p.sort("symbol", "open_time")
        .with_columns(
            ((pl.col("close") - pl.col("close").mean()) / pl.col("close").std())
            .over("symbol")
            .alias("value")
        )
        .select("symbol", pl.col("open_time").alias("ts"), "value")
    ),
)


def test_detector_rejects_forward_return(panel):
    """The blatant leak: tomorrow's return as today's feature."""
    with pytest.raises(LeakageError):
        assert_causal(FORWARD_RETURN, panel)


def test_detector_rejects_full_sample_normalization(panel):
    """The subtle leak that kills real projects: z-scoring with statistics
    computed over the whole sample, future included."""
    with pytest.raises(LeakageError):
        assert_causal(CENTERED_ZSCORE, panel)


def test_library_features_are_causal(panel):
    for spec in LIBRARY:
        assert_causal(spec, panel)


def test_violations_report_names_the_offending_rows(panel):
    found = causality_violations(FORWARD_RETURN, panel, default_cutoffs(panel, 3))
    assert not found.is_empty()
    assert set(found.columns) == {"symbol", "ts", "value", "value_trunc", "cutoff"}
    # every reported violation is at or before its cutoff — the detector only
    # indicts values that CLAIMED to be knowable by then
    assert (found["ts"] <= found["cutoff"]).all()
