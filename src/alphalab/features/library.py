"""Feature library. Every entry ships with its economic hypothesis —
a harness without priors is data-mining with extra steps.

The library starts deliberately tiny: Week-2 work registers the real ~20-40.
These two exist so the causal validator has honest features to certify and
the pipeline has something to move end-to-end.
"""

from __future__ import annotations

import polars as pl

from alphalab.features.base import FeatureSpec


def _momentum(hours: int):
    def fn(panel: pl.DataFrame) -> pl.DataFrame:
        return (
            panel.sort("symbol", "open_time")
            .with_columns(
                (pl.col("close") / pl.col("close").shift(hours) - 1.0)
                .over("symbol")
                .alias("value")
            )
            .drop_nulls("value")
            .select("symbol", pl.col("open_time").alias("ts"), "value")
        )

    return fn


MOMENTUM_24H = FeatureSpec(
    name="mom_24h",
    family="momentum",
    hypothesis=(
        "Cross-sectional continuation over ~1 day: under-reaction to "
        "coin-specific news plus herding of trend-following retail flow; the "
        "counterparty is the mean-reversion/liquidity provider who is paid in "
        "chop and pays in trends. Crowded — expected to be weak after costs."
    ),
    timestamp_rule="knowable at close of bar t (uses closes of bars t-24..t)",
    lookback_bars=24,
    fn=_momentum(24),
)

MOMENTUM_7D = FeatureSpec(
    name="mom_7d",
    family="momentum",
    hypothesis=(
        "Slower continuation at the weekly horizon; same under-reaction story "
        "with slower-moving allocator flows. Less turnover, so more likely to "
        "survive costs than fast momentum."
    ),
    timestamp_rule="knowable at close of bar t (uses closes of bars t-168..t)",
    lookback_bars=168,
    fn=_momentum(168),
)

LIBRARY: list[FeatureSpec] = [MOMENTUM_24H, MOMENTUM_7D]
