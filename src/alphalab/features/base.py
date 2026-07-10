"""Feature specification: every feature is registered BEFORE evaluation.

A feature is a pure function of the klines panel returning
(symbol, ts, value) where ``ts`` is the INFORMATION timestamp: the moment the
value becomes knowable. The contract enforced by `validate.assert_causal` is
that value(ts) is computable from bars with open_time <= ts and nothing later.

Registration discipline (HANDOFF §2): specs carry an economic hypothesis and
a timestamp rule at creation; discarded features stay in the registry as
trials and count toward the DSR trial budget.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

FEATURE_OUTPUT_COLUMNS = ["symbol", "ts", "value"]

FeatureFn = Callable[[pl.DataFrame], pl.DataFrame]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    hypothesis: str  # one paragraph: why should this alpha exist, who pays it
    timestamp_rule: str  # e.g. "knowable at close of bar t (uses bars <= t)"
    lookback_bars: int  # longest history the feature consumes
    fn: FeatureFn

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        out = self.fn(panel)
        missing = set(FEATURE_OUTPUT_COLUMNS) - set(out.columns)
        if missing:
            raise ValueError(f"feature {self.name} output missing columns {missing}")
        return out.select(FEATURE_OUTPUT_COLUMNS).sort("symbol", "ts")
