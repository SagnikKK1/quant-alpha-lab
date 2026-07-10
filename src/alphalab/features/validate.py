"""The leakage detector: mechanical causality validation by truncation.

Claim being tested: a feature's value at information timestamp ``ts`` is
computable from data with open_time <= ts. Test: delete everything after a
cutoff T and recompute — every value with ts <= T must be IDENTICAL to the
full-panel value. Any feature peeking past its own timestamp (forward
returns, centered rolling windows, full-sample normalization) changes under
truncation and is rejected. No model in the loop, no statistical judgment —
a feature either survives amputation of the future or it doesn't.

This is the CI half of the handoff's leakage-test pair; the model-side
canary ("a model fed the leak must light it up") lands with the model stage.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from alphalab.features.base import FeatureSpec


class LeakageError(AssertionError):
    pass


def causality_violations(
    spec: FeatureSpec,
    panel: pl.DataFrame,
    cutoffs: list[datetime],
    rel_tol: float = 1e-12,
) -> pl.DataFrame:
    """Rows where truncating the panel at a cutoff changed (or destroyed) a
    feature value with ts <= cutoff. Empty frame = causal at these cutoffs.
    """
    full = spec.compute(panel)
    violations = []
    for cutoff in cutoffs:
        truncated_panel = panel.filter(pl.col("open_time") <= cutoff)
        trunc = spec.compute(truncated_panel)
        joined = (
            full.filter(pl.col("ts") <= cutoff)
            .join(trunc, on=["symbol", "ts"], how="left", suffix="_trunc")
            .with_columns(
                (
                    (pl.col("value") - pl.col("value_trunc")).abs()
                    > rel_tol * pl.max_horizontal(pl.col("value").abs(), pl.lit(1.0))
                )
                .fill_null(True)  # value vanished under truncation == leak
                .alias("changed")
            )
            .filter(pl.col("changed"))
            .with_columns(pl.lit(cutoff).alias("cutoff"))
            .select("symbol", "ts", "value", "value_trunc", "cutoff")
        )
        if not joined.is_empty():
            violations.append(joined)
    if violations:
        return pl.concat(violations)
    return pl.DataFrame(
        schema={
            "symbol": pl.String,
            "ts": full.schema["ts"],
            "value": pl.Float64,
            "value_trunc": pl.Float64,
            "cutoff": pl.Datetime(time_unit="us", time_zone="UTC"),
        }
    )


def default_cutoffs(panel: pl.DataFrame, n: int = 5) -> list[datetime]:
    """Evenly spaced interior cutoffs (never the panel edge, where truncation
    is a no-op)."""
    times = panel.select(pl.col("open_time").unique().sort()).to_series()
    if len(times) < 3:
        raise ValueError("panel too short to choose interior cutoffs")
    idx = [int(len(times) * (i + 1) / (n + 1)) for i in range(n)]
    return [times[i] for i in idx]


def assert_causal(spec: FeatureSpec, panel: pl.DataFrame, n_cutoffs: int = 5) -> None:
    """Raise LeakageError if the feature fails the truncation test."""
    found = causality_violations(spec, panel, default_cutoffs(panel, n_cutoffs))
    if not found.is_empty():
        raise LeakageError(
            f"feature '{spec.name}' leaks: {found.height} values changed when "
            f"the future was truncated (declared rule: {spec.timestamp_rule})\n"
            f"{found.head(5)}"
        )
