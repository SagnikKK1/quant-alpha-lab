"""Per-symbol, per-date half-spread estimation — measured, not assumed.

Estimator: Abdi & Ranaldo (2017), run at the HOURLY frequency. With c = log
close and eta = midpoint of a bar's log high/low range,
E[(c_t - eta_t)(c_t - eta_{t+1})] = s^2/4: the close carries the bid-ask
bounce (it appears in BOTH factors, so the q_t^2 (s/2)^2 term survives)
while the range midpoints proxy the efficient price and everything else
averages to zero.

Frequency matters — this was tested, not assumed: at DAILY frequency the
window mean's noise (driven by sigma_daily^2 per observation, ~21 obs)
buries s^2/4 for realistic crypto spreads, and Corwin-Schultz's
clamp-negatives-then-average variant has a vol-driven noise floor (~20bps)
that overcharges tight-spread majors ~40x. At hourly frequency a 21-day
window holds ~500 observations of sqrt(24)-smaller vol, and the estimator
resolves single-digit-bps spreads. Names too tight to resolve fall to the
floor, which is the intended conservative behavior.

PIT discipline: the pair (h-1, h) is assigned to hour h; the daily estimate
for decision date D is the last hourly value of D — nothing after D's close.
Crypto trades 24/7, so the equity literature's overnight adjustments are
unnecessary.
"""

from __future__ import annotations

import polars as pl


def tick_bound_half_spread(
    klines: pl.DataFrame,
    window_days: int = 21,
    min_days: int = 5,
    cap_bps: float = 50.0,
) -> pl.DataFrame:
    """(symbol, date, half_spread_bps) = tick_size / (2 * price) — the
    STRUCTURAL spread model, and the primary one.

    Validated against real Binance bookTicker quotes (2023-06..2024-02,
    reports/spread_validation.csv): for every sampled liquid perp the median
    quoted spread equals exactly ONE TICK — BTC 0.019bps, DOGE 0.81bps,
    SAND 1.67bps, XTZ 7.6bps, CRV 12.0bps all match tick/(2*price) to ~3
    decimals. On this venue the spread is set by the price grid, not by a
    statistical process; the Abdi-Ranaldo estimator (below) is retained as a
    stress diagnostic, not the charged cost (its vol-noise floor overcharged
    majors ~100x, which the same validation exposed).

    Tick size is inferred PIT from our own data: the minimum nonzero
    |close-to-close| increment over a trailing window (hourly bars, so ~500
    samples — an active name touches adjacent ticks many times a day).
    Adapts within days when the venue re-tiers a symbol's tick.
    """
    hourly = (
        klines.sort("symbol", "open_time")
        .with_columns(
            pl.col("close").diff().abs().over("symbol").alias("_dp"),
        )
        .with_columns(pl.when(pl.col("_dp") > 0).then(pl.col("_dp")).alias("_tick"))
        .with_columns(
            pl.col("_tick")
            .rolling_min(window_size=window_days * 24, min_samples=min_days * 24)
            .over("symbol")
            .alias("_tick_est")
        )
    )
    return (
        hourly.drop_nulls("_tick_est")
        .with_columns(
            (pl.col("_tick_est") / (2.0 * pl.col("close")) * 1e4)
            .clip(upper_bound=cap_bps)
            .alias("half_spread_bps"),
            pl.col("open_time").dt.date().alias("date"),
        )
        .group_by("symbol", "date", maintain_order=True)
        .agg(pl.col("half_spread_bps").last())
        .select("symbol", "date", "half_spread_bps")
    )


def abdi_ranaldo_half_spread(
    klines: pl.DataFrame,
    window_days: int = 21,
    min_days: int = 10,
    floor_bps: float = 0.5,
    cap_bps: float = 50.0,
) -> pl.DataFrame:
    """(symbol, date, half_spread_bps) from hourly klines.

    floor_bps: minimum charged half-spread — a stated conservative
    assumption (liquid majors' true half-spread is a fraction of a bp;
    charging literally zero would flatter high-turnover strategies).
    cap_bps: winsorizes blow-ups on thin names. Both are sensitivity knobs.
    """
    hourly = (
        klines.sort("symbol", "open_time")
        .with_columns(
            pl.col("high").log().alias("lh"),
            pl.col("low").log().alias("ll"),
            pl.col("close").log().alias("c"),
        )
        .with_columns(((pl.col("lh") + pl.col("ll")) / 2.0).alias("eta"))
    )
    prev_c = pl.col("c").shift(1).over("symbol")
    prev_eta = pl.col("eta").shift(1).over("symbol")
    # x_h = (c_{h-1} - eta_{h-1}) (c_{h-1} - eta_h); E[x] = s^2 / 4
    x = (prev_c - prev_eta) * (prev_c - pl.col("eta"))

    return (
        hourly.with_columns(x.alias("_x"))
        .with_columns(
            (
                pl.col("_x")
                .rolling_mean(window_size=window_days * 24, min_samples=min_days * 24)
                .over("symbol")
                .clip(lower_bound=0.0)
                .sqrt()
                * 2.0  # sqrt(E[x]) = s/2 = half-spread... in log terms s is
                # the FULL spread; half-spread = s/2 = sqrt(E[x]); the *2/2
                # cancels — kept explicit: s = 2*sqrt(E[x]), half = s/2
                / 2.0
                * 1e4
            )
            .clip(lower_bound=floor_bps, upper_bound=cap_bps)
            .alias("half_spread_bps")
        )
        .drop_nulls("half_spread_bps")
        .with_columns(pl.col("open_time").dt.date().alias("date"))
        .group_by("symbol", "date", maintain_order=True)
        .agg(pl.col("half_spread_bps").last())
        .select("symbol", "date", "half_spread_bps")
    )
