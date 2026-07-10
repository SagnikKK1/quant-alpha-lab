"""Static tear sheet (matplotlib PNG) for a backtest result.

Design notes (dataviz method): one axis per panel — never dual-axis; thin
marks with a recessive grid; categorical hues in fixed order from the
validated palette (blue #2a78d6, aqua #1baf7a; red #e34948 reserved for
loss/drawdown semantics); series identity carried by legend + direct end
labels, values in text ink, not series color.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

BLUE = "#2a78d6"
AQUA = "#1baf7a"
RED = "#e34948"
INK = "#333330"
MUTED = "#6b6a63"
GRID = dict(color="#c9c8c0", linewidth=0.6, alpha=0.5)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, axis="y", **GRID)
    ax.set_axisbelow(True)


def render_tearsheet(
    pnl: pl.DataFrame,
    title: str,
    stats: dict,
    out_path: Path,
) -> Path:
    """pnl: (date, gross_ret, cost, funding_pnl, net_ret, turnover, ...)."""
    dates = pnl["date"].to_list()
    net = pnl["net_ret"].to_numpy()
    gross = pnl["gross_ret"].to_numpy()
    eq_net = (1 + pnl["net_ret"]).cum_prod().to_numpy()
    eq_gross = (1 + pnl["gross_ret"]).cum_prod().to_numpy()
    peak = pl.Series(eq_net).cum_max().to_numpy()
    dd = eq_net / peak - 1.0
    turnover = pnl["turnover"].to_numpy()

    fig = plt.figure(figsize=(11, 12), facecolor="white")
    gs = fig.add_gridspec(5, 1, height_ratios=[0.9, 2.2, 1.1, 1.1, 1.1], hspace=0.45)

    # -- headline stats: text, not a chart ---------------------------------
    ax0 = fig.add_subplot(gs[0])
    ax0.axis("off")
    ax0.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold", pad=18)
    cells = [  # (x, row, label, value) — CI gets a double-width slot
        (0.00, 0, "Net Sharpe (ann)", f"{stats['sharpe']:+.2f}"),
        (0.14, 0, "95% CI (block bootstrap)",
         f"[{stats['ci_lo']:+.2f}, {stats['ci_hi']:+.2f}]"),
        (0.44, 0, "Ann return", f"{stats['ann_return']:+.1%}"),
        (0.63, 0, "Max DD", f"{stats['max_drawdown']:.1%}"),
        (0.82, 0, "Hit rate", f"{stats['hit_rate']:.1%}"),
        (0.00, 1, "Block len", f"{stats['block_length']:.1f}d"),
        (0.14, 1, "Turnover/d", f"{stats['avg_daily_turnover']:.2f}"),
        (0.44, 1, "Cost drag (ann)", f"{stats['total_cost_drag_ann']:.1%}"),
        (0.63, 1, "Funding (ann)", f"{stats['total_funding_ann']:+.1%}"),
    ]
    for x, row, label, value in cells:
        y = 0.55 - row * 0.55
        ax0.text(x, y, value, fontsize=14, color=INK, fontweight="bold",
                 transform=ax0.transAxes)
        ax0.text(x, y - 0.28, label, fontsize=8, color=MUTED, transform=ax0.transAxes)

    # -- equity -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[1])
    _style(ax1)
    ax1.plot(dates, eq_gross, color=AQUA, linewidth=1.6, label="Gross (pre-cost)")
    ax1.plot(dates, eq_net, color=BLUE, linewidth=2.0, label="Net (costs + funding)")
    ax1.text(dates[-1], eq_net[-1], f"  {eq_net[-1]:.2f}x", color=INK, fontsize=9,
             va="center", fontweight="bold")
    ax1.text(dates[-1], eq_gross[-1], f"  {eq_gross[-1]:.2f}x", color=MUTED,
             fontsize=8, va="center")
    ax1.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    ax1.set_title("Equity (1 = start)", loc="left", fontsize=10, color=INK)

    # -- drawdown ------------------------------------------------------------
    ax2 = fig.add_subplot(gs[2])
    _style(ax2)
    ax2.fill_between(dates, dd, 0, color=RED, alpha=0.35, linewidth=0)
    ax2.plot(dates, dd, color=RED, linewidth=1.2)
    ax2.set_title("Drawdown (net)", loc="left", fontsize=10, color=INK)

    # -- rolling 90d Sharpe ---------------------------------------------------
    ax3 = fig.add_subplot(gs[3])
    _style(ax3)
    s = pl.Series(net)
    roll_mu = s.rolling_mean(90)
    roll_sd = s.rolling_std(90)
    roll_sharpe = (roll_mu / roll_sd * (365**0.5)).to_numpy()
    ax3.plot(dates, roll_sharpe, color=BLUE, linewidth=1.6)
    ax3.axhline(0, color=MUTED, linewidth=0.8)
    ax3.set_title("Rolling 90d Sharpe (ann)", loc="left", fontsize=10, color=INK)

    # -- turnover -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[4])
    _style(ax4)
    t = pl.Series(turnover).rolling_mean(30).to_numpy()
    ax4.plot(dates, t, color=BLUE, linewidth=1.6)
    ax4.set_title("Turnover, 30d avg (fraction of AUM/day)", loc="left",
                  fontsize=10, color=INK)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
