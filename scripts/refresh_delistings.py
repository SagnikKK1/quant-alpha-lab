"""Refresh the delisting registry from the curated panel.

Workflow (run after any broad ingest):
1. infer settlement candidates from the curated klines (symbols that went
   quiet while the panel kept trading);
2. merge into registry/delistings.csv — existing entries WIN (a confirmed
   announcement time beats an inferred one; the registry is append-mostly and
   reviewed in PRs);
3. print what changed, so the re-ingest that applies it is a deliberate step.

Usage:
    python scripts/refresh_delistings.py [--stale-days 30]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from alphalab.data import delistings, store

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "registry" / "delistings.csv"
EXCLUSIONS_PATH = Path(__file__).resolve().parents[1] / "registry" / "exclusions.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stale-days", type=int, default=30)
    args = parser.parse_args()

    klines = store.scan("klines").collect()
    changed = False

    # Trailing placeholder era -> settlement candidates.
    inferred = delistings.infer_settlements(klines, stale_days=args.stale_days)
    existing = delistings.load_registry(REGISTRY_PATH)
    new = inferred.filter(~pl.col("symbol").is_in(existing["symbol"].implode()))
    if not new.is_empty():
        merged = pl.concat([existing, new.select(existing.columns)]).sort("symbol")
        merged.with_columns(
            pl.col("settlement_time").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ).write_csv(REGISTRY_PATH)
        print(f"added {new.height} inferred settlement(s):")
        print(new)
        changed = True

    # Interior placeholder runs -> exclusion-window candidates (delist-relist).
    inferred_ex = delistings.infer_exclusions(klines)
    existing_ex = delistings.load_exclusions(EXCLUSIONS_PATH)
    new_ex = inferred_ex.join(
        existing_ex.select("symbol", "start_time"),
        on=["symbol", "start_time"],
        how="anti",
    )
    if not new_ex.is_empty():
        merged_ex = pl.concat([existing_ex, new_ex.select(existing_ex.columns)]).sort(
            "symbol", "start_time"
        )
        merged_ex.with_columns(
            pl.col("start_time").dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            pl.col("end_time").dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ).write_csv(EXCLUSIONS_PATH)
        print(f"added {new_ex.height} inferred exclusion window(s):")
        print(new_ex)
        changed = True

    if not changed:
        print(
            f"registries unchanged ({existing.height} settlements, "
            f"{existing_ex.height} exclusions)"
        )
        return 0
    print("review against delisting announcements, then re-run ingest to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
