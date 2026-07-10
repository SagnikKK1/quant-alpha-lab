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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stale-days", type=int, default=30)
    args = parser.parse_args()

    klines = store.scan("klines").collect()
    inferred = delistings.infer_settlements(klines, stale_days=args.stale_days)
    existing = delistings.load_registry(REGISTRY_PATH)

    new = inferred.filter(~pl.col("symbol").is_in(existing["symbol"].implode()))
    if new.is_empty():
        print(f"registry unchanged ({existing.height} entries)")
        return 0

    merged = pl.concat([existing, new.select(existing.columns)]).sort("symbol")
    out = merged.with_columns(
        pl.col("settlement_time").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    out.write_csv(REGISTRY_PATH)
    print(f"added {new.height} inferred settlement(s):")
    print(new)
    print("review against delisting announcements, then re-run ingest to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
