"""Paths and constants for the data layer."""

import os
from pathlib import Path

# Data root: gitignored ./data by default, overridable for CI / external disks.
DATA_DIR = Path(os.environ.get("ALPHALAB_DATA_DIR", "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"

BINANCE_VISION_BASE = "https://data.binance.vision"
UM_FUTURES_PREFIX = "data/futures/um"

# Interval string -> milliseconds. Only the intervals the lab uses.
INTERVAL_MS = {
    "1h": 3_600_000,
    "4h": 14_400_000,
    "8h": 28_800_000,
    "1d": 86_400_000,
}

# Verified facts about the source (HANDOFF §12.1, checked 2026-07-10):
# - monthly klines/fundingRate exist from 2020-01; metrics daily from 2020-09
# - um futures CSVs have a header row only from 2022-01 onward
# - um futures timestamps are milliseconds (spot moved to µs in 2025; um did not)
EARLIEST_MONTH = "2020-01"
