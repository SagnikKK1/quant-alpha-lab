"""Append-only trial registry — the multiplicity ledger.

Every strategy variant evaluated against the OOS path gets a row, INCLUDING
dead ends: the registry's row count is the N that deflates the final Sharpe
(DSR), and the stored per-trial OOS P&L series are the M×N matrix CSCV needs.
Deleting or editing a row is falsifying the experiment log — the API only
appends, and re-logging an identical config is a no-op while a SAME-ID,
DIFFERENT-PAYLOAD write raises.

Counting rule (HANDOFF §2): a trial = anything evaluated on the OOS path.
Nested inner-fold search counts once via its pipeline's single OOS entry.

Lives in the repo (not the gitignored data dir) — it's an honesty artifact,
reviewed in PRs like the delisting registry.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

TRIALS_DIR = Path(__file__).resolve().parents[3] / "trials"
LEDGER = "trials.jsonl"


def _canonical(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def trial_id(config: dict) -> str:
    return hashlib.sha256(_canonical(config).encode()).hexdigest()[:16]


def log_trial(
    config: dict,
    oos_pnl: pl.DataFrame,  # (date, net_ret) concatenated OOS series
    metrics: dict,
    trials_dir: Path | None = None,
) -> str:
    """Append a trial; returns its id. Idempotent on identical payloads."""
    tdir = trials_dir or TRIALS_DIR
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "pnl").mkdir(exist_ok=True)
    tid = trial_id(config)

    payload_hash = hashlib.sha256(
        (_canonical(config) + _canonical(metrics)).encode()
    ).hexdigest()[:16]
    ledger_path = tdir / LEDGER
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            row = json.loads(line)
            if row["trial_id"] == tid:
                if row["payload_hash"] != payload_hash:
                    raise ValueError(
                        f"trial {tid} already logged with different results — "
                        "the registry is append-only; register a new config "
                        "instead of overwriting an experiment"
                    )
                return tid  # identical re-run, no-op

    oos_pnl.select("date", "net_ret").write_parquet(tdir / "pnl" / f"{tid}.parquet")
    entry = {
        "trial_id": tid,
        "logged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config,
        "metrics": metrics,
        "payload_hash": payload_hash,
        "pnl_file": f"pnl/{tid}.parquet",
    }
    with ledger_path.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return tid


def load_ledger(trials_dir: Path | None = None) -> list[dict]:
    tdir = trials_dir or TRIALS_DIR
    path = tdir / LEDGER
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]
