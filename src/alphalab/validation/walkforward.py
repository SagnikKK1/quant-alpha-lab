"""Purged walk-forward splits.

Scoping, stated precisely (HANDOFF §2 — interviewers use this as the depth
probe): in FORWARD-CHAINING walk-forward, training always precedes testing,
so the only leakage channel is training labels whose forward window overlaps
the test period. PURGING removes them: a decision date t with label horizon H
is dropped from training when t + H reaches past the test start. EMBARGO
proper is a CSCV/k-fold concept (training data can FOLLOW test data there);
it does not apply here and is deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Fold:
    fold: int
    train: tuple[date, date]  # inclusive decision-date range, post-purge
    test: tuple[date, date]  # inclusive


def purged_walk_forward(
    dates: list[date],
    n_folds: int,
    label_horizon_days: int,
    min_train_days: int = 180,
) -> list[Fold]:
    """Split sorted decision dates into sequential test blocks with expanding
    purged training windows.

    Training for fold k = all dates from the start up to
    (test_start - label_horizon_days - 1): a training decision made at t owns
    the return window [t+1 open, t+1+H open); it is purged iff that window
    touches the test block.
    """
    if not dates:
        raise ValueError("no dates")
    dates = sorted(dates)
    first_test_idx = next(
        (i for i, d in enumerate(dates) if d >= dates[0] + timedelta(days=min_train_days)),
        None,
    )
    if first_test_idx is None or first_test_idx >= len(dates):
        raise ValueError(f"not enough history for min_train_days={min_train_days}")
    test_dates = dates[first_test_idx:]
    block = len(test_dates) // n_folds
    if block == 0:
        raise ValueError("more folds than test dates")

    folds = []
    for k in range(n_folds):
        lo = k * block
        hi = (k + 1) * block if k < n_folds - 1 else len(test_dates)
        test_start, test_end = test_dates[lo], test_dates[hi - 1]
        purge_cutoff = test_start - timedelta(days=label_horizon_days + 1)
        train_dates = [d for d in dates if d <= purge_cutoff]
        if not train_dates:
            raise ValueError(f"fold {k}: purge removed the entire training set")
        folds.append(
            Fold(fold=k, train=(train_dates[0], train_dates[-1]), test=(test_start, test_end))
        )
    return folds
