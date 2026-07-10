"""Purged walk-forward scoping + block bootstrap behavior."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from alphalab.registry.trials import load_ledger, log_trial
from alphalab.validation.bootstrap import politis_white_block_length, sharpe_ci
from alphalab.validation.walkforward import purged_walk_forward


def _dates(n, start=date(2022, 1, 1)):
    return [start + timedelta(days=i) for i in range(n)]


class TestPurgedWalkForward:
    def test_purge_gap_matches_label_horizon(self):
        folds = purged_walk_forward(_dates(400), n_folds=4, label_horizon_days=5,
                                    min_train_days=100)
        for f in folds:
            train_end, test_start = f.train[1], f.test[0]
            # a train decision at t owns returns through t + horizon; the
            # last allowed t satisfies t + 5 < test_start
            assert (test_start - train_end).days > 5

    def test_test_blocks_are_sequential_and_disjoint(self):
        folds = purged_walk_forward(_dates(400), n_folds=4, label_horizon_days=5,
                                    min_train_days=100)
        for a, b in zip(folds, folds[1:], strict=False):
            assert a.test[1] < b.test[0]
        # expanding training window
        assert all(f.train[0] == folds[0].train[0] for f in folds)

    def test_training_always_precedes_testing(self):
        folds = purged_walk_forward(_dates(300), n_folds=3, label_horizon_days=1,
                                    min_train_days=90)
        assert all(f.train[1] < f.test[0] for f in folds)

    def test_insufficient_history_raises(self):
        with pytest.raises(ValueError):
            purged_walk_forward(_dates(50), n_folds=3, label_horizon_days=1,
                                min_train_days=100)


class TestBlockBootstrap:
    def test_iid_series_gets_short_blocks(self):
        rng = np.random.default_rng(0)
        b = politis_white_block_length(rng.normal(size=2000))
        assert b < 5

    def test_autocorrelated_series_gets_long_blocks(self):
        rng = np.random.default_rng(0)
        x = np.zeros(2000)
        eps = rng.normal(size=2000)
        for t in range(1, 2000):
            x[t] = 0.9 * x[t - 1] + eps[t]
        b_ar = politis_white_block_length(x)
        b_iid = politis_white_block_length(rng.normal(size=2000))
        assert b_ar > 4 * b_iid  # strongly persistent -> much longer blocks

    def test_sharpe_ci_covers_truth_and_widens_with_dependence(self):
        rng = np.random.default_rng(1)
        n = 1500
        iid = rng.normal(0.001, 0.01, n)
        ar = np.zeros(n)
        eps = rng.normal(0.0, 0.01, n)
        for t in range(1, n):
            ar[t] = 0.8 * ar[t - 1] + eps[t]
        ar = ar + 0.001
        ci_iid = sharpe_ci(iid, ann_factor=365)
        ci_ar = sharpe_ci(ar, ann_factor=365)
        assert ci_iid["ci_lo"] < ci_iid["sharpe"] < ci_iid["ci_hi"]
        width_iid = ci_iid["ci_hi"] - ci_iid["ci_lo"]
        width_ar = ci_ar["ci_hi"] - ci_ar["ci_lo"]
        assert width_ar > width_iid  # ignoring dependence understates risk

    def test_bootstrap_is_seeded_and_reproducible(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0.0005, 0.01, 800)
        assert sharpe_ci(x) == sharpe_ci(x)


class TestTrialRegistry:
    def _pnl(self):
        return pl.DataFrame(
            {"date": _dates(10), "net_ret": [0.001] * 10}
        )

    def test_append_and_reload(self, tmp_path):
        tid = log_trial({"model": "baseline", "lb": 30}, self._pnl(),
                        {"sharpe": 1.0}, trials_dir=tmp_path)
        ledger = load_ledger(tmp_path)
        assert len(ledger) == 1
        assert ledger[0]["trial_id"] == tid
        assert (tmp_path / "pnl" / f"{tid}.parquet").exists()

    def test_identical_relog_is_noop(self, tmp_path):
        cfg = {"model": "baseline", "lb": 30}
        log_trial(cfg, self._pnl(), {"sharpe": 1.0}, trials_dir=tmp_path)
        log_trial(cfg, self._pnl(), {"sharpe": 1.0}, trials_dir=tmp_path)
        assert len(load_ledger(tmp_path)) == 1

    def test_overwriting_results_is_refused(self, tmp_path):
        cfg = {"model": "baseline", "lb": 30}
        log_trial(cfg, self._pnl(), {"sharpe": 1.0}, trials_dir=tmp_path)
        with pytest.raises(ValueError, match="append-only"):
            log_trial(cfg, self._pnl(), {"sharpe": 2.0}, trials_dir=tmp_path)

    def test_different_configs_are_different_trials(self, tmp_path):
        log_trial({"lb": 30}, self._pnl(), {"s": 1}, trials_dir=tmp_path)
        log_trial({"lb": 60}, self._pnl(), {"s": 1}, trials_dir=tmp_path)
        assert len(load_ledger(tmp_path)) == 2
