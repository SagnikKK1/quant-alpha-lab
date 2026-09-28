# Quant Alpha Lab

Cross-sectional ML alpha platform whose centerpiece is the **anti-overfitting
harness itself**: a point-in-time data layer, leakage tests in CI, purged
walk-forward validation, a full trial registry feeding Deflated Sharpe / PBO,
and live shadow-trading cost reconciliation.

> Status: **data layer, validation harness and cost-aware engine built**; a baseline
> cross-sectional momentum signal is tear-sheeted. Implemented: point-in-time store,
> future-truncation leakage detection with CI canaries, purged walk-forward validation,
> stationary block bootstrap, append-only trial registry, funding- and cost-aware
> execution engine. **Not yet implemented: Deflated Sharpe and PBO** (the registry
> stores what they need); no lockbox-holdout results are claimed.

## Data layer guarantees

1. **Point-in-time, survivorship-aware.** Universe membership is derived from
   rolling volume ranks using only information available strictly before each
   rebalance date. Delisted contracts are retained with history truncated at
   settlement.
2. **The placeholder-bar trap is handled.** Binance keeps generating fake
   klines for delisted perps (frozen price, zero volume) for years after
   settlement. Ingestion is faithful-to-source; the curation step truncates at
   settlement and CI asserts no zero-volume placeholder run survives.
3. **Checksum-verified ingestion.** Every bulk file's SHA256 `.CHECKSUM` is
   verified before parsing.
4. **Reproducibility by content, not bytes.** The store exposes a canonical
   content hash (sorted rows, ms-integer timestamps) that is stable across
   Parquet encodings and library versions; CI compares hashes, not files.
5. **No hardcoded funding cadence.** Funding rows carry
   `funding_interval_hours`; the grid audit validates against it (Binance is
   8h today; the harness doesn't assume it).

## Layout

```
src/alphalab/
  config.py            # paths + constants
  data/
    binance_bulk.py    # data.binance.vision download client (checksummed)
    schema.py          # CSV parsing: header seam (2022-01), ms timestamps
    store.py           # Parquet/DuckDB PIT store + canonical content hash
    audits.py          # gaps, duplicates, grid alignment, placeholder runs
    delistings.py      # settlement inference + truncation
    universe.py        # rolling volume-rank universe (no lookahead)
    ingest.py          # CLI: download → parse → curate → audit → manifest
tests/                 # synthetic-fixture unit tests + network-marked smoke
```

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"
pytest                      # unit tests (no network)
pytest -m network           # live smoke test against data.binance.vision
python -m alphalab.data.ingest --symbols BTCUSDT ETHUSDT \
    --start 2020-01 --end 2020-03 --interval 1h
```

Data is written under `./data/` (gitignored); override with
`ALPHALAB_DATA_DIR`.

## Related project

The execution-cost models this repo's backtests charge (fees, structural
spreads, calibrated impact law) live canonically in
[tca-lab](https://github.com/SagnikKK1/tca-lab), together with their market
studies (spread reconciliation vs real quotes, tape calibration of the
impact exponents, order-book walk validation, cost-model bake-off).
`src/alphalab/backtest/costs.py` and `spreads.py` are vendored copies kept
in sync until tca-lab is public/installable; the strategy-coupled execution
studies (delay-cost, execution frontier) remain here because they reprice
this repo's trades.

## Project plan

See `Quant_ML_Alpha_Lab_HANDOFF.md` (project root, outside this repo) for the
full v4 plan: leakage tests, trial registry, DSR/PBO, execution sim,
implementation-shortfall lab, and the shadow-trading protocol.
