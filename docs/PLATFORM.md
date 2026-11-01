# Platform guide

How the code is organised, what each pluggable part must provide, and how to change the pipeline. New to the repo? Start with [ONBOARDING.md](ONBOARDING.md). Thesis decisions (what to build and why) live in [../context/](../context/README.md); this file covers how the code works.

## 1. Idea

The run flow is fixed and lives in a few framework files. Everything that is a research choice is a **component**: a function or class registered under a name and selected in YAML.

```
YAML config ──► qam validate ──► build_dataset ──► train ──► evaluate ──► backtest
                 (schema +        (4 tables,        (folds,    (metrics     (placeholder)
                  names)           cached)           model,     from YAML)
                                                     loss)
```

Changing the pipeline means one of:
- **Change a value:** edit the experiment spec YAML. Parameters are not passed on the command line, except the period (`--start`, `--end`).
- **Change a part:** write a component, register it under a name, and set `func: <name>` in the spec.

The framework files should not need edits for either.

## 2. Layout

```
configs/
  experiments/*.yaml            one self-contained spec per experiment (all stages, see §2.1)
  env/local.yaml, env/modal.yaml  paths only (warehouse, data_root, runs_root)
specs/                          one short design note per experiment (why, what, how to verify)
data/                           local data, git-ignored (warehouse download: data/README.md)
src/qam/                        FRAMEWORK
  cli.py                        `qam` command
  registry.py                   name -> component lookup
  config.py                     pydantic schema + YAML loader (env overlay, CLI time parameters)
  data.py                       dataset store (DuckDB), load_window, torch Dataset
  pipeline.py                   build_dataset / evaluate / backtest stages, run()
  train.py                      walk-forward folds + generic Lightning module
  tracking.py                   run directory, meta, metrics, optional W&B
src/qam/components/             COMPONENTS (edit here)
  sources.py features.py filters.py indicators.py labels.py
  transforms.py models.py losses.py splitters.py metrics.py backtest.py
tests/                          pytest on synthetic data, including legacy parity tests
legacy/                         the previous code, unchanged, used by the parity tests
context/                        thesis context (decisions, open questions)
reports/                        LaTeX reports (proposal, phase 1, thesis) and final figures
```

When a component file grows past roughly 300 lines or holds many variants, turn it into a package (`components/models/__init__.py`, `components/models/tft.py`, …). Every module under `components/` is imported automatically, so nothing else changes.

### 2.1 Experiment spec

One experiment is one YAML file that holds the whole pipeline, one section per stage, in run order. There is no inheritance: reading the file shows everything the run does. The environment file (`--env`) adds only `paths`.

| Section | Holds | Read by |
|---|---|---|
| `name`, `seed` | run name, random seed | run directory, training |
| `data_spec` | `dataset_name`, `source`, `period`, `warmup_days` | build |
| `feature_spec` | list of features | build → `features` table |
| `filter_spec` | the Filter | build → `universe` table |
| `indicator_spec` | list of economic indicators | build → `market` table |
| `label_spec` | the label | build → `labels` table |
| `window_spec` | `lookback`, `max_missing`, `fill`, `transforms` | `load_window` |
| `model_spec`, `loss_spec`, `split_spec` | model, loss, walk-forward splitter | train |
| `train_spec` | epochs, batch size, optimiser, checkpoint monitor, device | train |
| `evaluation_spec` | `metrics` list | evaluate |
| `backtest_spec` | backtest engine | backtest |
| `tracking_spec` | W&B settings | tracking |

A pluggable part is always written the same way, in block style:

```yaml
filter_spec:
  func: topk_dollar_volume    # registered name
  kwargs:                     # passed to it as keyword arguments
    k: 500
    ttl: 21
```

Lists (`feature_spec`, `indicator_spec`, `window_spec.transforms`, `evaluation_spec.metrics`) hold one such entry per item. Unknown keys and unknown `func` names fail at `qam validate`.

**Where parameters come from.** Every parameter comes from the spec file, so a run is described by one file (saved as `config.resolved.yaml`). There are two exceptions:
- The environment file (`--env local|modal`) sets `paths`, and nothing else. Any other key in it is an error.
- **Time parameters** may be given on the command line. Today these are `--start` and `--end`, which set `data_spec.period`; a future `qam predict` will add a run date. The command line wins over the spec, and when the two differ a warning is logged: `--start 2018-01-01 overrides data_spec.period.start = 2000-07-01 from the spec`. Allowed flags are listed in `CLI_TIME_KEYS` in `config.py`.

To try another value of anything else, write a new spec (copy and edit).

## 3. Data: one file, one read function

Each dataset is one DuckDB file, `<data_root>/datasets/<data_spec.dataset_name>.duckdb`, with four tables. Each table depends only on its own part of the config. That part is hashed, and every hash is stored as its own table version `<table>__<hash>`. A version is never overwritten, so a new hash adds a version and an existing one is reused.

| Table | Rows | Built from | Depends on |
|---|---|---|---|
| `features` | (date, security_id) for every stock in the master list | features | `feature_spec` |
| `universe` | k rows per refresh date: slot, security_id, is_pad, valid_until | filter | `filter_spec` |
| `market` | one row per trading day; `ts__*` and `static__*` columns. Also the trading calendar | indicators | `indicator_spec` |
| `labels` | (decision date, security_id) → label; NaN when it cannot be computed | label | `label_spec` |
| `_manifest` | one row per table version: config hash, rows, dates, build time, config | | |

All four tables also depend on `data_spec` (source, period, warmup) and the warehouse file (size and modification time). Changing any of these builds new versions of all four.

The hash also covers the **code** of the components that build the table: the component's own definition plus the shared code of its module (helpers, constants, imports). Comments, docstrings and formatting are ignored, and so are the other registered components in the module. So fixing a bug in `vol()` builds a new `features` version, while adding a new feature to `features.py` does not. Framework code (`data.py`, `pipeline.py`) is not hashed: bump `BUILD_VERSION` in `data.py` when a change there alters table contents.

Versions make runs reproducible and keep experiments apart:
- Experiments that share `dataset_name` but differ in features, filter or label each keep their own versions.
- `qam run` / `qam train` record the four versions in `meta.json` (`dataset_tables`). `qam evaluate` / `qam backtest` on that run directory read exactly those versions, whatever was built since.
- Without the warehouse (Modal, an inference machine) the warehouse fingerprint cannot be computed. The newest stored version built from the same config and component code is used, and a change that needs a rebuild fails with a clear error.
- Writing a version and its manifest row is one transaction. A DuckDB file has one writer or many readers at a time, so a process waits (up to `LOCK_TIMEOUT_S`) while another holds the lock.
- `qam gc CONFIG` lists versions that no run under `runs_root` and not the current config use; `--yes` drops them. Runs kept under another `runs_root` (Modal vs local) are not seen, so run gc where all runs are. DuckDB reuses the freed space but does not shrink the file.

`DatasetView.load_window(t)` is the only read path, used by both training and inference:
1. Find the universe period that covers t → k slots (ids, padding).
2. Take the slots' features on the **L trading days before t** → `[k, L, F]`.
3. A slot is masked when it is padding or has more than `window_spec.max_missing` incomplete days. With `fill: ffill` gaps are forward-filled; other missing cells stay NaN.
4. Apply `window_spec.transforms` per day **across the k real slots only**: z-score, clip, rank. Missing cells are left out of the statistics and set to 0 afterwards, which after a z-score is the cross-sectional mean (not an extreme raw value).
5. Add market indicators for the same L days (time series) and the value at t − 1 (static).
6. For training, add the label at t for every slot and a `label_mask`.

Features are stored raw, so the filter can change without rebuilding them. Cross-sectional normalisation happens at load time, within the stocks the model actually sees.

## 4. Component contracts

Register with `@register(kind, name)` from `qam.registry`. The spec selects it with `func: <name>`, and `kwargs` are passed as keyword arguments (`**kw` below).

| Kind | File | Signature | Notes |
|---|---|---|---|
| source | sources.py | `fn(warehouse, start, end, **kw) -> pl.DataFrame` | Columns: security_id, date, open, high, low, close, volume, adj_close, ret_total |
| feature | features.py | `fn(**kw) -> pl.Expr` (aliased) | Run per security on date-sorted rows; past and current rows only |
| filter | filters.py | `fn(panel, calendar, start, end, **kw) -> pl.DataFrame[refresh_date, valid_until, slot, security_id, is_pad]` | Exactly k rows per refresh; use only data before refresh_date |
| indicator | indicators.py | `fn(panel, calendar, **kw) -> pl.DataFrame[date, value]`; set `fn.kind = "ts"` or `"static"` | Optional `alias` kwarg names the column |
| label | labels.py | `fn(panel, calendar, start, end, **kw) -> pl.DataFrame[date, security_id, label]` | NaN = left out of the objective |
| transform | transforms.py | `fn(x[k,L,F], mask[k], **kw) -> x` | Statistics over real slots; missing cells are NaN: ignore them and keep them NaN. Optional `features: [...]` kwarg restricts columns |
| model | models.py | `nn.Module`, built with `n_features, n_mkt_ts, n_mkt_static, lookback, k` + kwargs; `forward(batch) -> [B,k]` | No positional or identity encoding on stock slots |
| loss | losses.py | class; `__call__(scores[B,k], labels[B,k], mask[B,k]) -> scalar` | Per date, then averaged; never pool across dates |
| splitter | splitters.py | class; `split(points) -> list[Fold(name, train, val, test)]`, points = decision dates in time order | Ordered, disjoint, gaps ≥ label horizon |
| metric | metrics.py | `fn(df[date, security_id, score, label], **kw) -> MetricResult(value, per_date)` | Optional `alias` kwarg |
| backtest | backtest.py | `BacktestEngine.run(scores, load_returns) -> BacktestResult` | `load_returns()` reads daily returns through the source component; call it only when needed. Placeholder for now |

Invariants every component must respect (see [../context/02_formulation.md](../context/02_formulation.md)):
- **Point-in-time:** at decision date t, use data strictly before t. Tests check this for the filter and `load_window`.
- **Cross-sectional:** compare stocks only within the same date.
- **Order-free:** the order of stock slots carries no meaning.
- **Synthetic test data only:** never put CRSP data in tests or in git.

## 5. Onboarding a change

1. **Write the component** in the matching file under `src/qam/components/`, following the contract above. Add a test in `tests/` using the synthetic panel in `tests/conftest.py`.
2. **Write the spec:** copy [../specs/_TEMPLATE.md](../specs/_TEMPLATE.md) to `specs/<experiment>.md`.
3. **Write the experiment spec:** copy `configs/experiments/baseline_hybrid.yaml` to `configs/experiments/<experiment>.yaml`, change `name` and the sections that differ, and say in the header comment what differs from the baseline.
4. **Check and run:**
   ```bash
   qam components                                   # is the new name registered?
   qam validate configs/experiments/<experiment>.yaml
   pytest
   qam run configs/experiments/<experiment>.yaml
   ```

To change only a value: copy the spec, change the value and `name`, and run the copy. Only the period can be changed on the command line: `qam run configs/experiments/baseline_hybrid.yaml --start 2010-01-01 --end 2020-12-31`.

## 6. Commands

| Command | Does |
|---|---|
| `qam validate CONFIG` | Apply env and CLI time parameters; validate the schema and component names; print the result |
| `qam build-dataset CONFIG [--force]` | Build the table versions this config needs that are not stored yet |
| `qam train CONFIG` | New run: walk-forward training and out-of-sample predictions |
| `qam evaluate CONFIG --run-dir runs/<id>` | Metrics on an existing run (use the same config) |
| `qam backtest CONFIG --run-dir runs/<id>` | Backtest on an existing run (placeholder: skipped) |
| `qam run CONFIG [--force-build]` | build → train → evaluate → backtest in one run directory |
| `qam gc CONFIG [--yes]` | List (or drop) table versions no run uses |
| `qam components` | List registered components |

Common options: `--env local|modal` (default `local`), `--start YYYY-MM-DD` and `--end YYYY-MM-DD` (win over `data_spec.period`, with a warning when they differ).

On Modal: `modal run modal_app.py --config configs/experiments/baseline_hybrid.yaml [--start ...] [--end ...]`. Paths come from `configs/env/modal.yaml`. Uploading CRSP-derived files to the Modal volume puts licensed data on a third-party cloud, so check the WRDS terms first.

## 7. Runs and logging

Every run writes `runs/<UTC time>_<name>_<config hash>/`:

| File | Content |
|---|---|
| `config.resolved.yaml` | The exact config used |
| `meta.json` | run id, git sha, dirty flag, config hash, dataset table versions (`dataset_tables`), package versions, start and end time, status, backtest status |
| `metrics.json` | Scalars: `test/<metric>`, `fold_i/test/<metric>`, `fold_i/best_val_ic` |
| `metrics_per_date.parquet` | Per-date series (IC, spread) |
| `predictions.parquet` | date, security_id, slot, score, fold for every real slot of every test date |
| `checkpoints/`, `lightning/` | Best checkpoint and Lightning CSV logs per fold |
| `run.log` | Log lines, tagged with the run id |

W&B is optional: `tracking_spec.wandb.enabled: true` (install with `pip install -e '.[tracking]'`; the key comes from the environment or the Modal secret). Only params and scalar metrics are sent, never data or tables. Paths are removed from the params.

## 8. Legacy parity

`tests/test_legacy_parity.py` runs the old code in `legacy/src` and the new code on the same synthetic inputs.

| Piece | Result |
|---|---|
| Features: log return (clipped), 60-day vol, distance to SMA 20 and 60 | Identical |
| Per-date z-score + clip ±3 | Identical (ddof = 1, as in polars) |
| Model `HybridAlphaRanker` vs `hybrid_lstm_tfm` | Same parameters; the legacy weights load as-is and give the same scores |
| Loss `HybridAlphaLoss` vs `hybrid_hinge` | Identical (`target: raw` on z-scored labels, or `target: zscore` on raw labels) |
| Mean IC, ICIR, decile spread vs `evaluate_alpha` | Identical when the same return is passed |
| Validation IC vs `scipy.stats.spearmanr` | Identical |
| Walk-forward test blocks | Identical; the new train span = old train span minus a purge gap and the validation block |

Intentional differences, each covered by a test:
- **Checkpoint selection.** Checkpoints are chosen on a validation block carved from the training history. The legacy code chose them on the test fold, which leaked the test set.
- **Decile spread.** It uses the forward label; the legacy version used the same-day return.
- **Burn-in.** No burn-in rows are needed: `load_window` reads history from the features table.
- **Universe.** A top-k filter on the survivorship-free warehouse replaces the strict-intersection yfinance universe.
- **Target.** The forward total return (z-scored inside the loss) replaces the legacy Fama-French residual, which is not ported. Open question 4 in [../context/08_open_questions.md](../context/08_open_questions.md).
- **Inputs.** `log_ret` is computed by the legacy code but was not a model input there. `configs/experiments/legacy_parity.yaml` keeps the legacy inputs: raw OHLCV, vol 60 and dist SMA 20/60.

## 9. Not done yet

- Backtest engine (interface only).
- `on_delist: refill` in the filter (open question 3).
- Residualised target as a label component.
- XAI (Integrated Gradients) and hyperparameter search: the legacy code is in `legacy/src/evaluation/xai.py` and `legacy/src/train.py`.
- A `qam predict --date` command for live inference. `load_window` alone is not enough yet:
  - It only accepts dates in the calendar, and the calendar holds only days that already have data. The next trading day after the data is not one of them.
  - Features, calendar and universe are cut at `period.end`, so predicting later needs a longer period, which builds new versions of every table.
- The 2025–2026 data (not yet in the warehouse silver layer).
