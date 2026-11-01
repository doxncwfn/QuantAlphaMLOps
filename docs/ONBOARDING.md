# Onboarding

From a fresh clone to your own experiment. Read this first, then [PLATFORM.md](PLATFORM.md) for how the code works and [../context/](../context/README.md) for the thesis decisions.

## 1. Set up (once)

Needs Python 3.11 or newer.

```bash
git clone <repo URL> && cd QuantAlphaMLOps
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'          # add ,tracking for W&B: pip install -e '.[dev,tracking]'
pytest                           # synthetic data only; must pass before you change anything
```

`pytest` needs no real data. If it fails on a clean clone, fix that first.

## 2. Get the data (once)

The warehouse is not in git (CRSP data, licensed through WRDS). Download it from MotherDuck into `data/warehouse/` by following [../data/README.md](../data/README.md). You need access to the share first; ask Hào.

Check it:

```bash
qam validate configs/experiments/baseline_hybrid.yaml   # prints the resolved spec
```

## 3. First run: a small smoke test

A full run takes long. Start with [`smoke.yaml`](../configs/experiments/smoke.yaml): the baseline on two years, 50 stocks, one epoch and two folds.

```bash
qam run configs/experiments/smoke.yaml
```

The command prints the run directory, `runs/<time>_<name>_<hash>/`. Look at:

| File | What to check |
|---|---|
| `meta.json` | `status: finished`, the git sha, the dataset table versions used |
| `metrics.json` | `test/mean_ic`, `test/icir`, … and the same per fold (`fold_1/test/...`) |
| `predictions.parquet` | one score per stock and test date |
| `run.log` | what happened, step by step |

The first run builds the dataset tables into `data/datasets/`. Later runs reuse them. Only a table whose part of the spec changed is built again.

## 4. Read an experiment spec

One experiment is one YAML file in `configs/experiments/`. It holds the whole pipeline, one section per stage, in run order. Open [../configs/experiments/baseline_hybrid.yaml](../configs/experiments/baseline_hybrid.yaml) next to this table:

| Section | Stage |
|---|---|
| `data_spec` | where the prices come from and which period |
| `feature_spec` | stock time series fed to the model |
| `filter_spec` | which k stocks are selected at each date ([context/03](../context/03_filter.md)) |
| `indicator_spec` | economic indicators |
| `label_spec` | the training target |
| `window_spec` | lookback, missing-data handling, cross-sectional transforms |
| `model_spec`, `loss_spec` | the model and its loss |
| `split_spec`, `train_spec` | walk-forward folds and training settings |
| `evaluation_spec` | the metrics written to `metrics.json` |
| `backtest_spec`, `tracking_spec` | backtest (placeholder) and W&B |

A pluggable part always has the same shape: `func` is the registered name, `kwargs` are its arguments.

```yaml
label_spec:
  func: forward_return
  kwargs:
    h: 5
    offset: 0
```

Paths are not in the spec. They come from `configs/env/local.yaml`, or `configs/env/modal.yaml` with `--env modal`.

**Parameters live in the spec, not on the command line.** The only exception is time: `--start` and `--end` set the period. When they differ from the spec, the command line wins and a warning is logged, for example `--start 2018-01-01 overrides data_spec.period.start = 2000-07-01 from the spec`.

## 5. Make a change

Pick the smallest kind that does the job.

### a. Try a different value

- **Period only:** pass it on the command line, e.g. `qam run configs/experiments/baseline_hybrid.yaml --start 2010-01-01 --end 2020-12-31`.
- **Anything else:** make a new spec as in (b), even for a single value. This keeps every run described by a file. The run also saves the spec it used as `config.resolved.yaml`.

### b. A new experiment from existing parts

1. Copy the baseline: `cp configs/experiments/baseline_hybrid.yaml configs/experiments/<name>.yaml`.
2. Change `name`, and the sections you want different. In the header comment, write what differs from the baseline.
3. Write a short note: copy [../specs/_TEMPLATE.md](../specs/_TEMPLATE.md) to `specs/<name>.md` (why, what changes, how to verify).
4. Run `qam validate configs/experiments/<name>.yaml`, then `qam run ...`.

### c. A new component (a feature, filter, label, transform, model, loss, metric, …)

Example: a momentum feature.

1. **Write it** in the file for its kind under `src/qam/components/`. Follow the contract at the top of that file and in [PLATFORM.md §4](PLATFORM.md#4-component-contracts).
   ```python
   # src/qam/components/features.py
   @register("feature", "momentum")
   def momentum(n: int = 20) -> pl.Expr:
       """Log return of the adjusted close over the last n rows."""
       return pl.col("adj_close").log().diff(n).alias(f"momentum_{n}")
   ```
2. **Test it** in `tests/`, on the synthetic panel from `tests/conftest.py` (never real data). Check the values, and check that it uses no data from day t or later.
3. **Use it** in a spec:
   ```yaml
   feature_spec:
     - func: momentum
       kwargs:
         n: 20
   ```
4. **Check:**
   ```bash
   qam components                                   # is "momentum" listed?
   qam validate configs/experiments/<name>.yaml
   pytest
   ```

The framework (`src/qam/*.py`) does not need edits for any of this. If you think it does, talk to the team first.

Editing a component's code is detected automatically. The next build creates new versions of the tables it feeds, so a stale cache is never used.

## 6. Rules

- **Point-in-time:** at decision date t, use only data from before t. This is the most common way to get results that are too good to be true.
- **Cross-sectional:** compare stocks only within the same date. Never pool across dates.
- **Never commit data:** no warehouse, dataset, parquet or run files. `data/` and `runs/` are git-ignored. Tests use synthetic data only.
- **Secrets** (MotherDuck token, W&B key) live in environment variables, never in files.
- **A decision changed?** Update the matching file in `context/`. Record open decisions in [../context/08_open_questions.md](../context/08_open_questions.md).
- **`tests/test_legacy_parity.py` must keep passing.** An intentional change of behaviour updates that test and [PLATFORM.md §8](PLATFORM.md#8-legacy-parity).

## 7. Useful commands

| Command | Does |
|---|---|
| `qam components` | List every registered component |
| `qam validate SPEC` | Check a spec and print it resolved |
| `qam build-dataset SPEC` | Build only the dataset tables |
| `qam run SPEC` | Build, train, evaluate, backtest |
| `qam evaluate SPEC --run-dir runs/<id>` | Recompute the metrics of an existing run |
| `qam gc SPEC [--yes]` | List (or drop) dataset table versions no run uses |

All of them take `--env local|modal`, and `--start` / `--end` for the period.
