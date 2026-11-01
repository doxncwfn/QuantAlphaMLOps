# CLAUDE.md

Context for AI coding agents working in this repository. Read this first, then the two linked documents:

- @context/README.md: thesis context (problem, filter, objective, model, features, data, open questions)
- @docs/PLATFORM.md: how the code works (layout, experiment spec, data tables, component contracts, commands)
- docs/ONBOARDING.md: step-by-step guide for new team members (setup, data, first run, making a change)

## Project

**QuantAlphaMLOps**: HCMUT thesis by Hào, Cường and Hiếu. Cross-sectional ranking of US stocks for a long–short strategy. At each decision date t, a Filter picks k stocks from a survivorship-free Russell 1000 master list (2000–2024, CRSP via WRDS). An LSTM per series plus a Transformer across stocks then scores them, and the scores are judged by per-date Spearman IC.

The code is a config-driven platform (package `qam`, `src/qam/`):
- **Framework**, rarely edited: `cli.py`, `registry.py`, `config.py`, `data.py`, `pipeline.py`, `train.py`, `tracking.py`.
- **Components**, where research changes go: `src/qam/components/*.py`. Each is registered with `@register(kind, name)` and selected by name in YAML.
- Configs in `configs/`, one spec per experiment in `specs/`, previous code kept unchanged in `legacy/` (used by parity tests).
- Reports (LaTeX) are in `reports/`.

## Invariants (do not change silently)

- **Point-in-time:** at decision date t, only data strictly before t. Filter, features and `load_window` are tested for this.
- **Cross-sectional:** losses and metrics are computed within one date, then averaged. Never pool stocks across dates.
- **Order-free stock slots:** no positional or identity encoding for stocks. The stock-ID embedding in `legacy/src/models/hybrid_embedded.py` was dropped for memorization.
- **Masking:** padding slots (fewer than k eligible) and missing labels (e.g. delisted before t + h) are masked out of the loss and the metrics.
- **Walk-forward only:** purged and embargoed, with gaps of at least the label horizon. No shuffling across time.
- **Use `security_id`** (= CRSP PERMNO), never ticker, as the key.
- **Legacy parity:** `tests/test_legacy_parity.py` must keep passing. An intentional behaviour change updates that test and docs/PLATFORM.md §8.

## Running

```bash
# Python ≥ 3.11. During development the sibling venv is used: ../.venv
pip install -e '.[dev]'          # add [tracking] for W&B
pytest                           # synthetic data only
qam validate configs/experiments/baseline_hybrid.yaml
qam run configs/experiments/smoke.yaml        # small slice; parameters only in YAML, CLI takes only --start/--end
```

- Local paths are in `configs/env/local.yaml`; the warehouse is `data/warehouse/russell1000.duckdb` (git-ignored; downloaded from MotherDuck, see `data/README.md`).
- Modal: `modal run modal_app.py --config ...` with `configs/env/modal.yaml`.

## Rules for this repo

- **Never commit data.** CRSP/WRDS data is licensed: no warehouse files, dataset DuckDB files, parquet or run outputs in git (`data/`, `runs/` are ignored). Tests use synthetic data only.
- **Secrets:** API keys (W&B, Alpaca) come from the environment or Modal secrets. Never read `.env*` files and never hardcode keys.
- **Remote is read-only from this machine.** The repo was cloned with a read-only deploy key (SSH alias `github-thesis`). Claude does not run `git pull`/`push`; the user does. Local commits are fine.
- **Ask before installing packages.**
- **Parameters come only from the spec YAML.** Do not add generic command-line overrides. The command line takes only time parameters (`CLI_TIME_KEYS` in `config.py`: today `--start`/`--end`). They win over the spec and log a warning when they differ. `configs/env/*.yaml` may set `paths` only.
- Keep `context/` in sync when a decision changes. Record open decisions in `context/08_open_questions.md`.
- New components need a test and, for a new experiment, a spec in `specs/`.
