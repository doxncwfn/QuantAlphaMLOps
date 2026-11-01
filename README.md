# QuantAlphaMLOps

Cross-sectional stock ranking for a long–short strategy (HCMUT thesis). A Filter selects k liquid stocks from a survivorship-free Russell 1000 master list. A hybrid LSTM–Transformer then ranks them, and the ranking is evaluated with purged walk-forward validation.

The code is a config-driven platform. The run flow is fixed; features, filter, label, model, loss, metrics and backtest are pluggable components selected in YAML.

## Quick start

```bash
pip install -e '.[dev]'                 # Python >= 3.11; add [tracking] for W&B
# download the warehouse: data/README.md
pytest                                  # synthetic data only
qam validate configs/experiments/baseline_hybrid.yaml
qam run configs/experiments/baseline_hybrid.yaml
```

Each run writes `runs/<id>/` with the resolved config, metadata, metrics, predictions and logs.

## Layout

| Path | Content |
|---|---|
| `src/qam/` | Framework: CLI, registry, config, data, pipeline, training, tracking |
| `src/qam/components/` | Pluggable components: features, filters, labels, models, losses, metrics, … |
| `configs/` | Base config, environments, datasets, experiments |
| `specs/` | One spec per experiment |
| `tests/` | Unit, end-to-end and legacy parity tests |
| `docs/ONBOARDING.md` | Start here: setup, data download, first run, adding an experiment or component |
| `docs/PLATFORM.md` | Architecture, component contracts, data tables |
| `context/` | Thesis context: decisions and open questions |
| `reports/` | Proposal, Phase 1 report, thesis (LaTeX) |
| `legacy/` | Previous implementation, kept for comparison |

## Data

Prices come from the project warehouse `data/warehouse/russell1000.duckdb` (CRSP via WRDS). It is not in git: download it from MotherDuck as described in [data/README.md](data/README.md). CRSP data is licensed: never commit data or outputs.
