# Thesis context

Working context for the thesis: cross-sectional stock ranking for a long–short strategy. Each file covers one part of the system. Status markers: **decided**, **open** (needs a decision), **in progress**, **planned**.

| File | Topic | Status | Owner |
|---|---|---|---|
| [01_problem.md](01_problem.md) | Problem statement | decided | team |
| [02_formulation.md](02_formulation.md) | Notation and inference flow at time t | decided | team |
| [03_filter.md](03_filter.md) | Filter: top k by dollar volume with a retention buffer, cached for a TTL; fully parameterized | decided | Hào |
| [04_objective.md](04_objective.md) | Ranking criterion / loss: maximize Spearman rank IC | partly open | Hiếu |
| [05_model.md](05_model.md) | LSTM + Transformer architecture | decided at design level | Cường |
| [06_features.md](06_features.md) | Feature selection and engineering | in progress | Cường (domain), Hào (implementation) |
| [07_data.md](07_data.md) | Universe construction and data collection | done up to 2024-12-31 | Hào |
| [08_open_questions.md](08_open_questions.md) | All open decisions in one place | — | — |

Related material outside this folder:
- Code: this repository (see `../CLAUDE.md` and `../docs/PLATFORM.md`)
- Data warehouse: `../../warehouse/` (`README.md`, `SPEC.md`)
- Data pipeline scripts: `../../pipeline/`

## Notation (used in every file)

| Symbol | Meaning |
|---|---|
| t | Decision date. Only information available before t may be used |
| U_t | Universe at date t |
| k | Number of stocks selected by the Filter; fixed |
| L | Lookback length; the model sees the window [t − L, t) |
| m | Number of economic (market-level) indicators; fixed, in a fixed order |
| h | Prediction horizon; the target is the return from t to t + h (example: h = 5) |
| offset | Trading days between the last input day (t − 1) and the start of the target. **Default 0, as in the legacy code:** the position is entered at the close of t − 1, the same close the features end on. offset = 1 enters at the close of t |
| s_i | Model score for stock i at date t, i = 0 … k − 1 |
| r_i | Realized return of stock i from t to t + h |
| TTL | How long the Filter's selected list stays valid before it is recomputed |

The original notes used *n* both for the lookback length and for the number of economic indicators. These files use **L** and **m** to keep them apart.
