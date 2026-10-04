# Interactive Notebooks

This directory contains Jupyter notebooks for exploratory analysis, performance evaluation, and visual auditing of the Russell 1000 dataset and alpha models.

---

## 1. Tracked Notebooks

- **[`audit.ipynb`](./audit.ipynb)**: Complete suite of **16 publication-quality analytical visualizations** developed for the Russell 1000 historical dataset (2000–2026).
  - Evaluates reconstitution cycle semantics ($t_{\text{June}} \rightarrow t_{\text{June}+1}$).
  - Plots annual index turnover, churn, and Jaccard similarity.
  - Visualizes 40-day model lookback eligibility and continuous cross-boundary stitching.
  - Powered by the reproducible audit summary tables in `report/quality/tables/`.

---

## 2. Scratch Notebook Policy

To prevent repository bloat from large embedded outputs and base64 plots:
- Experimental and scratch notebooks (`metrics.ipynb`, `metrics_2.ipynb`, etc.) and Jupyter checkpoint folders (`.ipynb_checkpoints/`) are ignored by default via `.gitignore`.
- Only curated, version-controlled notebooks (such as `audit.ipynb`) are tracked in the repository.

---

## 3. Environment & Execution

To launch Jupyter with project dependencies:

```bash
uv sync --extra dev
jupyter lab
```

Or using standard pip:
```bash
pip install -r requirements.txt
pip install jupyterlab
jupyter lab
```
