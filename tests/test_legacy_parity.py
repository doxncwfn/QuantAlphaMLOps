"""Old code (legacy/src) and new code run on the same synthetic inputs and must agree.

Ported pieces must match numerically. Pieces changed on purpose are tested for the exact
difference, so a change in behaviour is always visible here.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch

from qam import registry

LEGACY = Path(__file__).resolve().parents[1] / "legacy" / "src"


def load_legacy(relative: str, stubs: tuple[str, ...] = ()):
    """Import a legacy file by path. Unused heavy imports (plots, downloaders) are stubbed."""
    for name in stubs:
        sys.modules.setdefault(name, types.ModuleType(name))
    path = LEGACY / relative
    spec = importlib.util.spec_from_file_location(f"legacy_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- features


def test_features_match_legacy(panel):
    residualizer = load_legacy("data/residualizer.py", stubs=("pandas_datareader", "pandas_datareader.data"))
    # Stocks with a full, gap-free history whose prices follow their returns exactly.
    clean = panel.filter(pl.col("security_id") < 10_027).sort(["security_id", "date"])

    old_in = clean.select(
        pl.col("date").cast(pl.Datetime).alias("Date"),
        pl.col("security_id").cast(pl.Utf8).alias("Ticker"),
        pl.col("close").alias("Close"),
    )
    old = residualizer.UniversalDataProcessor._engineer_features(None, old_in)

    exprs = [
        registry.build("feature", spec)
        for spec in (
            _spec("log_ret", clip=[np.log(0.8), np.log(1.2)]),
            _spec("vol", n=60),
            _spec("dist_sma", n=20),
            _spec("dist_sma", n=60),
        )
    ]
    new = clean.with_columns([e.over("security_id") for e in exprs])

    pairs = {"Log_Ret": "log_ret", "Vol_60": "vol_60", "Dist_SMA_20": "dist_sma_20", "Dist_SMA_60": "dist_sma_60"}
    for old_col, new_col in pairs.items():
        a = old[old_col].to_numpy().astype(float)
        b = new[new_col].to_numpy().astype(float)
        assert np.array_equal(np.isnan(a), np.isnan(b)), old_col
        np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-9, atol=1e-12, err_msg=old_col)


def _spec(func, **kwargs):
    from qam.config import Component

    return Component(func=func, kwargs=kwargs)


def test_zscore_matches_legacy_polars_expression():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3, 2))
    mask = np.ones(40, dtype=bool)
    new = registry.get("transform", "clip")(registry.get("transform", "cs_zscore")(x.copy(), mask), mask)

    # The legacy expression from UniversalDataProcessor.load_and_process, applied per (day, feature).
    long = pl.DataFrame({"Date": np.repeat(np.arange(3), 40), "v": x[:, :, 0].T.ravel()})
    old = long.with_columns(((pl.col("v") - pl.col("v").mean().over("Date")) / (pl.col("v").std().over("Date") + 1e-8)).clip(-3, 3))
    np.testing.assert_allclose(old["v"].to_numpy().reshape(3, 40).T, new[:, :, 0], rtol=1e-10)


# ---------------------------------------------------------------- model


def test_model_matches_legacy_weights_and_outputs():
    hybrid = load_legacy("models/hybrid.py")
    torch.manual_seed(0)
    old = hybrid.HybridAlphaRanker(num_stocks=12, input_dim=5, hidden_dim=32, dropout=0.2).eval()
    new = registry.get("model", "hybrid_lstm_tfm")(
        n_features=5, hidden_dim=32, lstm_layers=2, tfm_layers=2, heads=4, ff_dim=128, dropout=0.2
    ).eval()

    state = {k.replace("lstm.", "stock_lstm.", 1) if k.startswith("lstm.") else k: v for k, v in old.state_dict().items()}
    result = new.load_state_dict(state, strict=True)  # same parameter shapes = same architecture
    assert not result.missing_keys and not result.unexpected_keys

    x = torch.randn(3, 12, 20, 5)
    with torch.no_grad():
        expected = old(x)
        got = new({"x": x, "mask": torch.ones(3, 12, dtype=torch.bool)})
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------- loss


@pytest.mark.parametrize("margin", [0.1, 0.25])
def test_loss_matches_legacy(margin):
    losses = load_legacy("models/losses.py")
    torch.manual_seed(0)
    preds = torch.randn(4, 50)
    raw = torch.randn(4, 50) * 0.03
    zscored = (raw - raw.mean(1, keepdim=True)) / (raw.std(1, keepdim=True) + 1e-8)
    mask = torch.ones(4, 50, dtype=torch.bool)

    expected = losses.HybridAlphaLoss(margin=margin, decile=0.1, lambda_mse=1.0)(preds, zscored)
    new_raw = registry.get("loss", "hybrid_hinge")(margin=margin, target="raw")(preds, zscored, mask)
    new_z = registry.get("loss", "hybrid_hinge")(margin=margin, target="zscore")(preds, raw, mask)
    torch.testing.assert_close(new_raw, torch.as_tensor(expected), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(new_z, torch.as_tensor(expected), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------- splitter


def test_splitter_test_blocks_match_legacy_and_val_is_carved_from_train():
    splitters = load_legacy("data/splitters.py")
    dates = pd.bdate_range("2015-01-01", periods=1601)
    with redirect_stdout(io.StringIO()):
        old_folds = splitters.PurgedWalkForwardSplitter(n_folds=5, purge_days=5, embargo_days=20).split(pd.DataFrame({"Date": dates}))
    new_folds = registry.get("splitter", "purged_walk_forward")(n_folds=5, purge=5, embargo=20, val_frac=0.15).split(
        dates.to_numpy().astype("datetime64[D]")
    )
    for (old_train, old_test), new in zip(old_folds, new_folds):
        old_train_dates = old_train["Date"].to_numpy().astype("datetime64[D]")
        old_test_dates = old_test["Date"].to_numpy().astype("datetime64[D]")[59:]  # legacy prepends a 59-day burn-in
        # Same test block.
        assert np.array_equal(new.test, old_test_dates)
        # New train + purge + val is exactly the legacy train span.
        assert new.val[-1] == old_train_dates[-1]
        assert new.train[0] == old_train_dates[0]
        assert np.isin(new.train, old_train_dates).all() and np.isin(new.val, old_train_dates).all()
        gap = np.searchsorted(old_train_dates, new.val[0]) - np.searchsorted(old_train_dates, new.train[-1]) - 1
        assert gap == 5


# ---------------------------------------------------------------- metrics


def _old_metrics(tmp_path, df: pd.DataFrame) -> dict[str, float]:
    metrics = load_legacy("evaluation/metrics.py", stubs=("matplotlib", "matplotlib.pyplot"))
    path = tmp_path / "oos.parquet"
    df.rename(columns={"date": "Date", "score": "Pred_Alpha", "label": "Target_5D_Z"}).assign(Log_Ret=df["label"]).to_parquet(path)
    out = io.StringIO()
    with redirect_stdout(out):
        metrics.evaluate_alpha(str(path))
    parsed = {}
    for line in out.getvalue().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = float(value.split()[0].rstrip("%"))
    return parsed


def test_metrics_match_legacy(tmp_path):
    pytest.importorskip("scipy")
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2021-01-01", periods=60)
    rows = []
    for d in dates:
        signal = rng.normal(size=100)
        rows.append(pd.DataFrame({"date": d, "security_id": np.arange(100), "score": signal, "label": 0.01 * (0.2 * signal + rng.normal(size=100))}))
    df = pd.concat(rows, ignore_index=True)

    old = _old_metrics(tmp_path, df)
    mean_ic = registry.get("metric", "mean_ic")(df, min_stocks=21).value
    icir = registry.get("metric", "icir")(df, min_stocks=21, periods_per_year=252).value
    spread = registry.get("metric", "decile_spread")(df, n=10, cost_bps=10, min_stocks=21).value

    assert mean_ic == pytest.approx(old["Mean Purged IC"], abs=1e-5)
    assert icir == pytest.approx(old["Annualized ICIR"], abs=1e-2)
    # Legacy prints the annualised net spread in percent (mean per date * 252 * 100).
    assert spread * 252 * 100 == pytest.approx(old["Long/Short Spread"], abs=1e-2)


def test_decile_spread_uses_forward_label_not_same_day_return():
    """Intentional change: the legacy spread used same-day Log_Ret; the new metric uses the label."""
    df = pd.DataFrame({"date": pd.Timestamp("2021-01-04"), "security_id": range(30), "score": np.arange(30.0), "label": np.arange(30.0)})
    assert registry.get("metric", "decile_spread")(df, n=3, min_stocks=5).value == pytest.approx(20.0)


def test_validation_ic_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    from qam.train import spearman_per_date

    torch.manual_seed(0)
    scores, labels = torch.randn(5, 40), torch.randn(5, 40)
    mask = torch.ones(5, 40, dtype=torch.bool)
    expected = [stats.spearmanr(scores[b], labels[b])[0] for b in range(5)]
    np.testing.assert_allclose(spearman_per_date(scores, labels, mask), expected, rtol=1e-10)
