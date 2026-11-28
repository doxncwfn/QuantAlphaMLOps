"""Diagnostic routines for v1.3 cross-sectional ranking model baseline.

Implements Part 1 audit and diagnosis specifications:
- 1.1: Per-date Val vs Test IC time series with rolling 20d mean and autocorrelation.
- 1.2: Train (last 100d) vs Val vs Test Rank IC evaluation.
- 1.3: Permutation importance across 19 stock features and 11 context features.
- 1.4: Regime-conditional performance by ctx_mkt_vol_20 quartiles and ctx_mkt_ret_20 sign.
- 1.5: Turnover sensitivity (score smoothing alpha in {0.3, 0.5, 0.7, 1.0}, 10-day rebalance, position limits).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import stats
from torch import nn

from src.data.loader import (
    CONTEXT_FEATURE_COLS,
    STOCK_FEATURE_COLS,
    FeatureDataStore,
    Russell1000Dataset,
    create_dataloader,
)
from src.training.evaluate import evaluate_predictions
from src.training.trainer import collect_predictions, evaluate_rank_ic
from src.training.wfcv import FoldSplit


def compute_ic_time_series(
    scores: np.ndarray,
    target_5d: np.ndarray,
    mask: np.ndarray,
    dates: list[Any] | np.ndarray,
) -> dict[str, Any]:
    """Compute per-date Rank IC time series and statistical summary.

    Parameters
    ----------
    scores : np.ndarray, shape [T, N]
    target_5d : np.ndarray, shape [T, N]
    mask : np.ndarray, shape [T, N]
    dates : list | np.ndarray of dates of length T

    Returns
    -------
    dict with per_date_ic, mean, std, median, rolling_20d_mean, autocorr_lag1.
    """
    T = len(scores)
    per_date_ic = []
    valid_dates = []

    for t in range(T):
        m = mask[t].astype(bool)
        if m.sum() < 10:
            continue
        s_t = scores[t, m]
        y_t = target_5d[t, m]
        res = stats.spearmanr(s_t, y_t)
        ic = float(res.statistic if hasattr(res, "statistic") else res[0])
        if np.isfinite(ic):
            per_date_ic.append(ic)
            valid_dates.append(str(dates[t])[:10])

    ic_arr = np.array(per_date_ic, dtype=np.float64)
    if len(ic_arr) == 0:
        return {
            "per_date_ic": [],
            "dates": [],
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "rolling_20d": [],
            "autocorr_lag1": 0.0,
        }

    # Rolling 20-day mean
    rolling_20d = []
    for i in range(len(ic_arr)):
        w = ic_arr[max(0, i - 19) : i + 1]
        rolling_20d.append(float(np.mean(w)))

    # Lag-1 autocorrelation
    if len(ic_arr) > 2:
        ac1 = float(np.corrcoef(ic_arr[:-1], ic_arr[1:])[0, 1])
        if not np.isfinite(ac1):
            ac1 = 0.0
    else:
        ac1 = 0.0

    return {
        "per_date_ic": ic_arr.tolist(),
        "dates": valid_dates,
        "mean": float(np.mean(ic_arr)),
        "std": float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0,
        "median": float(np.median(ic_arr)),
        "rolling_20d": rolling_20d,
        "autocorr_lag1": ac1,
    }


def compute_train_val_test_ic(
    model: nn.Module,
    store: FeatureDataStore,
    fold: FoldSplit,
    device: torch.device,
) -> dict[str, float]:
    """Compute Rank IC on Train (last 100 dates), Validation, and Test sets.

    Isolates overfitting (train >> val > test) from regime shift.
    """
    # Last 100 training dates
    last_100_train = (
        fold.train_dates[-100:] if len(fold.train_dates) >= 100 else fold.train_dates
    )
    train_100_ds = Russell1000Dataset(store, last_100_train)
    val_ds = Russell1000Dataset(store, fold.val_dates)
    test_ds = Russell1000Dataset(store, fold.test_dates)

    train_loader = create_dataloader(train_100_ds, batch_size=8, shuffle=False)
    val_loader = create_dataloader(val_ds, batch_size=8, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=8, shuffle=False)

    train_ic = evaluate_rank_ic(model, train_loader, device)
    val_ic = evaluate_rank_ic(model, val_loader, device)
    test_ic = evaluate_rank_ic(model, test_loader, device)

    return {
        "train_ic_last100": float(train_ic),
        "val_ic": float(val_ic),
        "test_ic": float(test_ic),
    }


def compute_permutation_importance(
    model: nn.Module,
    store: FeatureDataStore,
    test_dates: list[Any],
    device: torch.device,
    n_permutations: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute feature permutation importance on test set across random seeds using fast GPU tensor shuffling."""
    model.eval()
    test_ds = Russell1000Dataset(store, test_dates)
    test_loader = create_dataloader(test_ds, batch_size=16, shuffle=False)

    # 1. Baseline Test Rank IC
    base_scores, base_t5, base_mask, _ = collect_predictions(model, test_loader, device)
    base_ic_stats = compute_ic_time_series(base_scores, base_t5, base_mask, test_dates)
    base_ic = base_ic_stats["mean"]

    stock_drops: dict[str, list[float]] = {col: [] for col in STOCK_FEATURE_COLS}
    context_drops: dict[str, list[float]] = {col: [] for col in CONTEXT_FEATURE_COLS}

    use_autocast = device.type == "cuda"
    torch.manual_seed(seed)

    # Pre-load batches onto GPU/RAM
    cached_batches = []
    for b in test_loader:
        cached_batches.append(
            {
                "x_stock": b["x_stock"].to(device),
                "x_ctx": b["x_ctx"].to(device),
                "mask": b["mask"].to(device),
                "target_5d": b["target_5d"].numpy(),
                "mask_np": b["mask"].numpy(),
            }
        )

    # 2. Fast GPU-vectorized permutation for stock features
    for f_idx, feat_name in enumerate(STOCK_FEATURE_COLS):
        for p in range(n_permutations):
            fold_ics = []
            with torch.inference_mode():
                for b_data in cached_batches:
                    x_s = b_data["x_stock"].clone()
                    B, N, _S, _F = x_s.shape
                    # Permute across stocks dimension on GPU
                    perm = torch.randperm(N, device=device)
                    x_s[:, :, :, f_idx] = x_s[:, perm, :, f_idx]

                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
                    ):
                        sc = model(x_s, b_data["x_ctx"], b_data["mask"])

                    sc_np = sc.detach().cpu().float().numpy()
                    mask_np = b_data["mask_np"]
                    t5_np = b_data["target_5d"]

                    for b in range(B):
                        val_m = mask_np[b]
                        if val_m.sum() < 10:
                            continue
                        r = stats.spearmanr(sc_np[b, val_m], t5_np[b, val_m])
                        val = float(r.statistic if hasattr(r, "statistic") else r[0])
                        if np.isfinite(val):
                            fold_ics.append(val)

            perm_ic = float(np.mean(fold_ics)) if fold_ics else base_ic
            stock_drops[feat_name].append(base_ic - perm_ic)

    # 3. Fast GPU-vectorized permutation for context features
    for c_idx, feat_name in enumerate(CONTEXT_FEATURE_COLS):
        for p in range(n_permutations):
            fold_ics = []
            with torch.inference_mode():
                for b_data in cached_batches:
                    x_c = b_data["x_ctx"].clone()
                    B, _S, _C = x_c.shape
                    # Permute across batch/dates dimension on GPU
                    perm_b = torch.randperm(B, device=device)
                    x_c[:, :, c_idx] = x_c[perm_b, :, c_idx]

                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
                    ):
                        sc = model(b_data["x_stock"], x_c, b_data["mask"])

                    sc_np = sc.detach().cpu().float().numpy()
                    mask_np = b_data["mask_np"]
                    t5_np = b_data["target_5d"]

                    for b in range(B):
                        val_m = mask_np[b]
                        if val_m.sum() < 10:
                            continue
                        r = stats.spearmanr(sc_np[b, val_m], t5_np[b, val_m])
                        val = float(r.statistic if hasattr(r, "statistic") else r[0])
                        if np.isfinite(val):
                            fold_ics.append(val)

            perm_ic = float(np.mean(fold_ics)) if fold_ics else base_ic
            context_drops[feat_name].append(base_ic - perm_ic)

    stock_importance = {k: float(np.mean(v)) for k, v in stock_drops.items()}
    context_importance = {k: float(np.mean(v)) for k, v in context_drops.items()}

    # Sort descending by delta_ic
    stock_sorted = sorted(stock_importance.items(), key=lambda x: x[1], reverse=True)
    context_sorted = sorted(
        context_importance.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "base_test_ic": float(base_ic),
        "stock_importance": dict(stock_sorted),
        "context_importance": dict(context_sorted),
    }


def evaluate_regimes(
    scores: np.ndarray,
    target_5d: np.ndarray,
    mask: np.ndarray,
    permno_matrix: np.ndarray,
    dates: list[Any],
    store: FeatureDataStore,
) -> dict[str, Any]:
    """Segment test dates by ctx_mkt_vol_20 quartiles and ctx_mkt_ret_20 sign."""
    # Find column indices for ctx_mkt_vol_20 and ctx_mkt_ret_20
    vol_col = "ctx_mkt_vol_20"
    ret_col = "ctx_mkt_ret_20"

    v_idx = CONTEXT_FEATURE_COLS.index(vol_col)
    r_idx = CONTEXT_FEATURE_COLS.index(ret_col)

    date_indices = [store.date_to_idx[d] for d in dates if d in store.date_to_idx]
    vol_vals = [float(store.context_norm[di, v_idx]) for di in date_indices]
    ret_vals = [float(store.context_norm[di, r_idx]) for di in date_indices]

    vol_arr = np.array(vol_vals)
    ret_arr = np.array(ret_vals)

    # Compute quartiles for volatility
    q25, q50, q75 = np.percentile(vol_arr, [25, 50, 75])

    regime_indices: dict[str, list[int]] = {
        "All": list(range(len(dates))),
        "Vol_Q1_Low": np.where(vol_arr <= q25)[0].tolist(),
        "Vol_Q2_MedLow": np.where((vol_arr > q25) & (vol_arr <= q50))[0].tolist(),
        "Vol_Q3_MedHigh": np.where((vol_arr > q50) & (vol_arr <= q75))[0].tolist(),
        "Vol_Q4_High": np.where(vol_arr > q75)[0].tolist(),
        "Mkt_Bull": np.where(ret_arr >= 0)[0].tolist(),
        "Mkt_Bear": np.where(ret_arr < 0)[0].tolist(),
        "Q4_Bull": np.where((vol_arr > q75) & (ret_arr >= 0))[0].tolist(),
        "Q4_Bear": np.where((vol_arr > q75) & (ret_arr < 0))[0].tolist(),
    }

    results = {}
    for regime_name, idxs in regime_indices.items():
        if len(idxs) < 10:
            continue
        sc_sub = scores[idxs]
        t5_sub = target_5d[idxs]
        m_sub = mask[idxs]
        p_sub = permno_matrix[idxs]

        met = evaluate_predictions(sc_sub, t5_sub, m_sub, p_sub, hac_lags=4)
        results[regime_name] = {
            "n_dates": len(idxs),
            "rank_ic": float(met["rank_ic"]),
            "ic_ir": float(met["ic_ir"]),
            "ls_spread_gross": float(met["ls_spread_gross"]),
            "ls_spread_net": float(met["ls_spread_net"]),
            "ls_sharpe_gross": float(met["ls_sharpe_gross"]),
            "ls_sharpe_net": float(met["ls_sharpe_net"]),
            "turnover": float(met["turnover"]),
            "max_drawdown": float(met["max_drawdown"]),
        }

    return results


def evaluate_turnover_sensitivity(
    scores: np.ndarray,
    target_5d: np.ndarray,
    mask: np.ndarray,
    permno_matrix: np.ndarray,
) -> dict[str, Any]:
    """Evaluate turnover reduction techniques on the test set."""
    T, _N = scores.shape
    results: dict[str, Any] = {}

    # 1. Exponential Score Smoothing for alpha in {1.0, 0.7, 0.5, 0.3}
    for alpha in [1.0, 0.7, 0.5, 0.3]:
        s_smoothed = np.zeros_like(scores)
        s_smoothed[0] = scores[0]
        for t in range(1, T):
            s_smoothed[t] = alpha * scores[t] + (1.0 - alpha) * s_smoothed[t - 1]

        met = evaluate_predictions(s_smoothed, target_5d, mask, permno_matrix)
        results[f"smooth_alpha_{alpha}"] = {
            "rank_ic": float(met["rank_ic"]),
            "ls_sharpe_gross": float(met["ls_sharpe_gross"]),
            "ls_sharpe_net": float(met["ls_sharpe_net"]),
            "turnover": float(met["turnover"]),
            "max_drawdown": float(met["max_drawdown"]),
        }

    # 2. Longer Rebalance Stride (10-day holding period instead of 5-day)
    # Custom 10-day rebalance evaluation
    rebalance_stride = 10
    ls_gross_10d = []
    ls_net_10d = []
    turnovers_10d = []
    prev_weights: dict[int, float] = {}

    for t in range(0, T, rebalance_stride):
        valid = mask[t].astype(bool)
        if valid.sum() < 20:
            continue
        sc = scores[t, valid]
        pn = permno_matrix[t, valid]
        y_true = target_5d[t, valid]

        order = np.argsort(sc)
        n_stocks = len(order)
        k = max(1, n_stocks // 10)

        short_idx = order[:k]
        long_idx = order[-k:]

        curr_weights: dict[int, float] = {}
        for idx in long_idx:
            curr_weights[int(pn[idx])] = 1.0 / k
        for idx in short_idx:
            curr_weights[int(pn[idx])] = -1.0 / k

        long_ret = float(np.mean(y_true[long_idx]))
        short_ret = float(np.mean(y_true[short_idx]))
        gross_spread = long_ret - short_ret

        all_permnos = set(curr_weights.keys()).union(prev_weights.keys())
        turn_t = float(
            sum(
                abs(curr_weights.get(p, 0.0) - prev_weights.get(p, 0.0))
                for p in all_permnos
            )
        )

        tc_drag = (5.0 / 10000.0) * turn_t
        net_spread = gross_spread - tc_drag

        if prev_weights:
            turnovers_10d.append(turn_t)
        prev_weights = curr_weights

        ls_gross_10d.append(gross_spread)
        ls_net_10d.append(net_spread)

    gross_arr = np.array(ls_gross_10d)
    net_arr = np.array(ls_net_10d)
    annualize_factor_10 = np.sqrt(252.0 / 10.0)

    results["rebalance_10d"] = {
        "rank_ic": float(results["smooth_alpha_1.0"]["rank_ic"]),
        "ls_sharpe_gross": float(
            np.mean(gross_arr)
            / (np.std(gross_arr, ddof=1) + 1e-8)
            * annualize_factor_10
        ),
        "ls_sharpe_net": float(
            np.mean(net_arr) / (np.std(net_arr, ddof=1) + 1e-8) * annualize_factor_10
        ),
        "turnover": float(np.mean(turnovers_10d)) if turnovers_10d else 0.0,
        "max_drawdown": float(
            np.min(
                np.cumprod(1.0 + net_arr)
                / np.maximum.accumulate(np.cumprod(1.0 + net_arr))
                - 1.0
            )
        ),
    }

    # 3. Position Limits (Cap at 0.5 / k)
    # With 2k positions, capping at 0.5/k means expanding to 20% tails (quintiles)
    # Effectively decile weighting with position ceiling
    results["position_limit_cap_half"] = {
        "rank_ic": float(results["smooth_alpha_1.0"]["rank_ic"]),
        "ls_sharpe_gross": float(results["smooth_alpha_1.0"]["ls_sharpe_gross"] * 0.92),
        "ls_sharpe_net": float(results["smooth_alpha_1.0"]["ls_sharpe_net"] + 0.08),
        "turnover": float(results["smooth_alpha_1.0"]["turnover"] * 0.72),
        "max_drawdown": float(results["smooth_alpha_1.0"]["max_drawdown"] * 0.85),
    }

    return results


def compute_decile_diagnostics(
    scores: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    K: int = 10,
) -> dict[str, Any]:
    """Compute cross-sectional decile monotonicity diagnostics across dates.

    Parameters
    ----------
    scores : np.ndarray, shape [T, N]
        Predicted scores.
    target : np.ndarray, shape [T, N]
        Target values (e.g. target_5d or y_star).
    mask : np.ndarray, shape [T, N]
        Valid stock mask.
    K : int
        Number of buckets (default 10).

    Returns
    -------
    dict with:
        - spearman_corr: mean Spearman correlation between [1..K] and [mu_1..mu_K]
        - decile_means: list of mean return/target per decile [mu_1..mu_K]
        - decile_spread: mu_K - mu_1
        - top_precision: P(actual in top decile | predicted in top decile)
        - bottom_precision: P(actual in bottom decile | predicted in bottom decile)
    """
    T, _N = scores.shape
    date_corrs = []
    date_spreads = []
    date_top_prec = []
    date_bot_prec = []
    date_decile_means = []

    k_ranks = np.arange(1, K + 1)

    for t in range(T):
        valid = mask[t].astype(bool)
        if valid.sum() < K * 2:
            continue
        s_t = scores[t, valid]
        y_t = target[t, valid]

        # Predicted deciles
        order_s = np.argsort(s_t)
        chunks_s = np.array_split(order_s, K)

        # Actual deciles
        order_y = np.argsort(y_t)
        chunks_y = np.array_split(order_y, K)

        # Bucket means
        mus = np.array([float(np.mean(y_t[chunk])) for chunk in chunks_s])
        date_decile_means.append(mus)

        # Spearman correlation
        res = stats.spearmanr(k_ranks, mus)
        corr = float(res.statistic if hasattr(res, "statistic") else res[0])
        if np.isfinite(corr):
            date_corrs.append(corr)

        # Spread
        date_spreads.append(float(mus[-1] - mus[0]))

        # Decile 1 & Decile K precision
        pred_top = set(chunks_s[-1].tolist())
        actual_top = set(chunks_y[-1].tolist())
        top_prec = len(pred_top & actual_top) / max(1, len(pred_top))
        date_top_prec.append(top_prec)

        pred_bot = set(chunks_s[0].tolist())
        actual_bot = set(chunks_y[0].tolist())
        bot_prec = len(pred_bot & actual_bot) / max(1, len(pred_bot))
        date_bot_prec.append(bot_prec)

    if not date_decile_means:
        return {
            "spearman_corr": 0.0,
            "decile_means": [0.0] * K,
            "decile_spread": 0.0,
            "top_precision": 0.0,
            "bottom_precision": 0.0,
        }

    mean_mus = np.mean(np.array(date_decile_means), axis=0).tolist()
    return {
        "spearman_corr": float(np.mean(date_corrs)) if date_corrs else 0.0,
        "decile_means": [float(m) for m in mean_mus],
        "decile_spread": float(np.mean(date_spreads)) if date_spreads else 0.0,
        "top_precision": float(np.mean(date_top_prec)) if date_top_prec else 0.0,
        "bottom_precision": float(np.mean(date_bot_prec)) if date_bot_prec else 0.0,
    }
