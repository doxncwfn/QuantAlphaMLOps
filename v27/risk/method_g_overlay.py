"""Method G incremental causal portfolio-level risk overlays (VolTgt, GrossCap, DD, CVaR, Constraints)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from v27.backtest.simulator import backtest_weights_daily
from v27.config import PortfolioBacktestConfig, RiskOverlayConfig
from v27.data.bundle import V7DataBundle
from v27.risk.constraints import apply_portfolio_constraints


def build_method_g_risk_overlays(
    base_res: dict[str, Any],
    prefix: str,
    bundle: V7DataBundle,
    bt_cfg: PortfolioBacktestConfig | None = None,
    risk_cfg: RiskOverlayConfig | None = None,
) -> list[dict[str, Any]]:
    """Apply G0..G5 causal risk-management and institutional constraint overlays."""
    bt_cfg = bt_cfg or PortfolioBacktestConfig()
    risk_cfg = risk_cfg or RiskOverlayConfig()
    W0 = base_res["weights"].copy()
    dates = bundle.dates
    T, _ = W0.shape
    r1_np = bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64)
    adv_np = bundle.adv20_df.to_numpy(dtype=np.float64)
    sig_np = bundle.sigma_prev_df.to_numpy(dtype=np.float64)
    rf_np = bundle.ff6_df["rf"].to_numpy(dtype=np.float64)
    beta_np = bundle.beta60_df.to_numpy(dtype=np.float64)

    port_r_raw = np.einsum("ij,ij->i", W0, r1_np)

    # 1. Causal Vol Targeting (20d/63d blend lagged t-1)
    r_ser = pd.Series(port_r_raw)
    v20 = r_ser.rolling(20, min_periods=10).std().shift(1).to_numpy(
        dtype=np.float64
    ) * math.sqrt(bt_cfg.ann_days)
    v63 = r_ser.rolling(63, min_periods=20).std().shift(1).to_numpy(
        dtype=np.float64
    ) * math.sqrt(bt_cfg.ann_days)
    vol_hat = np.where(
        np.isfinite(v20) & np.isfinite(v63),
        np.maximum(v20, 0.5 * v20 + 0.5 * v63),
        0.10,
    )
    scale_vt = np.clip(
        risk_cfg.target_vol / np.maximum(vol_hat, 0.025),
        risk_cfg.vol_scale_min,
        risk_cfg.vol_scale_max,
    )
    scale_vt[:25] = 1.0

    # 2. Causal Drawdown Scaling (continuous brake above 3.5% lagged DD)
    nav_raw = np.cumprod(1.0 + np.clip(port_r_raw, -0.5, 1.0))
    hwm_raw = np.maximum.accumulate(np.maximum(nav_raw, 1.0))
    dd_lag = np.zeros(T, dtype=np.float64)
    dd_lag[1:] = -(nav_raw[:-1] / hwm_raw[:-1] - 1.0)
    scale_dd = np.clip(1.0 - np.maximum(dd_lag - 0.035, 0.0) / 0.085, 0.45, 1.0)

    # 3. Causal Rolling 95% CVaR cap (daily target max expected shortfall <= 1.25%)
    cvar_lag = np.full(T, 0.01, dtype=np.float64)
    for t in range(65, T):
        win = port_r_raw[t - 63 : t]
        q05 = np.percentile(win, 5.0)
        cvar_lag[t] = max(-float(win[win <= q05].mean()), 1e-4)
    scale_cvar = np.clip(0.0125 / np.maximum(cvar_lag, 0.005), 0.50, 1.0)

    def _eval_overlay(tag: str, W_mod: np.ndarray) -> dict[str, Any]:
        bt = backtest_weights_daily(
            W_mod,
            r1_np,
            dates,
            cost_bps=bt_cfg.baseline_cost_bps,
            adv20_np=adv_np,
            sigma_np=sig_np,
            rf_np=rf_np,
            book_size=bt_cfg.book_size,
            short_borrow_ann=bt_cfg.short_borrow_ann,
            impact_eta=bt_cfg.impact_coeff,
            ann_days=bt_cfg.ann_days,
        )
        return {
            "name": f"{prefix}_{tag}",
            "group": "Method_G_RiskOverlay",
            "schedule": base_res["schedule"],
            "weights": W_mod,
            "bt": bt,
        }

    W_vt = apply_portfolio_constraints(W0 * scale_vt[:, None], max_single_w=0.035)
    W_vt_gc = apply_portfolio_constraints(
        W0 * scale_vt[:, None], max_single_w=0.035, max_gross=risk_cfg.max_gross
    )
    W_vt_dd = apply_portfolio_constraints(
        W0 * (scale_vt * scale_dd)[:, None],
        max_single_w=0.035,
        max_gross=risk_cfg.max_gross,
    )
    W_vt_cvar = apply_portfolio_constraints(
        W0 * (scale_vt * scale_dd * scale_cvar)[:, None],
        max_single_w=0.035,
        max_gross=risk_cfg.max_gross,
    )
    W_full = apply_portfolio_constraints(
        W0 * (scale_vt * scale_dd * scale_cvar)[:, None],
        beta60_np=beta_np,
        sectors=bundle.sectors,
        adv20_np=adv_np,
        r1d_np=r1_np,
        max_single_w=risk_cfg.max_single_w,
        max_gross=risk_cfg.max_gross,
        max_abs_beta=risk_cfg.max_abs_beta,
        max_sector_net=risk_cfg.max_sector_net,
        max_adv_frac=risk_cfg.max_adv_frac,
        book_size=bt_cfg.book_size,
    )
    return [
        _eval_overlay("G0_Base", W0),
        _eval_overlay("G1_VolTgt09", W_vt),
        _eval_overlay("G2_VolTgt09_Gross18", W_vt_gc),
        _eval_overlay("G3_VolTgt09_DDScale", W_vt_dd),
        _eval_overlay("G4_VolTgt09_DD_CVaRCap", W_vt_cvar),
        _eval_overlay("G5_FullConstrainedOverlay", W_full),
    ]
