"""End-to-end strictly causal turnover-reduction & cost-resilient optimization pipeline for Stock Ranking v27."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd

from v27.backtest.simulator import backtest_weights_daily
from v27.config import CANDIDATES_24, EXPERIMENTS_V7, PortfolioBacktestConfig
from v27.cost_resilient.optimizers import (
    apply_l1_no_trade_band,
    apply_partial_rebalance,
    apply_smooth_method_g_overlay,
    apply_turnover_throttle,
    build_betaneu_invvol_hold,
    solve_admm_tail_l1,
    solve_admm_turnover_projection,
)
from v27.cost_resilient.signals import (
    build_signal_library,
    compute_signal_persistence_and_decay_table,
)
from v27.data.bundle import V7DataBundle
from v27.experiments.common import run_consolidated_strategy
from v27.experiments.method_f_hierarchical import build_method_f_hierarchical
from v27.optimization.estimators import bayes_stein_mu, ledoit_wolf_cov
from v27.optimization.schedule import causal_wf_schedule as _causal_wf_schedule
from v27.optimization.schedule import static_schedule as _static_schedule
from v27.optimization.solvers import (
    solve_erc,
    solve_min_var,
    solve_regularized_return_risk,
)
from v27.portfolio.consolidation import consolidate_positions
from v27.portfolio.constructors import build_8_strategies_for_score, strat_top10_buf
from v27.portfolio.neutralization import neutralize_beta
from v27.risk.constraints import apply_portfolio_constraints
from v27.risk.method_g_overlay import build_method_g_risk_overlays

_BT_CFG = PortfolioBacktestConfig()
ANN_DAYS = float(_BT_CFG.ann_days)
BASELINE_COST_BPS = float(_BT_CFG.baseline_cost_bps)
WARMUP_DAYS = _BT_CFG.warmup_days


def _eval_entry(
    name: str,
    group: str,
    W: np.ndarray,
    bundle: V7DataBundle,
    r1_np: np.ndarray,
    adv_np: np.ndarray,
    sig_np: np.ndarray,
    rf_np: np.ndarray,
    schedule: pd.DataFrame | None = None,
) -> dict[str, Any]:
    bt2 = backtest_weights_daily(
        W,
        r1_np,
        bundle.dates,
        cost_bps=2.0,
        adv20_np=adv_np,
        sigma_np=sig_np,
        rf_np=rf_np,
        inception_zero_trade=False,
    )
    bt5 = backtest_weights_daily(
        W,
        r1_np,
        bundle.dates,
        cost_bps=5.0,
        adv20_np=adv_np,
        sigma_np=sig_np,
        rf_np=rf_np,
        inception_zero_trade=False,
    )
    return {
        "name": name,
        "group": group,
        "weights": W,
        "schedule": schedule,
        "bt2": bt2,
        "bt5": bt5,
    }


def build_v27_baselines(
    bundle: V7DataBundle,
    signals: dict[str, pd.DataFrame],
    r1_np: np.ndarray,
    adv_np: np.ndarray,
    sig_np: np.ndarray,
    rf_np: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], np.ndarray]:
    """Reconstruct exact frozen v27 baselines directly from V7DataBundle without mutating v7_ensemble."""
    t0 = time.time()
    cand_weights: dict[str, np.ndarray] = {}
    ret_net_map: dict[str, pd.Series] = {}

    for exp in EXPERIMENTS_V7:
        w8 = build_8_strategies_for_score(
            bundle.scores[exp], bundle.ret1d_df, bundle.sigma_prev_df, bundle.beta60_df
        )
        for sname, wdf in w8.items():
            cid = f"{exp}__{sname}"
            W_np = wdf.to_numpy(dtype=np.float64)
            cand_weights[cid] = W_np
            bt = backtest_weights_daily(
                W_np, r1_np, bundle.dates, cost_bps=BASELINE_COST_BPS
            )
            ret_net_map[cid] = bt["net"]

    ret_net_df = pd.DataFrame(ret_net_map, index=bundle.dates)

    # Exact A1_EW_All24
    sch_ew24 = _static_schedule(
        bundle.dates, {c: 1.0 / len(CANDIDATES_24) for c in CANDIDATES_24}
    )
    res_ew24 = run_consolidated_strategy(
        "A1_EW_All24", "v27_Baseline", sch_ew24, cand_weights, bundle
    )

    # Exact F4_Hier_Dual1d5d_ERC + G_HierBest_G5_FullConstrainedOverlay
    f_list = build_method_f_hierarchical(bundle, cand_weights, ret_net_df)
    res_f4 = next(x for x in f_list if x["name"] == "F4_Hier_Dual1d5d_ERC")
    g_f4_list = build_method_g_risk_overlays(res_f4, "G_HierBest", bundle)
    res_g5 = next(
        x for x in g_f4_list if x["name"] == "G_HierBest_G5_FullConstrainedOverlay"
    )

    # Exact B5_ScoreEW_Rank_All3__top10_BetaNeu_InvVol_Buf + G_ScoreBest_G1_VolTgt09
    sc_b5 = signals["rk_ew3"]
    inv_vol_b5 = (1.0 / bundle.sigma_prev_df).where(sc_b5.notna()).fillna(0.0)
    W_b5 = strat_top10_buf(
        neutralize_beta(sc_b5, bundle.beta60_df), inv_vol_b5
    ).to_numpy(dtype=np.float64)
    bt_b5 = backtest_weights_daily(
        W_b5,
        r1_np,
        bundle.dates,
        cost_bps=BASELINE_COST_BPS,
        adv20_np=adv_np,
        sigma_np=sig_np,
        rf_np=rf_np,
    )
    res_b5 = {
        "name": "B5_ScoreEW_Rank_All3",
        "group": "v27_Baseline",
        "schedule": None,
        "weights": W_b5,
        "bt": bt_b5,
    }
    g_b5_list = build_method_g_risk_overlays(res_b5, "G_ScoreBest", bundle)
    res_g1 = next(x for x in g_b5_list if x["name"] == "G_ScoreBest_G1_VolTgt09")

    baselines = {
        "v27_G_HierBest_G5": _eval_entry(
            "v27_G_HierBest_G5",
            "v27_Baseline",
            res_g5["weights"],
            bundle,
            r1_np,
            adv_np,
            sig_np,
            rf_np,
            res_g5["schedule"],
        ),
        "v27_G_ScoreBest_G1": _eval_entry(
            "v27_G_ScoreBest_G1",
            "v27_Baseline",
            res_g1["weights"],
            bundle,
            r1_np,
            adv_np,
            sig_np,
            rf_np,
        ),
        "v27_risk_t5_BetaNeu_Buf": _eval_entry(
            "v27_risk_t5_BetaNeu_Buf",
            "v27_Baseline",
            cand_weights["risk_t5__top10_BetaNeu_InvVol_Buf"],
            bundle,
            r1_np,
            adv_np,
            sig_np,
            rf_np,
        ),
        "v27_A1_EW_All24": _eval_entry(
            "v27_A1_EW_All24",
            "v27_Baseline",
            res_ew24["weights"],
            bundle,
            r1_np,
            adv_np,
            sig_np,
            rf_np,
            res_ew24["schedule"],
        ),
        "v27_F4_Hier_Dual1d5d": _eval_entry(
            "v27_F4_Hier_Dual1d5d",
            "v27_Baseline",
            res_f4["weights"],
            bundle,
            r1_np,
            adv_np,
            sig_np,
            rf_np,
            res_f4["schedule"],
        ),
        "v27_B5_ScoreEW3": _eval_entry(
            "v27_B5_ScoreEW3",
            "v27_Baseline",
            W_b5,
            bundle,
            r1_np,
            adv_np,
            sig_np,
            rf_np,
        ),
    }
    print(f"[Baselines] Reconstructed 6 exact v27 baselines in {time.time() - t0:.1f}s")
    return baselines, res_f4["weights"]


def run_cost_resilient_experiments(
    bundle: V7DataBundle,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    """Run all 3 research pillars, ablations, and cost-aware portfolio-of-strategies ensembles."""
    t_start = time.time()
    r1_np = bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64)
    adv_np = bundle.adv20_df.to_numpy(dtype=np.float64)
    sig_np = bundle.sigma_prev_df.to_numpy(dtype=np.float64)
    rf_np = bundle.ff6_df["rf"].to_numpy(dtype=np.float64)
    beta_np = bundle.beta60_df.to_numpy(dtype=np.float64)
    mom_np = bundle.mom252_df.to_numpy(dtype=np.float64)
    inv_vol_full = (1.0 / bundle.sigma_prev_df).fillna(0.0)

    # ------------------------------------------------------------------
    # Pillar 1: Signal Library + Persistence/Decay Audit
    # ------------------------------------------------------------------
    signals = build_signal_library(bundle)
    sig_decay_df = compute_signal_persistence_and_decay_table(
        bundle, signals, warmup=WARMUP_DAYS
    )
    baselines, W_f4 = build_v27_baselines(bundle, signals, r1_np, adv_np, sig_np, rf_np)

    entries: dict[str, dict[str, Any]] = dict(baselines)

    def _add(
        name: str, group: str, W: np.ndarray, sched: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        e = _eval_entry(name, group, W, bundle, r1_np, adv_np, sig_np, rf_np, sched)
        entries[name] = e
        return e

    # Pillar 1 Evaluation (uniform top10_BetaNeu_InvVol_Buf 10/20 construction to isolate signal effect)
    p1_specs = [
        ("P1_Sig_risk_t5_EMA3", signals["z_r5_ema3"]),
        ("P1_Sig_EW3_Z_Raw", signals["z_ew3"]),
        ("P1_Sig_All3_EMA3", signals["z_all3_ema3"]),
        ("P1_Sig_All3_EMA5", signals["z_all3_ema5"]),
        ("P1_Sig_All3_Kalman", signals["z_all3_kal"]),
        ("P1_Sig_R5R1_EMA3", signals["z_r5r1_ema3"]),
        ("P1_Sig_Slow70_EMA3", signals["z_slow70_ema3"]),
    ]
    for nm, sc in p1_specs:
        sc_bn = neutralize_beta(sc, bundle.beta60_df)
        iv = inv_vol_full.where(sc.notna()).fillna(0.0)
        W = strat_top10_buf(sc_bn, iv, q_in=0.10, q_out=0.20).to_numpy(dtype=np.float64)
        _add(nm, "Pillar1_SignalSmoothing", W)
    print(f"[Pillar 1] Evaluated {len(p1_specs)} signal-smoothing variants")

    # ------------------------------------------------------------------
    # Pillar 2: Portfolio Construction & Rebalancing Mechanics
    # ------------------------------------------------------------------
    sc_p2 = signals["z_all3_ema3"]
    sc_p2_bn = neutralize_beta(sc_p2, bundle.beta60_df)
    iv_p2 = inv_vol_full.where(sc_p2.notna()).fillna(0.0)

    W_buf_10_25 = strat_top10_buf(sc_p2_bn, iv_p2, q_in=0.10, q_out=0.25).to_numpy(
        dtype=np.float64
    )
    W_buf_10_30 = strat_top10_buf(sc_p2_bn, iv_p2, q_in=0.10, q_out=0.30).to_numpy(
        dtype=np.float64
    )
    W_buf_15_30 = strat_top10_buf(sc_p2_bn, iv_p2, q_in=0.15, q_out=0.30).to_numpy(
        dtype=np.float64
    )

    _add("P2_Buf_10_25", "Pillar2_RebalMechanics", W_buf_10_25)
    _add("P2_Buf_10_30", "Pillar2_RebalMechanics", W_buf_10_30)
    _add("P2_Buf_15_30", "Pillar2_RebalMechanics", W_buf_15_30)
    _add(
        "P2_BetaNeu_InvVol_HOLD3",
        "Pillar2_RebalMechanics",
        build_betaneu_invvol_hold(sc_p2, bundle, hold_days=3),
    )
    _add(
        "P2_BetaNeu_InvVol_HOLD5",
        "Pillar2_RebalMechanics",
        build_betaneu_invvol_hold(sc_p2, bundle, hold_days=5),
    )
    _add(
        "P2_Buf10_25_NTB10bp",
        "Pillar2_RebalMechanics",
        apply_l1_no_trade_band(W_buf_10_25, r1_np, band=0.0010),
    )
    _add(
        "P2_Buf10_25_NTB20bp",
        "Pillar2_RebalMechanics",
        apply_l1_no_trade_band(W_buf_10_25, r1_np, band=0.0020),
    )
    _add(
        "P2_Buf10_25_GP_g40",
        "Pillar2_RebalMechanics",
        apply_partial_rebalance(W_buf_10_25, r1_np, gamma=0.40),
    )
    _add(
        "P2_Buf10_25_Throttle20pct",
        "Pillar2_RebalMechanics",
        apply_turnover_throttle(W_buf_10_25, r1_np, max_daily_1way=0.20),
    )
    _add(
        "P2_Buf10_25_SmoothG5_NTB10bp",
        "Pillar2_RebalMechanics",
        apply_smooth_method_g_overlay(
            W_buf_10_25,
            bundle,
            r1_np,
            adv_np,
            beta_np,
            vol_target=0.09,
            scale_deadband=0.10,
            scale_ema=0.30,
            post_ntb=0.0010,
        ),
    )
    print("[Pillar 2] Evaluated 10 buffer/HOLD/NTB/GP/throttle rebalancing variants")

    # ------------------------------------------------------------------
    # Pillar 3A: Direct Woodbury-ADMM Tail-Alpha + Explicit L1 Cost QP
    # ------------------------------------------------------------------
    def _run_admm(
        sc_df: pd.DataFrame,
        l1_bps: float = 4.0,
        dz: float = 0.80,
        sec: bool = True,
        vol: bool = True,
        mom: bool = False,
        vt: float | None = None,
        ddb: bool = False,
    ) -> np.ndarray:
        return solve_admm_tail_l1(
            sc_df,
            sig_np,
            beta_np,
            bundle.sectors,
            r1_np,
            mom_np=mom_np,
            adv_np=adv_np,
            tail_deadzone=dz,
            base_gross=1.80,
            l1_cost_bps=l1_bps,
            neutralize_sectors=sec,
            neutralize_vol=vol,
            neutralize_mom=mom,
            vol_target_ann=vt,
            dd_brake=ddb,
            max_single_w=0.025,
            max_gross=2.00,
        )

    # L1 Cost Penalty Pareto Sweep on z_all3_ema3 (0, 2, 3, 4, 5, 8 bps)
    for l1bp in (0.0, 2.0, 3.0, 4.0, 5.0, 8.0):
        _add(
            f"P3A_ADMM_All3Sm_L1_{int(l1bp)}bp",
            "Pillar3A_ADMM_L1_QP",
            _run_admm(signals["z_all3_ema3"], l1_bps=l1bp),
        )

    # Structural & Factor-Neutrality Ablations on z_all3_ema3
    _add(
        "P3A_Abl_BetaOnly_L1_3bp",
        "Pillar3A_Ablation",
        _run_admm(signals["z_all3_ema3"], l1_bps=3.0, sec=False, vol=False),
    )
    _add(
        "P3A_Abl_BetaSec_L1_3bp",
        "Pillar3A_Ablation",
        _run_admm(signals["z_all3_ema3"], l1_bps=3.0, sec=True, vol=False),
    )
    _add(
        "P3A_Abl_BetaSecVolMom_L1_3bp",
        "Pillar3A_Ablation",
        _run_admm(signals["z_all3_ema3"], l1_bps=3.0, sec=True, vol=True, mom=True),
    )
    _add(
        "P3A_Abl_NoDeadzone_dz0_L1_4bp",
        "Pillar3A_Ablation",
        _run_admm(signals["z_all3_ema3"], l1_bps=4.0, dz=0.0),
    )
    _add(
        "P3A_Abl_InsideVolTgt8_L1_3bp",
        "Pillar3A_Ablation",
        _run_admm(signals["z_all3_ema3"], l1_bps=3.0, vt=0.08, ddb=True),
    )

    # Multi-Horizon & Single-Model ADMM Sleeves (exact Sector + Beta + Vol Neutral)
    _add(
        "P3A_ADMM_EW3_L1_4bp",
        "Pillar3A_ADMM_L1_QP",
        _run_admm(signals["z_ew3"], l1_bps=4.0),
    )
    _add(
        "P3A_ADMM_EW3_L1_5bp",
        "Pillar3A_ADMM_L1_QP",
        _run_admm(signals["z_ew3"], l1_bps=5.0),
    )
    _add(
        "P3A_ADMM_Kal3_L1_3bp",
        "Pillar3A_ADMM_L1_QP",
        _run_admm(signals["z_all3_kal"], l1_bps=3.0),
    )
    _add(
        "P3A_ADMM_Kal3_L1_4bp",
        "Pillar3A_ADMM_L1_QP",
        _run_admm(signals["z_all3_kal"], l1_bps=4.0),
    )
    _add(
        "P3A_ADMM_R5R1_L1_3bp",
        "Pillar3A_ADMM_L1_QP",
        _run_admm(signals["z_r5r1_ema3"], l1_bps=3.0),
    )
    _add(
        "P3A_ADMM_R5R1_L1_4bp",
        "Pillar3A_ADMM_L1_QP",
        _run_admm(signals["z_r5r1_ema3"], l1_bps=4.0),
    )

    _add(
        "P3A_SLV_r5_L1_3bp",
        "Pillar3A_ADMM_Sleeve",
        _run_admm(signals["z_r5"], l1_bps=3.0),
    )
    _add(
        "P3A_SLV_r5_L1_5bp",
        "Pillar3A_ADMM_Sleeve",
        _run_admm(signals["z_r5"], l1_bps=5.0),
    )
    _add(
        "P3A_SLV_r1EMA3_L1_4bp",
        "Pillar3A_ADMM_Sleeve",
        _run_admm(signals["z_r1_ema3"], l1_bps=4.0),
    )
    _add(
        "P3A_SLV_rawEMA3_L1_4bp",
        "Pillar3A_ADMM_Sleeve",
        _run_admm(signals["z_raw_ema3"], l1_bps=4.0),
    )
    _add(
        "P3A_SLV_r1_L1_5bp",
        "Pillar3A_ADMM_Sleeve",
        _run_admm(signals["z_r1"], l1_bps=5.0),
    )
    _add(
        "P3A_SLV_raw_L1_5bp",
        "Pillar3A_ADMM_Sleeve",
        _run_admm(signals["z_raw"], l1_bps=5.0),
    )
    print(
        "[Pillar 3A] Evaluated 23 Woodbury-ADMM L1-QP sleeves and structural ablations"
    )

    # ------------------------------------------------------------------
    # Pillar 3B: Turnover-Aware Method G Overlays on v27 F4_Hier_Dual1d5d_ERC
    # ------------------------------------------------------------------
    W_f4_sm = apply_smooth_method_g_overlay(
        W_f4,
        bundle,
        r1_np,
        adv_np,
        beta_np,
        vol_target=0.09,
        scale_deadband=0.10,
        scale_ema=0.30,
        post_ntb=0.0,
    )
    W_f4_ntb4 = apply_smooth_method_g_overlay(
        W_f4,
        bundle,
        r1_np,
        adv_np,
        beta_np,
        vol_target=0.09,
        scale_deadband=0.10,
        scale_ema=0.30,
        post_ntb=0.0004,
    )
    W_f4_ntb8 = apply_smooth_method_g_overlay(
        W_f4,
        bundle,
        r1_np,
        adv_np,
        beta_np,
        vol_target=0.09,
        scale_deadband=0.10,
        scale_ema=0.30,
        post_ntb=0.0008,
    )
    W_f4_ntb12 = apply_smooth_method_g_overlay(
        W_f4,
        bundle,
        r1_np,
        adv_np,
        beta_np,
        vol_target=0.09,
        scale_deadband=0.10,
        scale_ema=0.30,
        post_ntb=0.0012,
    )
    # Smoothly vol-targeted F4 passed into ADMM turnover-penalized constraint projection
    pr_f4 = np.einsum("ij,ij->i", W_f4, r1_np)
    v20 = pd.Series(pr_f4).rolling(20, min_periods=10).std().shift(1).to_numpy(
        dtype=np.float64
    ) * math.sqrt(ANN_DAYS)
    v63 = pd.Series(pr_f4).rolling(63, min_periods=20).std().shift(1).to_numpy(
        dtype=np.float64
    ) * math.sqrt(ANN_DAYS)
    vh = np.where(
        np.isfinite(v20) & np.isfinite(v63),
        np.maximum(v20, 0.5 * v20 + 0.5 * v63),
        0.12,
    )
    sc_raw = np.clip(0.09 / np.maximum(vh, 0.025), 0.35, 1.35)
    sc_sm = np.ones_like(sc_raw)
    cur = 1.0
    for t in range(25, len(sc_raw)):
        if abs(sc_raw[t] - cur) / cur > 0.10:
            cur = 0.35 * sc_raw[t] + 0.65 * cur
        sc_sm[t] = cur
    W_f4_vt_sm = W_f4 * sc_sm[:, None]
    W_f4_proj10 = solve_admm_turnover_projection(
        W_f4_vt_sm, beta_np, bundle.sectors, r1_np, adv_np=adv_np, l1_weight=0.0010
    )
    W_f4_proj15 = solve_admm_turnover_projection(
        W_f4_vt_sm, beta_np, bundle.sectors, r1_np, adv_np=adv_np, l1_weight=0.0015
    )

    _add("P3B_F4_G5_SmoothScale", "Pillar3B_MethodG_Overlay", W_f4_sm)
    _add("P3B_F4_G5_Sm_NTB4bp", "Pillar3B_MethodG_Overlay", W_f4_ntb4)
    _add("P3B_F4_G5_Sm_NTB8bp", "Pillar3B_MethodG_Overlay", W_f4_ntb8)
    _add("P3B_F4_G5_Sm_NTB12bp", "Pillar3B_MethodG_Overlay", W_f4_ntb12)
    _add("P3B_F4_ADMM_Proj_L1_10", "Pillar3B_MethodG_Overlay", W_f4_proj10)
    _add("P3B_F4_ADMM_Proj_L1_15", "Pillar3B_MethodG_Overlay", W_f4_proj15)
    print("[Pillar 3B] Evaluated 6 turnover-aware Method G overlays on F4")

    # ------------------------------------------------------------------
    # Pillar 3C: Cost-Aware Portfolio-of-Strategies Ensembles (Optimized on Net-5bps!)
    # ------------------------------------------------------------------
    w_pool = {k: v["weights"] for k, v in entries.items()}
    r5_pool = pd.DataFrame(
        {k: v["bt5"]["net"] for k, v in entries.items()}, index=bundle.dates
    )

    # 1. Causal Expanding ERC on 6 individual-model ADMM sleeves (Net-5bps)
    ind6 = [
        "P3A_SLV_r5_L1_3bp",
        "P3A_SLV_r5_L1_5bp",
        "P3A_SLV_r1EMA3_L1_4bp",
        "P3A_SLV_rawEMA3_L1_4bp",
        "P3A_SLV_r1_L1_5bp",
        "P3A_SLV_raw_L1_5bp",
    ]
    sch_erc_ind6 = _causal_wf_schedule(
        r5_pool[ind6],
        bundle.dates,
        lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.35),
        ema_alpha=0.40,
        hyst_tau=0.02,
    )
    _add(
        "P3C_ENS_ExpERC_Net5_ADMM6",
        "Pillar3C_CostAware_Ensemble",
        consolidate_positions(sch_erc_ind6, w_pool),
        sch_erc_ind6,
    )

    # 2. Causal Expanding Regularized Mean-Variance on 8 ADMM sleeves (Net-5bps)
    admm8 = ind6 + ["P3A_ADMM_Kal3_L1_4bp", "P3A_ADMM_All3Sm_L1_3bp"]
    sch_reg8 = _causal_wf_schedule(
        r5_pool[admm8],
        bundle.dates,
        lambda X, wp, t: solve_regularized_return_risk(
            bayes_stein_mu(X, ledoit_wolf_cov(X)), ledoit_wolf_cov(X), wp, w_max=0.35
        ),
        ema_alpha=0.35,
        hyst_tau=0.02,
    )
    _add(
        "P3C_ENS_ExpRegMV_Net5_ADMM8",
        "Pillar3C_CostAware_Ensemble",
        consolidate_positions(sch_reg8, w_pool),
        sch_reg8,
    )

    # 3. Causal Dynamic Rolling 126d MinVar + Hysteresis on 8 ADMM sleeves (Net-5bps)
    sch_dyn8 = _causal_wf_schedule(
        r5_pool[admm8],
        bundle.dates,
        lambda X, wp, t: solve_min_var(ledoit_wolf_cov(X), w_max=0.35, l2_reg=0.15),
        roll_window=126,
        ema_alpha=0.30,
        hyst_tau=0.02,
    )
    _add(
        "P3C_ENS_DynRollMinVar_Net5_ADMM8",
        "Pillar3C_CostAware_Ensemble",
        consolidate_positions(sch_dyn8, w_pool),
        sch_dyn8,
    )

    # 4. Ultra-Low-Turnover (<10%/day) Causal ERC Ensemble (Net-5bps)
    sub10_cols = [
        "P3A_SLV_r5_L1_5bp",
        "P3A_ADMM_Kal3_L1_4bp",
        "P3A_ADMM_R5R1_L1_4bp",
        "P3B_F4_ADMM_Proj_L1_15",
    ]
    sch_sub10 = _causal_wf_schedule(
        r5_pool[sub10_cols],
        bundle.dates,
        lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.45),
        ema_alpha=0.40,
        hyst_tau=0.02,
    )
    _add(
        "P3C_ENS_UltraLowTurn_Sub10",
        "Pillar3C_CostAware_Ensemble",
        consolidate_positions(sch_sub10, w_pool),
        sch_sub10,
    )

    # 5. FLAGSHIP: Cross-Paradigm Causal Expanding ERC on Net-5bps (ADMM Factor-Neutral + Cost-Resilient F4)
    cross_cols = [
        "P3A_ADMM_Kal3_L1_4bp",
        "P3A_ADMM_All3Sm_L1_3bp",
        "P3A_SLV_r5_L1_5bp",
        "P3A_ADMM_EW3_L1_4bp",
        "P3B_F4_G5_Sm_NTB8bp",
    ]
    sch_cross_erc = _causal_wf_schedule(
        r5_pool[cross_cols],
        bundle.dates,
        lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.40),
        ema_alpha=0.40,
        hyst_tau=0.02,
    )
    W_cross_erc = consolidate_positions(sch_cross_erc, w_pool)
    _add(
        "P3C_ENS_Cross_ERC_ADMM_F4",
        "Pillar3C_CostAware_Ensemble",
        W_cross_erc,
        sch_cross_erc,
    )

    # 6. Two-Level Hierarchical Dual-Paradigm ERC (Level 2 inside ADMM & inside F4; Level 1 ERC across paradigms)
    l2_admm_cols = [
        "P3A_ADMM_Kal3_L1_4bp",
        "P3A_ADMM_All3Sm_L1_3bp",
        "P3A_SLV_r5_L1_5bp",
        "P3A_ADMM_EW3_L1_4bp",
    ]
    l2_f4_cols = ["P3B_F4_G5_Sm_NTB8bp", "P3B_F4_ADMM_Proj_L1_15"]
    sch_l2_admm = _causal_wf_schedule(
        r5_pool[l2_admm_cols],
        bundle.dates,
        lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.45),
        ema_alpha=0.40,
        hyst_tau=0.02,
    )
    sch_l2_f4 = _causal_wf_schedule(
        r5_pool[l2_f4_cols],
        bundle.dates,
        lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.65),
        ema_alpha=0.40,
        hyst_tau=0.02,
    )
    grp_r5 = pd.DataFrame(
        {
            "ADMM_Core": (
                r5_pool[l2_admm_cols].to_numpy() * sch_l2_admm.to_numpy()
            ).sum(axis=1),
            "F4_Hier_CostRes": (
                r5_pool[l2_f4_cols].to_numpy() * sch_l2_f4.to_numpy()
            ).sum(axis=1),
        },
        index=bundle.dates,
    )
    sch_l1 = _causal_wf_schedule(
        grp_r5,
        bundle.dates,
        lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.65),
        ema_alpha=0.40,
        hyst_tau=0.02,
    )
    hier_cols = l2_admm_cols + l2_f4_cols
    sch_hier = pd.DataFrame(0.0, index=bundle.dates, columns=hier_cols)
    sch_hier[l2_admm_cols] = (
        sch_l1["ADMM_Core"].to_numpy()[:, None] * sch_l2_admm.to_numpy()
    )
    sch_hier[l2_f4_cols] = (
        sch_l1["F4_Hier_CostRes"].to_numpy()[:, None] * sch_l2_f4.to_numpy()
    )
    W_hier_dp = consolidate_positions(sch_hier, w_pool)
    _add(
        "P3C_ENS_Cross_Hier_DualParadigm",
        "Pillar3C_CostAware_Ensemble",
        W_hier_dp,
        sch_hier,
    )

    # 7. Higher-CAGR Return-Oriented Cost-Resilient Portfolio (1.50x gross-scaled Dual-Paradigm ERC, <22%/d, DD <9%)
    W_ret_orient = apply_portfolio_constraints(
        1.50 * W_hier_dp,
        max_single_w=0.030,
        max_gross=2.40,
        adv20_np=adv_np,
        r1d_np=r1_np,
        max_adv_frac=0.05,
    )
    _add(
        "P3C_ENS_ReturnOriented_150x",
        "Pillar3C_CostAware_Ensemble",
        W_ret_orient,
        sch_hier,
    )
    print(
        f"[Pillar 3C] Evaluated 7 cost-aware ensembles in {time.time() - t_start:.1f}s total"
    )

    # Structured Ablation Comparison Order
    ablation_specs = [
        ("01_v27_Base_F4_Hier_Raw", "v27_F4_Hier_Dual1d5d", "Overlay_Ladder"),
        ("02_v27_Base_G_HierBest_G5", "v27_G_HierBest_G5", "Overlay_Ladder"),
        ("03_+SmoothScale_Hysteresis", "P3B_F4_G5_SmoothScale", "Overlay_Ladder"),
        ("04_+PostNTB_8bp", "P3B_F4_G5_Sm_NTB8bp", "Overlay_Ladder"),
        (
            "05_+CrossParadigm_Hier_ERC",
            "P3C_ENS_Cross_Hier_DualParadigm",
            "Overlay_Ladder",
        ),
        ("06_ADMM_BetaOnly_L1_3bp", "P3A_Abl_BetaOnly_L1_3bp", "ADMM_Factors"),
        ("07_ADMM_+SectorNeutral", "P3A_Abl_BetaSec_L1_3bp", "ADMM_Factors"),
        ("08_ADMM_+Sector+VolNeutral", "P3A_ADMM_All3Sm_L1_3bp", "ADMM_Factors"),
        (
            "09_ADMM_+Sector+Vol+MomNeutral",
            "P3A_Abl_BetaSecVolMom_L1_3bp",
            "ADMM_Factors",
        ),
        (
            "10_ADMM_NoDeadzone_dz0_L1_4bp",
            "P3A_Abl_NoDeadzone_dz0_L1_4bp",
            "ADMM_Design",
        ),
        ("11_ADMM_TailDeadzone_dz08_L1_4bp", "P3A_ADMM_All3Sm_L1_4bp", "ADMM_Design"),
        ("12_ADMM_InsideVolTgt8_L1_3bp", "P3A_Abl_InsideVolTgt8_L1_3bp", "ADMM_Design"),
        ("13_L1_Sweep_0bp", "P3A_ADMM_All3Sm_L1_0bp", "L1_Pareto"),
        ("14_L1_Sweep_2bp", "P3A_ADMM_All3Sm_L1_2bp", "L1_Pareto"),
        ("15_L1_Sweep_3bp", "P3A_ADMM_All3Sm_L1_3bp", "L1_Pareto"),
        ("16_L1_Sweep_4bp", "P3A_ADMM_All3Sm_L1_4bp", "L1_Pareto"),
        ("17_L1_Sweep_5bp", "P3A_ADMM_All3Sm_L1_5bp", "L1_Pareto"),
        ("18_L1_Sweep_8bp", "P3A_ADMM_All3Sm_L1_8bp", "L1_Pareto"),
    ]
    return entries, sig_decay_df, ablation_specs
