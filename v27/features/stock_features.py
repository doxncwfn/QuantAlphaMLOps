"""34 GPU/vectorized cross-sectional stock feature engineering pipeline."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as TF


def gpu_rolling_stats(
    x: torch.Tensor, windows: tuple[int, ...]
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
    """Compute exact rolling means, sample stds, and finite counts along axis=0."""
    _, N_ = x.shape
    valid = torch.isfinite(x)
    z = torch.where(valid, x, 0.0).to(torch.float64)
    z0 = torch.zeros((1, N_), dtype=torch.float64, device=x.device)
    c0 = torch.zeros((1, N_), dtype=torch.int32, device=x.device)
    cs1 = torch.cat([z0, z.cumsum(dim=0)], dim=0)
    cs2 = torch.cat([z0, (z * z).cumsum(dim=0)], dim=0)
    cnt = torch.cat([c0, valid.to(torch.int32).cumsum(dim=0)], dim=0)
    means, stds = {}, {}
    for n in windows:
        pad = torch.full(
            (n - 1, N_), float("nan"), dtype=torch.float32, device=x.device
        )
        s1 = cs1[n:] - cs1[:-n]
        s2 = cs2[n:] - cs2[:-n]
        ok = (cnt[n:] - cnt[:-n]) == n
        m = torch.where(ok, s1 / n, float("nan")).to(torch.float32)
        var = ((s2 - (s1 * s1) / n) / max(1, n - 1)).clamp(min=0.0)
        sd = torch.where(ok, var.sqrt(), float("nan")).to(torch.float32)
        means[n] = torch.cat([pad, m], dim=0)
        stds[n] = torch.cat([pad, sd], dim=0)
    return means, stds, cnt


def gpu_rolling_extrema(
    x: torch.Tensor, cnt: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute rolling max and min over window n requiring all n observations finite."""
    _, N_ = x.shape
    pad = torch.full((n - 1, N_), float("nan"), dtype=torch.float32, device=x.device)
    ok = (cnt[n:] - cnt[:-n]) == n
    x_t = x.T.unsqueeze(1)
    rmax = (
        TF.max_pool1d(torch.nan_to_num(x_t, nan=-1e30), kernel_size=n, stride=1)
        .squeeze(1)
        .T
    )
    rmin = (
        (-TF.max_pool1d(torch.nan_to_num(-x_t, nan=-1e30), kernel_size=n, stride=1))
        .squeeze(1)
        .T
    )
    return (
        torch.cat([pad, torch.where(ok, rmax, float("nan"))], dim=0),
        torch.cat([pad, torch.where(ok, rmin, float("nan"))], dim=0),
    )


def gpu_rolling_corr_or_beta(
    x: torch.Tensor, y: torch.Tensor, n: int, mode: str = "corr"
) -> torch.Tensor:
    """Compute rolling correlation or OLS slope of x against y over window n."""
    _, N_ = x.shape
    valid = torch.isfinite(x) & torch.isfinite(y)
    xv = torch.where(valid, x, 0.0).to(torch.float64)
    yv = torch.where(valid, y, 0.0).to(torch.float64)
    z0 = torch.zeros((1, N_), dtype=torch.float64, device=x.device)
    c0 = torch.zeros((1, N_), dtype=torch.int32, device=x.device)
    cs_x = torch.cat([z0, xv.cumsum(dim=0)], dim=0)
    cs_y = torch.cat([z0, yv.cumsum(dim=0)], dim=0)
    cs_xx = torch.cat([z0, (xv * xv).cumsum(dim=0)], dim=0)
    cs_yy = torch.cat([z0, (yv * yv).cumsum(dim=0)], dim=0)
    cs_xy = torch.cat([z0, (xv * yv).cumsum(dim=0)], dim=0)
    cnt = torch.cat([c0, valid.to(torch.int32).cumsum(dim=0)], dim=0)
    ok = (cnt[n:] - cnt[:-n]) == n
    mx = (cs_x[n:] - cs_x[:-n]) / n
    my = (cs_y[n:] - cs_y[:-n]) / n
    cov = (cs_xy[n:] - cs_xy[:-n]) / n - mx * my
    vx = ((cs_xx[n:] - cs_xx[:-n]) / n - mx * mx).clamp(min=0.0)
    vy = ((cs_yy[n:] - cs_yy[:-n]) / n - my * my).clamp(min=0.0)
    val = (
        (cov / (vx.sqrt() * vy.sqrt() + 1e-12)).clamp(-1.0, 1.0)
        if mode == "corr"
        else (cov / (vy + 1e-12)).clamp(-5.0, 5.0)
    )
    pad = torch.full((n - 1, N_), float("nan"), dtype=torch.float32, device=x.device)
    return torch.cat([pad, torch.where(ok, val, float("nan")).to(torch.float32)], dim=0)


def gpu_rolling_skew(x: torch.Tensor, n: int) -> torch.Tensor:
    """Compute rolling sample skewness over window n."""
    _, N_ = x.shape
    valid = torch.isfinite(x)
    z = torch.where(valid, x, 0.0).to(torch.float64)
    z0 = torch.zeros((1, N_), dtype=torch.float64, device=x.device)
    c0 = torch.zeros((1, N_), dtype=torch.int32, device=x.device)
    cs1 = torch.cat([z0, z.cumsum(dim=0)], dim=0)
    cs2 = torch.cat([z0, (z * z).cumsum(dim=0)], dim=0)
    cs3 = torch.cat([z0, (z * z * z).cumsum(dim=0)], dim=0)
    cnt = torch.cat([c0, valid.to(torch.int32).cumsum(dim=0)], dim=0)
    ok = (cnt[n:] - cnt[:-n]) == n
    mu = (cs1[n:] - cs1[:-n]) / n
    e2 = (cs2[n:] - cs2[:-n]) / n
    e3 = (cs3[n:] - cs3[:-n]) / n
    var = (e2 - mu * mu).clamp(min=0.0)
    m3 = e3 - 3.0 * mu * e2 + 2.0 * (mu * mu * mu)
    sk = (m3 / (var.pow(1.5) + 1e-12)).clamp(-10.0, 10.0)
    pad = torch.full((n - 1, N_), float("nan"), dtype=torch.float32, device=x.device)
    return torch.cat([pad, torch.where(ok, sk, float("nan")).to(torch.float32)], dim=0)


@torch.no_grad()
def compute_34_stock_features(
    op: torch.Tensor,
    hi: torch.Tensor,
    lo: torch.Tensor,
    cl: torch.Tensor,
    vo: torch.Tensor,
    adj: torch.Tensor,
    rt: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Compute all 34 daily cross-sectional stock features and 1d forward returns Y."""
    _, N_ = adj.shape
    dev = adj.device
    nan_row = torch.full((1, N_), float("nan"), dtype=torch.float32, device=dev)

    log_ret_raw = torch.log1p(rt.clamp(min=-0.9999))
    log_ret = log_ret_raw.clamp(-0.2231, 0.1823)
    moms = {
        n: torch.cat(
            [
                torch.full((n, N_), float("nan"), dtype=torch.float32, device=dev),
                adj[n:] / adj[:-n] - 1.0,
            ],
            dim=0,
        )
        for n in (5, 10, 20, 60)
    }
    sma_means, _, adj_cnt = gpu_rolling_stats(adj, (5, 10, 20, 60))
    ret_means, vol_stds, ret_cnt = gpu_rolling_stats(log_ret_raw, (5, 20, 60))
    dist_smas = {n: adj / sma_means[n] - 1.0 for n in (5, 10, 20, 60)}
    rmax_20, rmin_20 = gpu_rolling_extrema(adj, adj_cnt, 20)
    dist_high_20 = adj / rmax_20 - 1.0
    dist_low_20 = adj / rmin_20 - 1.0
    hl_range = (hi - lo) / (cl + 1e-8)
    close_open = cl / (op + 1e-8) - 1.0
    prev_cl = torch.cat([nan_row, cl[:-1]], dim=0)
    gap = (op / (prev_cl + 1e-8) - 1.0).clamp(-0.5, 0.5)
    diff = torch.cat([nan_row, adj[1:] - adj[:-1]], dim=0)
    up_m, _, _ = gpu_rolling_stats(diff.clamp(min=0.0), (14,))
    dn_m, _, _ = gpu_rolling_stats((-diff).clamp(min=0.0), (14,))
    rsi_14 = 100.0 * up_m[14] / (up_m[14] + dn_m[14] + 1e-12)
    vo_means, _, _ = gpu_rolling_stats(vo, (20,))
    dv_raw = cl * vo
    dv_means, _, _ = gpu_rolling_stats(dv_raw, (20,))
    vol_ratio_20 = vo / (vo_means[20] + 1e-9)
    log_dollar_volume_20 = torch.log1p(dv_means[20].clamp(min=0.0))

    dv_safe = torch.where(torch.isfinite(dv_raw), dv_raw.clamp(min=1.0), float("nan"))
    amihud_m, _, _ = gpu_rolling_stats(log_ret_raw.abs() / (dv_safe / 1e6), (20,))
    amihud_20 = torch.log1p(amihud_m[20].clamp(min=0.0))
    log_vol = torch.where(
        torch.isfinite(vo), torch.log1p(vo.clamp(min=0.0)), float("nan")
    )
    corr_pv_20 = gpu_rolling_corr_or_beta(adj, log_vol, 20, mode="corr")
    clv = ((cl - lo) - (hi - cl)) / (hi - lo + 1e-8)
    clv_vo_m, _, _ = gpu_rolling_stats(clv * vo, (20,))
    cmf_20 = (clv_vo_m[20] / (vo_means[20] + 1e-8)).clamp(-1.0, 1.0)

    mom_120_20 = torch.cat(
        [
            torch.full((120, N_), float("nan"), dtype=torch.float32, device=dev),
            adj[100:-20] / adj[:-120] - 1.0,
        ],
        dim=0,
    )
    mom_252_20 = torch.cat(
        [
            torch.full((252, N_), float("nan"), dtype=torch.float32, device=dev),
            adj[232:-20] / adj[:-252] - 1.0,
        ],
        dim=0,
    )
    sharpe_60 = ret_means[60] / (vol_stds[60] + 1e-8)
    vol_ratio_5_60 = vol_stds[5] / (vol_stds[60] + 1e-8)

    r_mkt = torch.nanmean(log_ret_raw, dim=1, keepdim=True).expand_as(log_ret_raw)
    beta_60 = gpu_rolling_corr_or_beta(log_ret_raw, r_mkt, 60, mode="beta")
    _, idio_stds, _ = gpu_rolling_stats(log_ret_raw - beta_60 * r_mkt, (20,))
    idio_vol_20 = idio_stds[20]

    max_ret_20, _ = gpu_rolling_extrema(log_ret_raw, ret_cnt, 20)
    skew_20 = gpu_rolling_skew(log_ret, 20)
    hl_log_sq = torch.where(
        torch.isfinite(hi) & torch.isfinite(lo) & (lo > 0),
        torch.log((hi / lo.clamp(min=1e-6)).clamp(min=1.0)).pow(2),
        float("nan"),
    )
    pk_m, _, _ = gpu_rolling_stats(hl_log_sq, (20,))
    parkinson_vol_20 = (pk_m[20] / (4.0 * math.log(2.0))).clamp(min=0.0).sqrt()
    gap_m, _, _ = gpu_rolling_stats(gap, (20,))
    co_m, _, _ = gpu_rolling_stats(close_open, (20,))
    overnight_vs_intraday_20 = gap_m[20] - co_m[20]

    feat_dict = {
        "log_ret": log_ret,
        "mom_5": moms[5],
        "mom_10": moms[10],
        "mom_20": moms[20],
        "mom_60": moms[60],
        "vol_5": vol_stds[5],
        "vol_20": vol_stds[20],
        "vol_60": vol_stds[60],
        "dist_sma_5": dist_smas[5],
        "dist_sma_10": dist_smas[10],
        "dist_sma_20": dist_smas[20],
        "dist_sma_60": dist_smas[60],
        "dist_high_20": dist_high_20,
        "dist_low_20": dist_low_20,
        "hl_range": hl_range,
        "close_open": close_open,
        "gap": gap,
        "rsi_14": rsi_14,
        "vol_ratio_20": vol_ratio_20,
        "log_dollar_volume_20": log_dollar_volume_20,
        "amihud_20": amihud_20,
        "corr_pv_20": corr_pv_20,
        "clv": clv,
        "cmf_20": cmf_20,
        "mom_120_20": mom_120_20,
        "mom_252_20": mom_252_20,
        "sharpe_60": sharpe_60,
        "vol_ratio_5_60": vol_ratio_5_60,
        "beta_60": beta_60,
        "idio_vol_20": idio_vol_20,
        "max_ret_20": max_ret_20,
        "skew_20": skew_20,
        "parkinson_vol_20": parkinson_vol_20,
        "overnight_vs_intraday_20": overnight_vs_intraday_20,
    }
    X_gpu = torch.stack(list(feat_dict.values()), dim=-1)
    X_gpu = torch.where(torch.isinf(X_gpu), float("nan"), X_gpu)
    alive_prev = ~torch.isnan(prev_cl)
    Y_gpu = torch.where(alive_prev, rt, float("nan"))
    return feat_dict, X_gpu, Y_gpu
