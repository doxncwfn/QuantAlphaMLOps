"""24-candidate complementarity: correlation regimes, Jaccard overlap, cosine similarity & tail dependence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from v27.config import CANDIDATES_24
from v27.data.bundle import V7DataBundle


def compute_return_correlations(
    net_df: pd.DataFrame,
    ew_mkt: pd.Series,
    tab_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute Pearson, Spearman, downside, high-vol, and large-drawdown correlation matrices."""
    cids = list(net_df.columns)
    corr_p = net_df.corr(method="pearson")
    corr_s = net_df.corr(method="spearman")

    X = net_df.to_numpy(dtype=np.float64)
    n_c = len(cids)
    down_mat = np.eye(n_c)
    for i in range(n_c):
        for j in range(i + 1, n_c):
            m = (X[:, i] < 0.0) | (X[:, j] < 0.0)
            c = float(np.corrcoef(X[m, i], X[m, j])[0, 1]) if m.sum() > 20 else np.nan
            down_mat[i, j] = down_mat[j, i] = c
    corr_down = pd.DataFrame(down_mat, index=cids, columns=cids)

    ew_eq = net_df.mean(axis=1)
    ew_nav = (1.0 + ew_eq).cumprod()
    ew_dd = ew_nav / ew_nav.cummax().clip(lower=1.0) - 1.0
    large_dd_mask = ew_dd <= ew_dd.quantile(0.15)
    v20_mkt = ew_mkt.rolling(20, min_periods=5).std().shift(1).bfill()
    hi_vol_mask = v20_mkt >= v20_mkt.quantile(0.80)

    corr_dd = net_df.loc[large_dd_mask].corr(method="pearson")
    corr_hivol = net_df.loc[hi_vol_mask].corr(method="pearson")

    if tab_dir is not None:
        corr_p.to_csv(tab_dir / "sec4_corr_pearson_24.csv")
        corr_s.to_csv(tab_dir / "sec4_corr_spearman_24.csv")
        corr_down.to_csv(tab_dir / "sec4_corr_downside_24.csv")
        corr_dd.to_csv(tab_dir / "sec4_corr_large_dd_24.csv")
        corr_hivol.to_csv(tab_dir / "sec4_corr_highvol_24.csv")

    return {
        "pearson": corr_p,
        "spearman": corr_s,
        "downside": corr_down,
        "large_dd": corr_dd,
        "high_vol": corr_hivol,
    }


def compute_position_overlap_and_tail(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    net_df: pd.DataFrame,
    tab_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute portfolio weight cosine similarity, Jaccard overlap (long/short/all), and 5% lower-tail dependence."""
    cids = list(CANDIDATES_24)
    n_c = len(cids)
    t_d = len(bundle.dates)
    cos_sim = np.eye(n_c)
    ov_long = np.eye(n_c)
    ov_short = np.eye(n_c)
    ov_all = np.eye(n_c)

    sl5 = slice(0, t_d, 5)
    W5_list = [cand_weights[c][sl5] for c in cids]
    W5_norms = [np.linalg.norm(W, axis=1) for W in W5_list]
    L5_list = [W > 1e-8 for W in W5_list]
    S5_list = [W < -1e-8 for W in W5_list]
    A5_list = [np.abs(W) > 1e-8 for W in W5_list]

    for i in range(n_c):
        for j in range(i + 1, n_c):
            dots = np.einsum("ti,ti->t", W5_list[i], W5_list[j])
            den = np.maximum(W5_norms[i] * W5_norms[j], 1e-12)
            cos_sim[i, j] = cos_sim[j, i] = float(np.mean(dots / den))
            ov_long[i, j] = ov_long[j, i] = float(
                np.mean(
                    (L5_list[i] & L5_list[j]).sum(axis=1)
                    / np.maximum((L5_list[i] | L5_list[j]).sum(axis=1), 1)
                )
            )
            ov_short[i, j] = ov_short[j, i] = float(
                np.mean(
                    (S5_list[i] & S5_list[j]).sum(axis=1)
                    / np.maximum((S5_list[i] | S5_list[j]).sum(axis=1), 1)
                )
            )
            ov_all[i, j] = ov_all[j, i] = float(
                np.mean(
                    (A5_list[i] & A5_list[j]).sum(axis=1)
                    / np.maximum((A5_list[i] | A5_list[j]).sum(axis=1), 1)
                )
            )

    df_wcos = pd.DataFrame(cos_sim, index=cids, columns=cids)
    df_ov_all = pd.DataFrame(ov_all, index=cids, columns=cids)
    df_ov_l = pd.DataFrame(ov_long, index=cids, columns=cids)
    df_ov_s = pd.DataFrame(ov_short, index=cids, columns=cids)

    X = net_df.to_numpy(dtype=np.float64)
    qs = np.quantile(X, 0.05, axis=0)
    t_ind = (X <= qs[None, :]).astype(np.float64)
    tail_dep = (t_ind.T @ t_ind) / max(t_d * 0.05, 1e-12)
    np.fill_diagonal(tail_dep, 1.0)
    df_tail = pd.DataFrame(tail_dep, index=cids, columns=cids)

    if tab_dir is not None:
        df_wcos.to_csv(tab_dir / "sec4_weight_cosine_similarity_24.csv")
        df_ov_all.to_csv(tab_dir / "sec4_position_overlap_all_24.csv")
        df_ov_l.to_csv(tab_dir / "sec4_position_overlap_long_24.csv")
        df_ov_s.to_csv(tab_dir / "sec4_position_overlap_short_24.csv")
        df_tail.to_csv(tab_dir / "sec4_tail_dependence_24.csv")

    return {
        "wcos": df_wcos,
        "ov_all": df_ov_all,
        "ov_long": df_ov_l,
        "ov_short": df_ov_s,
        "tail_dep": df_tail,
    }


def build_pairwise_complementarity(
    corrs: dict[str, pd.DataFrame],
    overlaps: dict[str, pd.DataFrame],
    tab_dir: Path | None = None,
) -> pd.DataFrame:
    """Flatten all 276 upper-triangular candidate pairs into a ranked complementarity table."""
    corr_p = corrs["pearson"]
    cids = list(corr_p.index)
    rows = []
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            rows.append(
                {
                    "Cand_A": cids[i],
                    "Cand_B": cids[j],
                    "Ret_Pearson": float(corr_p.iloc[i, j]),
                    "Ret_Spearman": float(corrs["spearman"].iloc[i, j]),
                    "Downside_Corr": float(corrs["downside"].iloc[i, j]),
                    "HighVol_Corr": float(corrs["high_vol"].iloc[i, j]),
                    "LargeDD_Corr": float(corrs["large_dd"].iloc[i, j]),
                    "Weight_CosSim": float(overlaps["wcos"].iloc[i, j]),
                    "Hold_Jaccard": float(overlaps["ov_all"].iloc[i, j]),
                    "Long_Jaccard": float(overlaps["ov_long"].iloc[i, j]),
                    "Short_Jaccard": float(overlaps["ov_short"].iloc[i, j]),
                }
            )
    df_pairs = pd.DataFrame(rows).sort_values("Ret_Pearson")
    if tab_dir is not None:
        df_pairs.to_csv(tab_dir / "sec4_pairwise_complementarity_276.csv", index=False)
    return df_pairs
