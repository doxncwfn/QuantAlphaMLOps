"""Date-batch generator yielding lookback feature windows and valid target masks."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch

from v27.targets.target_builder import PreparedAlphaDataset


def iter_batches(
    ds: PreparedAlphaDataset,
    indices: Sequence[int] | np.ndarray,
    target_key: str = "raw_1d",
    eval_key: str = "raw_1d",
    batch_size: int = 16,
    lookback: int = 60,
    max_missing: int = 2,
    shuffle: bool = False,
    sub_k: int | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield chronological or shuffled date mini-batches with causal lookback windows."""
    idx = np.array(indices, copy=True)
    if shuffle:
        np.random.shuffle(idx)
    dev = ds.Xn_g.device
    yt_g = ds.Y_map[target_key]
    ye_g = ds.Y_map[eval_key]
    y1_g = ds.Y_map["raw_1d"]
    for i in range(0, len(idx), batch_size):
        t_t = torch.as_tensor(idx[i : i + batch_size], dtype=torch.long, device=dev)
        cols = ds.slots_g[t_t]
        if sub_k is not None and sub_k < cols.shape[1]:
            cols = cols[:, torch.randperm(cols.shape[1], device=dev)[:sub_k]]
        safe = cols.clamp(min=0)
        real = cols >= 0
        miss = (
            ds.bad_cs_g[t_t[:, None], safe]
            - ds.bad_cs_g[(t_t - lookback)[:, None], safe]
        )
        mask = real & (miss <= max_missing)
        t_win = t_t[:, None, None] + ds.look_off[None, None, :]
        xw = (
            ds.Xn_g[t_win, safe[:, :, None]]
            .float()
            .masked_fill(~mask[:, :, None, None], 0.0)
        )
        yt = torch.where(real, yt_g[t_t[:, None], safe], float("nan"))
        ye = torch.where(real, ye_g[t_t[:, None], safe], float("nan"))
        y1 = torch.where(real, y1_g[t_t[:, None], safe], float("nan"))
        yield {
            "x": xw,
            "mask": mask,
            "y_train": torch.nan_to_num(yt, nan=0.0),
            "y_train_mask": mask & ~torch.isnan(yt),
            "y_eval": torch.nan_to_num(ye, nan=0.0),
            "y_eval_mask": mask & ~torch.isnan(ye),
            "y_1d": torch.nan_to_num(y1, nan=0.0),
            "y_1d_mask": mask & ~torch.isnan(y1),
            "mkt": ds.M_g[t_t],
            "date_index": t_t,
            "cols": cols,
        }
