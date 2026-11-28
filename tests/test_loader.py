"""Tests for data loader: tensor assembly, Blom normalization, padding, and masks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.loader import FeatureDataStore, Russell1000Dataset, create_dataloader


@pytest.fixture(scope="module")
def data_store() -> FeatureDataStore:
    return FeatureDataStore(
        warehouse_path="data/warehouse/russell1000.duckdb",
        table_name="silver.features_daily",
        n_max_stocks=1200,
        seq_len=60,
    )


def test_data_store_dimensions(data_store: FeatureDataStore) -> None:
    """Verify data store tensor shapes and index structures."""
    assert len(data_store.unique_dates) >= 6000
    assert len(data_store.unique_permnos) >= 2700
    assert data_store.grid.shape == (
        len(data_store.unique_permnos),
        len(data_store.unique_dates),
        19,
    )
    assert data_store.context_norm.shape == (len(data_store.unique_dates), 11)
    assert len(data_store.eval_date_indices) >= 6000


def test_dataset_item_shapes_and_padding(data_store: FeatureDataStore) -> None:
    """Verify dataset item dictionary shapes, padding to N_max=1200, and types."""
    sample_dates = data_store.unique_dates[data_store.eval_date_indices[:20]]
    ds = Russell1000Dataset(data_store, sample_dates)
    assert len(ds) == 20

    item = ds[0]
    assert item["x_stock"].shape == (1200, 60, 19)
    assert item["x_stock"].dtype == torch.float32
    assert item["x_ctx"].shape == (60, 11)
    assert item["x_ctx"].dtype == torch.float32
    assert item["target_5d"].shape == (1200,)
    assert item["y_star"].shape == (1200,)
    assert item["mask"].shape == (1200,)
    assert item["mask"].dtype == torch.bool
    assert item["permno"].shape == (1200,)
    assert item["permno"].dtype == torch.int64

    # Real in-universe stocks are populated; padded slots beyond n_stocks are zero
    n_stocks = len(data_store.date_permno_indices[int(data_store.eval_date_indices[0])])
    n_valid = int(item["mask"].sum())
    assert 500 <= n_valid <= n_stocks <= 1200
    assert (item["x_stock"][n_stocks:] == 0.0).all()
    assert (item["mask"][n_stocks:] == False).all()
    assert (item["permno"][n_stocks:] == 0).all()


def test_dataloader_batching_and_mask_consistency(data_store: FeatureDataStore) -> None:
    """Verify DataLoader collation and target validity mask consistency."""
    sample_dates = data_store.unique_dates[data_store.eval_date_indices[:64]]
    ds = Russell1000Dataset(data_store, sample_dates)
    dl = create_dataloader(ds, batch_size=32, shuffle=False)

    batch = next(iter(dl))
    assert batch["x_stock"].shape == (32, 1200, 60, 19)
    assert batch["x_ctx"].shape == (32, 60, 11)
    assert batch["target_5d"].shape == (32, 1200)
    assert batch["y_star"].shape == (32, 1200)
    assert batch["mask"].shape == (32, 1200)

    # Valid mask slots must have finite target_5d and y_star in [-0.5, 0.5]
    valid_ys = batch["y_star"][batch["mask"]]
    assert torch.all(valid_ys >= -0.5001)
    assert torch.all(valid_ys <= +0.5001)
    assert torch.isfinite(batch["target_5d"][batch["mask"]]).all()


def test_blom_transform_distribution_properties(data_store: FeatureDataStore) -> None:
    """Verify that Blom scores for in-universe stocks have mean near 0 and std near 1."""
    d_i = int(data_store.eval_date_indices[1000])
    p_indices = data_store.date_permno_indices[d_i]
    # Stock features on date d_i
    feat_slice = data_store.grid[p_indices, d_i, :]  # [N_stocks, 19]

    # For features with valid observations, Blom normal scores should be standardized
    for f in range(19):
        col = feat_slice[:, f]
        non_zero = col[col != 0.0]
        if len(non_zero) >= 500:
            assert abs(float(np.mean(non_zero))) < 0.1
            assert abs(float(np.std(non_zero)) - 1.0) < 0.15


def test_a5_ynorm_std_and_monotonicity(data_store: FeatureDataStore) -> None:
    """Fix 2.1 (A5):
    - y_norm has per-date std in [0.95, 1.05] over all evaluation dates.
    - y_norm is a strictly monotone transform of target_5d within each date.
    """
    assert hasattr(data_store, "date_ynorm")
    stds = []
    for d_i in data_store.eval_date_indices:
        d_int = int(d_i)
        mask = data_store.date_masks[d_int]
        n_valid = int(mask.sum())
        assert n_valid >= 500
        yn_v = data_store.date_ynorm[d_int][mask]
        t5_v = data_store.date_targets[d_int][mask]

        std_val = float(np.std(yn_v, ddof=1))
        assert 0.95 <= std_val <= 1.05, (
            f"Date {d_int}: std(y_norm[mask]) = {std_val:.4f} outside [0.95, 1.05]"
        )
        stds.append(std_val)

        # Check strict monotonicity wrt target_5d: when t5 sorted ascending,
        # strict increases in t5 must correspond to strict increases in y_norm
        order = np.argsort(t5_v, kind="mergesort")
        t5_sorted = t5_v[order]
        yn_sorted = yn_v[order]
        strict_t5_inc = np.diff(t5_sorted) > 1e-12
        assert np.all(np.diff(yn_sorted)[strict_t5_inc] > 0.0), (
            f"Date {d_int}: y_norm not strictly monotone wrt target_5d"
        )

    assert len(stds) == len(data_store.eval_date_indices)
