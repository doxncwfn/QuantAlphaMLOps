"""Unit tests for src/training/trainer.py."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.training.trainer import train_one_fold
from src.training.wfcv import FoldSplit


class DummyStore:
    """Minimal synthetic FeatureDataStore for fast unit tests."""

    def __init__(self, n_dates: int = 80, n_stocks: int = 20, seq_len: int = 60):
        self.n_max_stocks = 30
        self.seq_len = seq_len
        self.unique_dates = np.array(
            [
                np.datetime64("2020-01-01") + np.timedelta64(i, "D")
                for i in range(n_dates)
            ],
            dtype="datetime64[ns]",
        )
        self.unique_permnos = np.arange(1000, 1000 + n_stocks, dtype=np.int64)
        self.date_to_idx = {d: i for i, d in enumerate(self.unique_dates)}
        self.permno_to_idx = {p: i for i, p in enumerate(self.unique_permnos)}

        rng = np.random.default_rng(42)
        self.grid = rng.standard_normal((n_stocks, n_dates, 19)).astype(np.float32)
        self.context_norm = rng.standard_normal((n_dates, 11)).astype(np.float32)

        self.date_permno_indices = {}
        self.date_permnos = {}
        self.date_targets = {}
        self.date_ystar = {}
        self.date_ynorm = {}
        self.date_masks = {}

        for d_i in range(n_dates):
            self.date_permno_indices[d_i] = np.arange(n_stocks, dtype=np.int32)
            self.date_permnos[d_i] = self.unique_permnos.copy()
            t5 = np.linspace(-0.05, 0.05, n_stocks, dtype=np.float32)
            ys = np.linspace(-0.5, 0.5, n_stocks, dtype=np.float32)
            yn = np.linspace(-2.0, 2.0, n_stocks, dtype=np.float32)
            self.date_targets[d_i] = t5
            self.date_ystar[d_i] = ys
            self.date_ynorm[d_i] = yn
            self.date_masks[d_i] = np.ones(n_stocks, dtype=bool)

        self.eval_date_indices = np.arange(seq_len - 1, n_dates, dtype=np.int32)


def make_dummy_fold(store: DummyStore) -> FoldSplit:
    eval_dates = store.unique_dates[store.eval_date_indices]
    train_d = eval_dates[:4]
    val_d = eval_dates[4:8]
    test_d = eval_dates[8:18]
    return FoldSplit(
        fold_idx=1,
        train_dates=train_d,
        val_dates=val_d,
        test_dates=test_d,
    )


def test_d3_early_stopping_and_checkpoint_gate(tmp_path) -> None:
    """Fix 1.1 (D3): Synthetic 20-epoch mock loop with val IC peaking at Epoch 5;
    assert best_epoch == 5 and best_weights is not None, and min_checkpoint_epoch > warmup_epochs raises ValueError.
    """
    store = DummyStore()
    fold = make_dummy_fold(store)

    # 1. Config validation: min_checkpoint_epoch > warmup_epochs must raise ValueError
    with pytest.raises(ValueError, match="min_checkpoint_epoch"):
        train_one_fold(
            fold=fold,
            store=store,
            device=torch.device("cpu"),
            max_epochs=2,
            warmup_epochs=5,
            min_checkpoint_epoch=10,
            checkpoint_dir=None,
            verbose=False,
        )

    # 2. Synthetic 20-epoch mock loop with val IC peaking at Epoch 5
    # Epoch 1..20 val ICs: peak at epoch 5 (+0.08), lower on all other epochs
    synthetic_val_ics = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.08,  # Epochs 1..5 (peak at 5)
        0.03,
        0.02,
        0.01,
        0.00,
        -0.01,  # Epochs 6..10
        -0.01,
        -0.01,
        -0.01,
        -0.01,
        -0.01,  # Epochs 11..15
        -0.01,
        -0.01,
        -0.01,
        -0.01,
        -0.01,  # Epochs 16..20
    ]
    call_idx = {"i": 0}

    def mock_eval_rank_ic(model, dataloader, device):
        idx = min(call_idx["i"], len(synthetic_val_ics) - 1)
        call_idx["i"] += 1
        return synthetic_val_ics[idx]

    ckpt_dir = tmp_path / "ckpts"
    with patch("src.training.trainer.evaluate_rank_ic", side_effect=mock_eval_rank_ic):
        res = train_one_fold(
            fold=fold,
            store=store,
            device=torch.device("cpu"),
            batch_size=4,
            grad_accum_steps=1,
            max_epochs=20,
            patience=10,
            warmup_epochs=15,
            val_ma_window=1,
            checkpoint_dir=ckpt_dir,
            verbose=False,
        )

    assert res["best_epoch"] == 5
    assert np.isclose(res["best_val_ic"], 0.08)
    ckpt_file = ckpt_dir / "fold_1_best.pt"
    assert ckpt_file.exists()
    ckpt_data = torch.load(ckpt_file, map_location="cpu")
    assert ckpt_data["model_state_dict"] is not None
    assert ckpt_data["best_epoch"] == 5


def test_d6_bfloat16_loss_autocast() -> None:
    """Fix 1.2 (D6): Feed bf16 inputs through autocast block and assert loss is float32 and finite;
    assert uncast bf16 inputs to decile_mono_loss and CombinedRankingLoss raise AssertionError.
    """
    from src.losses.combined import CombinedRankingLoss
    from src.losses.decile_mono import DecileMonotonicityLoss, decile_mono_loss
    from src.training.trainer import center_scores_per_date

    B, N = 2, 20
    scores_bf16 = torch.randn(B, N, dtype=torch.bfloat16)
    t5_bf16 = torch.randn(B, N, dtype=torch.bfloat16) * 0.02
    ys_bf16 = (
        torch.linspace(-1.5, 1.5, N, dtype=torch.bfloat16).unsqueeze(0).repeat(B, 1)
    )
    mask_bool = torch.ones(B, N, dtype=torch.bool)

    comb_fn = CombinedRankingLoss(lambda_hub=0.1)
    dm_fn = DecileMonotonicityLoss(lam_hub=0.2)

    # 1. Direct bf16 inputs without float32 cast must fail the dtype assertion
    with pytest.raises(AssertionError):
        decile_mono_loss(scores_bf16, ys_bf16, mask_bool)
    with pytest.raises(AssertionError):
        comb_fn(scores_bf16, t5_bf16, ys_bf16, mask_bool)

    # 2. Feeding bf16 inputs through the trainer's autocast(enabled=False) + .float() block
    with (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True),
        torch.autocast(device_type="cpu", enabled=False),
    ):
        scores_f = scores_bf16.float()
        t5_f = t5_bf16.float()
        ys_f = ys_bf16.float()
        mask_f = mask_bool.float()
        scores_c = center_scores_per_date(scores_f, mask_f)
        loss_comb, _l_mle, _l_tail = comb_fn(scores_c, t5_f, ys_f, mask_f)
        loss_dm, _l_mono, _l_ls = dm_fn(scores_c, t5_f, ys_f, mask_f)

    assert loss_comb.dtype == torch.float32
    assert torch.isfinite(loss_comb)
    assert loss_dm.dtype == torch.float32
    assert torch.isfinite(loss_dm)


def test_d5_clip_grad_nan_poisoning_guard(tmp_path) -> None:
    """Fix 1.6 (D5): Inject one NaN gradient; assert model parameters are bit-identical
    before and after the (skipped) step.
    """
    store = DummyStore()
    fold = make_dummy_fold(store)
    orig_clip = torch.nn.utils.clip_grad_norm_
    snapshots: dict[str, list[torch.Tensor]] = {}
    ckpt_dir_test = tmp_path / "test_d5_ckpt"

    def inject_nan_all_steps(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        if "before" not in snapshots:
            snapshots["before"] = [p.detach().clone() for p in params]
        if params and params[0].grad is not None:
            params[0].grad.fill_(float("nan"))
        return orig_clip(params, max_norm, *args, **kwargs)

    with patch("torch.nn.utils.clip_grad_norm_", side_effect=inject_nan_all_steps):
        train_one_fold(
            fold=fold,
            store=store,
            device=torch.device("cpu"),
            batch_size=4,
            grad_accum_steps=1,
            max_epochs=1,
            patience=10,
            warmup_epochs=1,
            checkpoint_dir=ckpt_dir_test,
            checkpoint_filename="nan_test.pt",
            verbose=False,
        )

    ckpt = torch.load(ckpt_dir_test / "nan_test.pt", map_location="cpu")
    after_tensors = list(ckpt["model_state_dict"].values())
    assert len(snapshots["before"]) == len(after_tensors)
    for b_param, a_param in zip(snapshots["before"], after_tensors):
        assert torch.equal(b_param, a_param), (
            "Model parameter modified despite NaN gradient!"
        )
        assert torch.isfinite(a_param).all()


def test_e1_e2_d4_self_describing_artifacts_and_diag_keys(tmp_path) -> None:
    """Fix 1.8 (E1 / E2 / D4): Run 1-epoch dry run; assert metadata.json exists and
    contains all required fields; assert baseline diag keys are correct.
    """
    import json

    store = DummyStore()
    fold = make_dummy_fold(store)
    ckpt_dir = tmp_path / "test_e1_ckpt"

    res_base = train_one_fold(
        fold=fold,
        store=store,
        device=torch.device("cpu"),
        batch_size=4,
        grad_accum_steps=1,
        max_epochs=1,
        patience=10,
        warmup_epochs=1,
        checkpoint_dir=ckpt_dir,
        checkpoint_filename="base_dry_run.pt",
        loss_type="combined",
        lambda_hub=0.1,
        verbose=False,
    )

    meta_file = ckpt_dir / "metadata.json"
    assert meta_file.exists(), "metadata.json not created alongside checkpoint"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_fields = {
        "loss_type",
        "lam_hub",
        "device",
        "dtype",
        "lr",
        "weight_decay",
        "warmup_epochs",
        "min_checkpoint_epoch",
        "seed",
        "batch_size",
        "git_sha",
        "target_hash",
    }
    assert required_fields.issubset(meta.keys())
    assert meta["loss_type"] == "combined"
    assert meta["lam_hub"] == 0.1
    assert len(meta["git_sha"]) >= 7
    assert len(meta["target_hash"]) == 64

    # target_hash must depend on target_5d array values, not only date_targets keys
    store_mod = DummyStore()
    first_key = next(iter(store_mod.date_targets))
    store_mod.date_targets[first_key] = store_mod.date_targets[first_key] + 1.0
    ckpt_dir_mod = tmp_path / "test_e1_ckpt_mod"
    train_one_fold(
        fold=fold,
        store=store_mod,
        device=torch.device("cpu"),
        batch_size=4,
        grad_accum_steps=1,
        max_epochs=1,
        patience=10,
        warmup_epochs=1,
        checkpoint_dir=ckpt_dir_mod,
        checkpoint_filename="base_dry_run_mod.pt",
        loss_type="combined",
        verbose=False,
    )
    with open(ckpt_dir_mod / "metadata.json", "r", encoding="utf-8") as f_mod:
        meta_mod = json.load(f_mod)
    assert meta_mod["target_hash"] != meta["target_hash"]

    hist_entry = res_base["history"][0]
    assert "L_mle" in hist_entry and "L_tail" in hist_entry
    assert "L_mono" not in hist_entry and "L_ls" not in hist_entry
    assert "train_l_mono" not in hist_entry and "train_l_ls" not in hist_entry


def test_d8_turnover_and_cost() -> None:
    """Fix 2.2 (D8):
    - Full replacement (all names swap): half_turn == 2.0, cost == 10 bps / 1e4 (mean over 2 rebalance dates)
      and per-step cost_t = half_turn_t * 2 * c_bps / 1e4 == 20 bps / 1e4.
    - No trade: half_turn == 0, cost == 0.
    """
    from src.training.evaluate import compute_turnover_and_cost, evaluate_predictions

    # 1. Direct helper check for single transition
    w0 = {1: 0.5, 2: 0.5, 3: -0.5, 4: -0.5}
    w1_swap = {5: 0.5, 6: 0.5, 7: -0.5, 8: -0.5}
    ht_swap, cost_swap = compute_turnover_and_cost(w1_swap, w0, c_bps=5.0)
    assert np.isclose(ht_swap, 2.0)
    assert np.isclose(cost_swap, 2.0 * 2.0 * 5.0 / 1e4)

    ht_same, cost_same = compute_turnover_and_cost(w0, w0, c_bps=5.0)
    assert ht_same == 0.0
    assert cost_same == 0.0

    # 2. End-to-end evaluate_predictions across 2 rebalance dates (T=6, stride=5 -> t=0, t=5)
    N = 20
    scores = np.zeros((6, N), dtype=np.float64)
    target_5d = np.zeros((6, N), dtype=np.float64)
    mask = np.ones((6, N), dtype=bool)
    permnos = np.tile(np.arange(100, 100 + N, dtype=np.int64), (6, 1))

    # Date 0: top decile (2 stocks) = [18, 19], bottom decile = [0, 1]
    scores[0] = np.linspace(-1.0, 1.0, N)
    # Date 5 (full replacement): top decile = [2, 3], bottom decile = [4, 5], all others middle
    scores[5] = np.zeros(N)
    scores[5, [2, 3]] = 10.0
    scores[5, [4, 5]] = -10.0

    res_swap = evaluate_predictions(
        scores=scores,
        target_5d=target_5d,
        mask=mask,
        permno_matrix=permnos,
        rebalance_stride=5,
        c_bps=5.0,
    )
    assert np.isclose(res_swap["half_turn"], 2.0)
    assert np.isclose(res_swap["turnover"], 2.0)
    assert np.isclose(res_swap["cost"], 10.0 / 1e4)
    assert np.isclose(
        res_swap["ls_spread_gross"] - res_swap["ls_spread_net"], 10.0 / 1e4
    )

    # No trade: Date 5 has identical top/bottom deciles as Date 0
    scores[5] = scores[0].copy()
    res_notrade = evaluate_predictions(
        scores=scores,
        target_5d=target_5d,
        mask=mask,
        permno_matrix=permnos,
        rebalance_stride=5,
        c_bps=5.0,
    )
    assert res_notrade["half_turn"] == 0.0
    assert res_notrade["turnover"] == 0.0
    assert res_notrade["cost"] == 0.0
    assert np.isclose(res_notrade["ls_spread_gross"], res_notrade["ls_spread_net"])


def test_a7_stride_horizon_alignment() -> None:
    """Fix 2.3 (A7):
    - Assert stride == horizon in default config (5 == 5).
    - Assert mismatched stride raises ValueError without the override flag.
    - Assert S10 is removed from default train_one_fold return path.
    """
    import inspect

    from src.training.evaluate import evaluate_predictions

    sig_eval = inspect.signature(evaluate_predictions)
    assert sig_eval.parameters["rebalance_stride"].default == 5
    assert sig_eval.parameters["horizon"].default == 5
    assert (
        sig_eval.parameters["rebalance_stride"].default
        == sig_eval.parameters["horizon"].default
    )

    sig_train = inspect.signature(train_one_fold)
    assert sig_train.parameters["rebalance_stride"].default == 5
    assert sig_train.parameters["horizon"].default == 5

    scores = np.random.default_rng(42).normal(size=(15, 20))
    target_5d = np.random.default_rng(43).normal(size=(15, 20))
    mask = np.ones((15, 20), dtype=bool)
    permnos = np.tile(np.arange(100, 120, dtype=np.int64), (15, 1))

    # Mismatched stride without allow_mismatched_stride=True must raise ValueError
    with pytest.raises(ValueError, match="rebalance_stride"):
        evaluate_predictions(
            scores=scores,
            target_5d=target_5d,
            mask=mask,
            permno_matrix=permnos,
            rebalance_stride=10,
            horizon=5,
        )

    # Mismatched stride with allow_mismatched_stride=True succeeds and sets report_suppressed=True
    res_override = evaluate_predictions(
        scores=scores,
        target_5d=target_5d,
        mask=mask,
        permno_matrix=permnos,
        rebalance_stride=10,
        horizon=5,
        allow_mismatched_stride=True,
    )
    assert res_override.get("report_suppressed") is True

    # Default train_one_fold does not return metrics_stride10
    store = DummyStore()
    fold = make_dummy_fold(store)
    res_fold = train_one_fold(
        fold=fold,
        store=store,
        device=torch.device("cpu"),
        batch_size=4,
        grad_accum_steps=1,
        max_epochs=1,
        patience=10,
        warmup_epochs=1,
        save_checkpoint=False,
        verbose=False,
    )
    assert "metrics_stride10" not in res_fold
