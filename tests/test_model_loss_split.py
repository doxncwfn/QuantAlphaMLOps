from __future__ import annotations

import numpy as np
import pytest
import torch

from qam import registry

Model = registry.get("model", "hybrid_lstm_tfm")


def batch(B=2, K=8, L=10, F=3, m_ts=2, m_static=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[:, -2:] = False
    return {
        "x": torch.randn(B, K, L, F, generator=g),
        "mask": mask,
        "x_mkt_ts": torch.randn(B, L, m_ts, generator=g),
        "x_mkt_static": torch.randn(B, m_static, generator=g),
    }


def model(**kw):
    torch.manual_seed(0)
    return Model(n_features=3, n_mkt_ts=2, n_mkt_static=1, hidden_dim=16, lstm_layers=1, tfm_layers=1, heads=2, ff_dim=16, dropout=0.0, **kw).eval()


def test_shape_and_masked_scores_are_zero():
    out = model()(batch())
    assert out.shape == (2, 8)
    assert torch.all(out[:, -2:] == 0)


def test_padding_content_does_not_change_real_scores():
    m, b = model(), batch()
    with torch.no_grad():
        before = m(b)
        b["x"][:, -2:] = 1e3
        after = m(b)
    torch.testing.assert_close(before[:, :-2], after[:, :-2])


def test_stock_permutation_equivariance():
    m, b = model(), batch()
    perm = torch.randperm(8)
    with torch.no_grad():
        base = m(b)
        shuffled = m({**b, "x": b["x"][:, perm], "mask": b["mask"][:, perm]})
    torch.testing.assert_close(shuffled, base[:, perm], rtol=1e-5, atol=1e-5)


def test_all_masked_date_gives_finite_output():
    b = batch(m_ts=0, m_static=0)
    b["mask"][0] = False
    m = Model(n_features=3, hidden_dim=16, lstm_layers=1, tfm_layers=1, heads=2, ff_dim=16, dropout=0.0).eval()
    assert torch.isfinite(m(b)).all()


@pytest.mark.parametrize("name", ["hybrid_hinge", "neg_pearson_rank"])
def test_losses_ignore_masked_slots_and_backprop(name):
    loss_fn = registry.get("loss", name)()
    scores = torch.randn(3, 20, requires_grad=True)
    labels = torch.randn(3, 20)
    mask = torch.ones(3, 20, dtype=torch.bool)
    mask[:, 15:] = False
    base = loss_fn(scores, labels, mask)
    noisy = labels.clone()
    noisy[:, 15:] = 1e6
    assert torch.isclose(base, loss_fn(scores, noisy, mask))
    base.backward()
    assert scores.grad[:, 15:].abs().sum() == 0 and scores.grad[:, :15].abs().sum() > 0


def test_neg_pearson_rank_is_minus_one_for_perfect_ranking():
    labels = torch.randn(1, 30)
    loss = registry.get("loss", "neg_pearson_rank")()(labels.argsort().argsort().float(), labels, torch.ones(1, 30, dtype=torch.bool))
    assert loss.item() == pytest.approx(-1.0, abs=1e-6)


def test_splitter_segments_are_ordered_and_gapped():
    dates = np.arange(np.datetime64("2010-01-01"), np.datetime64("2014-01-01"))
    folds = registry.get("splitter", "purged_walk_forward")(n_folds=3, purge=5, embargo=10, val_frac=0.2).split(dates)
    for fold in folds:
        pos = lambda d: int(np.searchsorted(dates, d))
        assert pos(fold.val[0]) - pos(fold.train[-1]) - 1 == 5
        assert pos(fold.test[0]) - pos(fold.val[-1]) - 1 == 15
    assert folds[-1].test[-1] == dates[-1]
