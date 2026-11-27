# Loss Function Specifications — PiT Russell 1000 (2000–2026)

**Purpose of this document:** implementation-ready specifications for the top three candidate losses. Each spec is self-contained and can be handed to a separate implementation chat. Shared conventions are defined once in §0 and referenced by each spec.

---

## §0 — Shared conventions

**Tensors (per batch of B trading dates):**

| Symbol | Shape | Meaning |
|---|---|---|
| `y_hat` | `[B, N]` | model predictions (raw scores, unbounded) |
| `y` | `[B, N]` | realized forward returns over holding period (e.g., next-month return) |
| `mask` | `[B, N]` | bool, `True` = valid stock that date |
| `N` | scalar | ≈ 1,000 (Russell 1000 size varies 950–1,050) |
| `K` | scalar | 10 (deciles) |

**Target normalization (applied once, before any loss):**
Cross-sectionally **winsorize `y` at the 1st/99th percentile within each date**, then **rank-Gaussianize within each date** (van der Waerden: `y_norm = Φ⁻¹(rank(y)/(N+1))`). This makes all losses scale-consistent, robust to the heavy tails of 2000–2026, and consistent with the "rank-normalize within date" convention from §1. Use `y_norm` in every loss term below. Keep raw `y` only for reporting gross spread in economic units.

**Masking rule:** every sum over `i` is masked (`Σ_i → Σ_i mask_i`); every bucket mean divides by its own masked mass plus ε.

**ε = 1e-8** everywhere unless specified.

**Precision:** float32 training; float64 for any cumulative sums spanning > B=64 dates.

**Reproducibility:** log all hyperparameters, seed, and the version of `torchsort` (or equivalent soft-rank library) — numerical behavior differs subtly across versions.

---

## §1 — Loss Spec #1: Soft-Decile Monotonicity + Long-Short Spread

**Priority rank: 1 (run first).**

### 1.1 Objective
Enforce monotone ordering of the K soft-decile means of `y_norm` w.r.t. model prediction rank, with a minimum-margin separation between adjacent deciles, and jointly maximize the standardized top-minus-bottom spread. This is the primary candidate because at N≈1,000 each bucket contains ≈100 names, so bucket means are statistically stable, and the hinge-on-levels form is sensitive to spread magnitude (unlike a decile Spearman surrogate).

### 1.2 Formula

**Step A — soft rank** (choose one; torchsort is the recommended default):

- **A1 (torchsort, preferred):**
  ```
  r = soft_rank(y_hat, regularization=τ_sort, dim=1)     # r ∈ [B, N], r ∈ (1, N)
  ```
- **A2 (fallback, no external lib):**
  Approximate rank by softmax over a fixed grid `g = [1, 2, …, N]`:
  ```
  r_i = Σ_g g · softmax_g( -(g - s_i)² / (2 τ_sort²) )   with s_i = y_hat_i / std(y_hat) · (N/6)
  ```

**Step B — soft bucket assignment.** Define centroids `c_k = (k − 0.5) · N / K`, `k = 1..K` (for N=1000, K=10: c ∈ {50, 150, …, 950}). Assign:

```
p[i,k] = exp( -(r[i] - c_k)² / (2 τ_assign²) ) / Σ_{k'} exp( -(r[i] - c_k')² / (2 τ_assign²) )
```

Row-normalize so `Σ_k p[i,k] = 1`. Denote masked mass `M_k = Σ_i p[i,k] · mask_i`.

**Step C — soft bucket means:**

```
μ_k = ( Σ_i p[i,k] · mask_i · y_norm[i] ) / (M_k + ε)
```

**Step D — monotonicity hinge with margin:**

```
L_mono = ( 1 / (K − 1) ) · Σ_{k=1}^{K-1} ReLU( m_mono − (μ_{k+1} − μ_k) )²
```

Note: `m_mono ≥ 0`. Setting `m_mono = 0` recovers standard monotonicity (penalize only inversions). Setting `m_mono > 0` requires a minimum gap and prevents degenerate score compression.

**Step E — long-short spread (normalized):**

```
L_ls = − ( μ_K − μ_1 ) / ( σ_y + ε )
```
where `σ_y = std( y_norm[mask] )` within the date (masked std). This makes the spread interpretable in cross-sectional σ units and prevents the model from trading spread for raw magnitude.

**Step F — Huber auxiliary (required — prevents rank collapse to a single point):**

```
L_hub = mean( SmoothL1( y_hat, y_norm, β = 1.0 ) )
```

**Total loss:**

```
L_1 = L_mono + λ_ls · L_ls + λ_hub · L_hub
```

### 1.3 Hyperparameters & suggested ranges

| Name | Symbol | Range | Default | Notes |
|---|---|---|---|---|
| Sort temperature | τ_sort | 1e-3 – 1e-1 | 0.01 | Smaller = closer to exact rank, noisier gradients |
| Assignment temperature | τ_assign | 5 – 50 (rank units) | 0.3 · (N/K) | ~30 for N=1000, K=10 |
| Monotonicity margin | m_mono | 0 – 0.3 (σ units) | 0.05 | >0 discourages compression |
| LS spread weight | λ_ls | 0.1 – 1.0 | 0.3 | Increase if spread collapses |
| Huber weight | λ_hub | 0.1 – 0.5 | 0.2 | Keep > 0 to prevent rank collapse |
| Huber β | β | 1.0 | 1.0 | In `y_norm` σ units |

### 1.4 Pseudocode

```python
def loss_decile_mono(
    y_hat,
    y_norm,
    mask,
    K=10,
    tau_sort=0.01,
    tau_assign=None,
    m_mono=0.05,
    lam_ls=0.3,
    lam_hub=0.2,
    beta=1.0,
):
    B, N = y_hat.shape
    if tau_assign is None:
        tau_assign = 0.3 * (N / K)
    eps = 1e-8

    # Step A: soft rank (torchsort)
    r = soft_rank(y_hat, regularization=tau_sort)  # [B, N]

    # Step B: soft bucket assignment
    c = (torch.arange(1, K + 1, device=y_hat.device) - 0.5) * (N / K)  # [K]
    d = r.unsqueeze(-1) - c.unsqueeze(0).unsqueeze(0)  # [B, N, K]
    logits = -(d**2) / (2 * tau_assign**2)
    p = torch.softmax(logits, dim=-1) * mask.unsqueeze(-1)  # [B, N, K]

    # Step C: bucket means
    M = p.sum(dim=1)  # [B, K]
    num = (p * y_norm.unsqueeze(-1)).sum(dim=1)  # [B, K]
    mu = num / (M + eps)  # [B, K]

    # Step D: monotonicity hinge with margin
    diff = mu[:, 1:] - mu[:, :-1]  # [B, K-1]
    L_mono = (torch.relu(m_mono - diff) ** 2).mean()

    # Step E: LS spread (standardized)
    sigma_y = masked_std(y_norm, mask, dim=1)  # [B]
    spread = mu[:, -1] - mu[:, 0]  # [B]
    L_ls = -(spread / (sigma_y + eps)).mean()

    # Step F: Huber
    L_hub = F.smooth_l1_loss(
        y_hat * mask, y_norm * mask, beta=beta, reduction="sum"
    ) / (mask.sum() + eps)

    return L_mono + lam_ls * L_ls + lam_hub * L_hub, {
        "mu": mu.detach(),
        "L_mono": L_mono.detach(),
        "L_ls": L_ls.detach(),
        "L_hub": L_hub.detach(),
    }
```

### 1.5 Numerical stability & guardrails

- **τ_sort floor:** do not go below 1e-3; torchsort gradients blow up as τ→0.
- **Empty bucket:** if `M_k < 5` for any k in a batch (pathological small-N dates), skip that date's monotonicity term (mask it) rather than letting `μ_k` become garbage.
- **Degenerate compression monitor:** log `spread = μ_K − μ_1` in raw-return units per epoch. If `spread_raw` median drops below 0.5% annualized over a rolling 5-epoch window, increase `λ_ls` by 1.5× (capped at 1.0).
- **NaN policy:** if `torch.isnan(L_1).any()`, skip the optimizer step and log the offending date-batch for inspection (usually empty-bucket or zero-variance dates).
- **Winsorization precedes** all losses; never feed raw `y` to `L_mono` / `L_ls`.

### 1.6 Known failure modes

1. **Rank collapse:** model pushes all predictions to a narrow band so all `μ_k ≈ μ̄` and the hinge's margin is trivially satisfied. **Detect via** `std(y_hat) / std(y_norm)` dropping below 0.1; **fix by** raising `λ_hub`.
2. **Spread inversion in the middle:** adjacent hinges can be satisfied with a flat middle while top and bottom look good. Acceptable if only top/bottom are used; otherwise add a global monotonicity variant `Σ_{k<l} ReLU(μ_k − μ_l)²` as an auxiliary (O(K²)=100 terms, negligible).
3. **Date-dependent τ_assign drift** as N varies 950–1,050: define `τ_assign = ratio · N/K` rather than a fixed constant.

### 1.7 Success criteria for experimentation

Train 3 variants: (`λ_ls=0`, `λ_ls=0.3`, `λ_ls=1.0`) and compare on holdout:
- **Primary:** net-of-cost long-short Sharpe of top-minus-bottom decile (monthly rebalance, 10 bp one-way cost).
- **Secondary:** decile monotonicity (Spearman rank correlation of `μ_k` with `k`, mean across holdout dates) and turnover.
- **Baseline:** soft-rank IC loss (§2 optional comparator).

---

## §2 — Loss Spec #2: Tail-Restricted Pairwise Hinge + Huber

**Priority rank: 2.**

### 2.1 Objective
Directly enforce ordering **only between predicted-top and predicted-bottom names**, avoiding wasted capacity on mid-rank pairs that are economically irrelevant. This is the RICHE-style "tail focus" hybrid, motivated by the benchmark evidence that pairwise ranking losses beat MSE for portfolio construction but full O(N²) pairs waste effort on mid-ranked names that won't be traded.

### 2.2 Formula

**Step A — soft tail membership.** Using the same soft rank `r` as in §1.2 Step A:

```
t_i = σ( (r_i − r_hi) / τ_tail )      # soft "is top"
b_i = σ( (r_lo − r_i) / τ_tail )      # soft "is bottom"
```

with `r_hi = (1 − α) · N`, `r_lo = α · N`, `α = 0.1` (top/bottom decile), `τ_tail ∈ [0.05N, 0.2N]` in rank units.

Multiply by `mask` so invalid stocks contribute nothing.

**Step B — tail-restricted pairwise hinge (normalized):**

```
L_tail = ( Σ_{i,j, i≠j} t_i · b_j · ReLU( m_pair − (ŷ_i − ŷ_j) ) · mask_i · mask_j )
         / ( Σ_i t_i·mask_i · Σ_j b_j·mask_j + ε )
```

Interpretation: for every (predicted-top i, predicted-bottom j) pair, penalize if `ŷ_i − ŷ_j < m_pair`. This is a hinge (linear, robust to outliers), not squared.

**Step C — Huber auxiliary:**

```
L_hub = mean( SmoothL1( y_hat, y_norm, β=1.0 ) )
```

**Total loss:**

```
L_2 = L_tail + λ_hub · L_hub
```

### 2.3 Computational note

With `N≈1000`, `α=0.1`, the effective pair count is `~100 × 100 = 10⁴` per date. Vectorize directly as a `[B, N, N]` tensor if `B·N² ≈ 64 · 10⁶` fits memory (fp16 → ~128 MB); otherwise use **hard-tail sampling**: sample `S = 512` pairs `(i, j)` per date with probability `∝ t_i b_j`, and compute:

```
L_tail = mean_over_samples ReLU( m_pair − (ŷ_i − ŷ_j) )
```

Sampling is statistically consistent but **noisier**; use `S ≥ 256` and average over an epoch before judging.

### 2.4 Hyperparameters

| Name | Symbol | Range | Default | Notes |
|---|---|---|---|---|
| Tail fraction | α | 0.05 – 0.15 | 0.10 | Top/bottom decile |
| Tail temperature | τ_tail | 0.05N – 0.2N | 0.10N | Rank units; ~100 for N=1000 |
| Pair margin | m_pair | 0.3 – 1.5 | 0.5 | In `y_norm` σ units |
| Huber weight | λ_hub | 0.1 – 1.0 | 0.5 | Higher than §1 because tail loss alone is sparse |
| Huber β | β | 1.0 | 1.0 | `y_norm` σ units |
| Sort temperature | τ_sort | 0.01 | 0.01 | Same as §1 |

### 2.5 Pseudocode

```python
def loss_tail_pairwise(
    y_hat,
    y_norm,
    mask,
    alpha=0.1,
    tau_sort=0.01,
    tau_tail=None,
    m_pair=0.5,
    lam_hub=0.5,
    beta=1.0,
    sample_pairs=None,
):  # None = full NxN; else int
    B, N = y_hat.shape
    if tau_tail is None:
        tau_tail = 0.10 * N
    eps = 1e-8

    r = soft_rank(y_hat, regularization=tau_sort)
    r_hi = (1 - alpha) * N
    r_lo = alpha * N
    t = torch.sigmoid((r - r_hi) / tau_tail) * mask
    b = torch.sigmoid((r_lo - r) / tau_tail) * mask

    if sample_pairs is None:
        # Full pairwise
        diff = y_hat.unsqueeze(2) - y_hat.unsqueeze(1)  # [B, N, N]
        hinge = torch.relu(m_pair - diff)  # [B, N, N]
        w = t.unsqueeze(2) * b.unsqueeze(1)  # [B, N, N]
        num = (w * hinge).sum(dim=(1, 2))
        den = (t.sum(1) * b.sum(1)) + eps  # [B]
        L_tail = (num / den).mean()
    else:
        # Sampled pairwise (vectorized over B)
        probs = (t.unsqueeze(2) * b.unsqueeze(1)).flatten(1)  # [B, N*N]
        idx = torch.multinomial(probs + eps, sample_pairs, replacement=True)  # [B, S]
        i_idx = idx // N
        j_idx = idx % N
        yh_i = torch.gather(y_hat, 1, i_idx)  # [B, S]
        yh_j = torch.gather(y_hat, 1, j_idx)
        L_tail = torch.relu(m_pair - (yh_i - yh_j)).mean()

    L_hub = F.smooth_l1_loss(
        y_hat * mask, y_norm * mask, beta=beta, reduction="sum"
    ) / (mask.sum() + eps)
    return L_tail + lam_hub * L_hub, {
        "L_tail": L_tail.detach(),
        "L_hub": L_hub.detach(),
    }
```

### 2.6 Stability & guardrails

- **Avoid full `[B, N, N]`** if `B > 32`; use sampled pairs.
- **Pair-count sanity:** if `Σ t · Σ b < 50` for a date (extreme small-N or drift in τ_tail), skip that date's `L_tail` term.
- **Winsorize `y_norm` at ±4σ** before computing `L_hub` to prevent a single 2008-style extreme move from dominating.
- **`m_pair` rescaling:** since `y_norm` is rank-Gaussianized, its σ ≈ 1 per date by construction. `m_pair = 0.5` therefore means "top prediction must beat bottom prediction by at least half a cross-sectional standard deviation of returns" — a strong, interpretable requirement.
- **No compression risk** in this loss because Huber still anchors magnitudes; the main risk is the opposite — **over-concentration in a few extreme predictions**. Monitor `top-quintile prediction std` vs `bottom-quintile prediction std`.

### 2.7 Known failure modes

1. **Tail-saturation collapse:** model outputs +10 for one name and −10 for all others to trivially satisfy the hinge. **Detect via** prediction kurtosis > 50; **fix by** adding an L2 penalty `λ_w · mean(||y_hat||²)` with `λ_w ≈ 1e-3`.
2. **Middle-blindness:** predictions in the middle of the distribution are effectively unconstrained → poor if you later use intermediate buckets. This is **by design** for a top/bottom-only book; document as an explicit limitation.
3. **Sampled-pair variance:** loss curve will look noisier than §1; judge on 5-epoch moving average only.

### 2.8 Success criteria

- **Primary:** net-of-cost long-short Sharpe vs §1.
- **Secondary:** prediction-distribution diagnostics (kurtosis, std ratio tail/middle) to confirm the model isn't degenerate.
- **Comparison target:** if §2 beats §1 on net Sharpe and holds up under a 2× cost stress test, §2 is preferable for its stronger tail focus.

---

## §3 — Loss Spec #3: Portfolio-Return with Turnover Penalty

**Priority rank: 3 — run only after PiT universe audit.**

**Precondition (hard gate):** verify (a) delisted-name presence in every month 2000–2026, (b) terminal returns for delistings filled with a documented convention (e.g., −30% for bankruptcies, actual CRSP delisting return otherwise), (c) Russell 1000 membership reconstructed from the historical constituent file, not from current constituents. **If any of (a)–(c) fails, do not run this loss** — it will memorize the survivorship structure and produce inflated backtests.

### 3.1 Objective
Directly optimize the net-of-cost long-short portfolio return implied by the model's implied weights, without an intermediate rank-based objective. Most economically aligned loss; highest risk of bias memorization.

### 3.2 Formula

**Step A — soft decile weights.** Reuse soft assignment `p[i,k]` from §1.2 Steps A–B. Construct dollar-neutral long-short weights with unit gross:

```
w_long_i  = p[i,K] / ( Σ_j p[j,K] + ε )
w_short_i = p[i,1] / ( Σ_j p[j,1] + ε )
w_i       = w_long_i − w_short_i
w_i       = 2 · w_i / ( Σ_j |w_j| + ε )         # 100/100 gross
```

**Step B — realized portfolio return per date:**

```
R_t = Σ_i w_{i,t} · y_{i,t}                     # y is the RAW (unwinsorized) realized return
```

⚠️ Use **raw `y`** here, not `y_norm`. This loss is in economic units.

**Step C — turnover and cost per date.** Let `w_{t−1}` be the previous date's weight vector (requires sequential batches within a backtest window):

```
TO_t  = 0.5 · Σ_i | w_{i,t} − w_{i,t−1} |       # half-turnover, standard convention
C_t   = c_bps · Σ_i | w_{i,t} − w_{i,t−1} |     # = 2 · c_bps · TO_t
R_net_t = R_t − C_t
```

**Step D — per-date loss with target-vol scaling:**

```
σ_target_t = target_annual_vol / sqrt(periods_per_year)     # e.g., 0.10 / sqrt(12) monthly
L_t = − R_net_t / ( σ_target_t + ε )
```

Alternative (batch-Sharpe form, use only when B ≥ 36):

```
L_sharpe = − mean_t(R_net) / ( std_t(R_net) + ε )
```

**Step E — weight-concentration regularizer (required):**

```
L_conc = λ_conc · mean_i( w_i² )
```

**Total loss:**

```
L_3 = mean_t(L_t) + λ_conc · L_conc
```

### 3.3 Hyperparameters

| Name | Symbol | Range | Default | Notes |
|---|---|---|---|---|
| Transaction cost | c_bps | 5 – 15 (bp) | 8 | One-way; 8 bp reasonable for Russell 1000 monthly rebalance |
| Target annual vol | σ_target | 5% – 15% | 10% | Scales loss magnitude |
| Concentration weight | λ_conc | 1e-3 – 1e-2 | 3e-3 | Prevents single-name blow-up |
| Rebalance frequency | — | monthly / weekly / daily | monthly | Drives turnover; monthly standard for PiT consistency |
| Sort temp | τ_sort | 0.01 | 0.01 | Same as §1 |
| Assign temp | τ_assign | 0.3·N/K | 30 | Same as §1 |

### 3.4 Pseudocode

```python
def loss_portfolio(
    y_raw,
    y_hat,
    mask,
    w_prev,
    K=10,
    tau_sort=0.01,
    tau_assign=None,
    c_bps=8.0,
    target_vol=0.10,
    periods_per_year=12,
    lam_conc=3e-3,
):
    B, N = y_hat.shape
    if tau_assign is None:
        tau_assign = 0.3 * (N / K)
    eps = 1e-8

    # Soft assignment (same as §1 Step A-B)
    r = soft_rank(y_hat, regularization=tau_sort)
    c = (torch.arange(1, K + 1, device=y_hat.device) - 0.5) * (N / K)
    d = r.unsqueeze(-1) - c.unsqueeze(0).unsqueeze(0)
    p = torch.softmax(-(d**2) / (2 * tau_assign**2), dim=-1) * mask.unsqueeze(-1)

    # Weights
    w_long = p[..., -1] / (p[..., -1].sum(1, keepdim=True) + eps)
    w_short = p[..., 0] / (p[..., 0].sum(1, keepdim=True) + eps)
    w = w_long - w_short  # [B, N]
    gross = w.abs().sum(1, keepdim=True) + eps
    w = 2.0 * w / gross

    # Realized return (raw y)
    R = (w * y_raw * mask).sum(1)  # [B]

    # Turnover vs previous weights; w_prev shape [B, N], aligned date-shift
    dW = (w - w_prev) * mask
    cost = (c_bps / 1e4) * dW.abs().sum(1)  # [B]

    R_net = R - cost
    sigma_target = target_vol / (periods_per_year**0.5)
    L_t = -R_net / (sigma_target + eps)  # [B]

    L_conc = lam_conc * (w * mask).pow(2).mean()
    return L_t.mean() + L_conc, {
        "R": R.detach(),
        "cost": cost.detach(),
        "turnover": (0.5 * dW.abs().sum(1)).detach(),
        "w": w.detach(),
    }
```

### 3.5 Critical implementation details

- **`w_prev` alignment:** `w_prev[t]` must be the weight vector actually held going into period `t`. At the start of each training epoch or backtest window, initialize `w_prev = 0` and warm-up over the first 5–10 dates, discarding their losses.
- **Sequence-preserving batching:** dates inside a batch must be contiguous in time. Do **not** shuffle dates within a batch; use **block CV** with strictly time-ordered train/val/test splits (no random CV — random CV leaks future returns across adjacent dates).
- **`y_raw` input:** winsorize at ±20% to prevent a single delisting-return bug from producing a 10× outlier loss. Otherwise pass raw returns.
- **Numerical safety in `w`:** cap `w_i` at ±0.05 (5% of gross per name) via `w = w.clamp(-0.05, 0.05)` then renormalize; this is a hard concentration guard that duplicates `L_conc`.

### 3.6 Guardrails against survivorship-bias memorization

1. **Holdout by decade:** evaluate on 2000–2005, 2006–2012, 2013–2019, 2020–2026 separately. A portfolio loss that shows strong spread in 2013–2019 but weak spread in 2000–2005 is likely memorizing the modern mega-cap structure.
2. **Compare against §1 / §2 in the same splits.** If §3's advantage over §1 is concentrated in late-period (post-2013) splits, suspect survivorship contamination.
3. **Placebo test:** re-run §3 on a **randomly subsampled 50%** of the PiT universe. If Sharpe drops by less than 10%, suspect spurious concentration; if it drops by more than 40%, the loss is fragile.
4. **Cost stress test:** report Sharpe at 8 bp and at 25 bp. A portfolio loss that beats §1 at 8 bp but loses at 25 bp is chasing turnover and not robust.

### 3.7 Known failure modes

1. **Survivorship memorization** — the dominant risk; addressed by §3.6 gating.
2. **Turnover explosion** — as soft deciles oscillate with small prediction changes. **Detect via** median monthly turnover > 60% (100/100 gross); **fix by** increasing `c_bps` or adding a `λ_turn · mean(TO_t)` penalty.
3. **Regime over-fit** — model finds a weighting that concentrates in whatever factor dominated the latest regime. **Detect via** rolling 12-month Sharpe std across holdout; **fix by** shorter training windows (rolling 5-year retrain).
4. **Loss scale instability** across B: per-date `L_t` is unbounded (individual return can be ±30%); `mean_t(L_t)` is dominated by worst dates. **Fix by** winsorizing `R_net` at ±5σ_target before averaging, or use the Sharpe form.

### 3.8 Success criteria

- **Primary:** net Sharpe (10 bp) on each decade split, compared to §1/§2 on the same splits.
- **Secondary:** turnover median, net Sharpe at 25 bp stress, decade-wise stability (min over 4 splits of Sharpe).
- **Kill criterion:** if §3 does not beat §1 on at least 2 of 4 decade splits **and** on the 25-bp stress test, reject §3 and keep §1 as the production candidate.

---

## §4 — Shared evaluation protocol (applies to all three)

Regardless of the loss function, always report on the same holdout for comparability:

| Metric | Definition | Why |
|---|---|---|
| **Net LS Sharpe** | long-short top/bottom decile of `y_hat`, monthly rebalance, `c_bps=10` per side | Primary economic metric |
| **Decile monotonicity** | mean over dates of `Spearman([1..10], [μ_1..μ_10])` on **raw** `y` | Regime-robust signal quality |
| **Turnover** | median monthly half-turnover | Cost sensitivity |
| **Gross spread** | median `(μ_10 − μ_1)` in raw-return units | Economic magnitude |
| **Max drawdown** | of cumulative net long-short return | Tail robustness across 2000–2026 |
| **Decade splits** | Sharpe on 2000–05 / 06–12 / 13–19 / 20–26 | Regime stability |

**Baselines to beat:** (1) standard MSE on `y_norm`, (2) differentiable soft-rank IC loss, (3) equal-weight 12-1 momentum as a non-ML sanity floor.

**Kill criteria for any loss:** if it does not beat equal-weight momentum net Sharpe by ≥ 0.3 across **at least 3 of 4 decade splits**, reject.

---

## §5 — Summary table for implementation chat

| Spec | Loss | Key inputs beyond `y_hat, y_norm, mask` | Required libs | Priority |
|---|---|---|---|---|
| §1 | `L_mono + λ_ls·L_ls + λ_hub·L_hub` | — | `torchsort` (or fallback) | **1** |
| §2 | `L_tail + λ_hub·L_hub` | — | `torchsort` (or fallback) | 2 |
| §3 | `mean_t(−R_net/σ_target) + λ_conc·L_conc` | `y_raw`, `w_prev` (sequence-aligned) | `torchsort` + backtest harness | 3 (post-audit) |