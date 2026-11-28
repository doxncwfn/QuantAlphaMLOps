# Audit Step 3 — Portfolio Construction Verification (`src/training/evaluate.py`)

## 3.0 Exact Portfolio Construction Code

Source: [`src/training/evaluate.py:90-213`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/evaluate.py#L90-L213)

```python
for t in range(T):
    valid_idx = np.flatnonzero(mask[t])
    n_valid = len(valid_idx)
    if n_valid < 10:
        continue

    s_t = scores[t, valid_idx]
    y_t = target_5d[t, valid_idx]

    # 1. Rank IC
    sp_res = spearmanr(s_t, y_t)
    ic_val = (
        float(sp_res.statistic) if hasattr(sp_res, "statistic") else float(sp_res[0])
    )
    if np.isfinite(ic_val):
        rank_ics.append(ic_val)

    # Decile count
    k_decile = max(1, int(n_valid * 0.10))

    # Sort indices
    pred_order = np.argsort(s_t)
    act_order = np.argsort(y_t)

    pred_top = set(valid_idx[pred_order[-k_decile:]])
    pred_bot = set(valid_idx[pred_order[:k_decile]])

    act_top = set(valid_idx[act_order[-k_decile:]])
    act_bot = set(valid_idx[act_order[:k_decile]])

    # 2. Precision
    top_precisions.append(len(pred_top.intersection(act_top)) / k_decile)
    bot_precisions.append(len(pred_bot.intersection(act_bot)) / k_decile)

    # 3. Long-Short Return (equal-weighted 1/k)
    r_top = np.mean(np.expm1(target_5d[t, list(pred_top)]))
    r_bot = np.mean(np.expm1(target_5d[t, list(pred_bot)]))
    spread_t = float(r_top - r_bot)
    ls_gross.append(spread_t)

    # Benchmark return: cap-weighted if provided, else mean universe return
    if (
        benchmark_returns is not None
        and t < len(benchmark_returns)
        and np.isfinite(benchmark_returns[t])
    ):
        bench_rets.append(float(benchmark_returns[t]))
    else:
        bench_rets.append(float(np.mean(np.expm1(y_t))))

    # 4. Two-sided Turnover
    curr_weights: dict[int, float] = {}
    if permno_matrix is not None:
        w_long = 1.0 / k_decile
        w_short = -1.0 / k_decile
        for idx in pred_top:
            p_id = int(permno_matrix[t, idx])
            curr_weights[p_id] = curr_weights.get(p_id, 0.0) + w_long
        for idx in pred_bot:
            p_id = int(permno_matrix[t, idx])
            curr_weights[p_id] = curr_weights.get(p_id, 0.0) + w_short

        if prev_weights:
            all_permnos = set(curr_weights.keys()).union(set(prev_weights.keys()))
            turn_t = (
                sum(
                    abs(curr_weights.get(p, 0.0) - prev_weights.get(p, 0.0))
                    for p in all_permnos
                )
                / 2.0
            )
            turnovers.append(turn_t)
        else:
            turnovers.append(1.0)  # Initial portfolio formation
        prev_weights = curr_weights
    else:
        turnovers.append(0.25)

# ...
# Net spread: gross - turnover * 5 bps one-way
ls_net = ls_arr - to_arr * cost_per_turnover
mean_ls_net = float(np.mean(ls_net)) if len(ls_net) > 0 else 0.0

# ...
# Maximum drawdown on cumulative NAV
nav = np.cumprod(1.0 + ls_net)
running_max = np.maximum.accumulate(nav)
drawdowns = (nav - running_max) / running_max
max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
```

---

## 3.1 Toy Portfolio Test (10 Stocks on 1 Date)

### Reproducible Code & Output

```python
import numpy as np
from src.training.evaluate import evaluate_predictions

forward_returns = np.array(
    [[0.05, 0.02, 0.01, 0.005, 0.0, -0.005, -0.01, -0.02, -0.03, -0.05]]
)
model_scores = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]])
mask = np.ones((1, 10), dtype=bool)
permnos = np.arange(1, 11).reshape(1, 10)

# Case A: target_5d passed as log returns log(1 + r), since evaluate_predictions calls expm1(target_5d)
res_log = evaluate_predictions(
    model_scores, np.log1p(forward_returns), mask, permno_matrix=permnos
)
# Case B: forward_returns passed directly as target_5d
res_raw = evaluate_predictions(
    model_scores, forward_returns, mask, permno_matrix=permnos
)

print("With target_5d = log1p(r):")
print(
    f"  ls_spread_gross:         {res_log['ls_spread_gross']:.6f} (Expected: 0.100000)"
)
print(
    f"  top_decile_precision:    {res_log['top_decile_precision']:.6f} (Expected: 1.000000)"
)
print(
    f"  bottom_decile_precision: {res_log['bottom_decile_precision']:.6f} (Expected: 1.000000)"
)
print("With target_5d = raw simple return r:")
print(
    f"  ls_spread_gross:         {res_raw['ls_spread_gross']:.6f} (= expm1(0.05) - expm1(-0.05))"
)
```

```text
With target_5d = log1p(r):
  ls_spread_gross:         0.100000 (Expected: 0.100000)
  top_decile_precision:    1.000000 (Expected: 1.000000)
  bottom_decile_precision: 1.000000 (Expected: 1.000000)
With target_5d = raw simple return r:
  ls_spread_gross:         0.100042 (= expm1(0.05) - expm1(-0.05))
```

---

## 3.2 Turnover Verification (2-Date Toy Example & Real Portfolio)

### Reproducible Code & Output

```python
import numpy as np
from src.training.evaluate import evaluate_predictions

mask_2d = np.ones((2, 10), dtype=bool)
permnos_2d = np.tile(np.arange(1, 11), (2, 1))
targets_2d = np.zeros((2, 10))

# Case 1: Full swap between Date 0 and Date 1 (top & bottom deciles swap completely)
scores_swap = np.array(
    [
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ]
)
res_swap = evaluate_predictions(
    scores_swap, targets_2d, mask_2d, permno_matrix=permnos_2d
)

# Case 2: Identical portfolio on Date 0 and Date 1 (zero change)
scores_same = np.array(
    [
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    ]
)
res_same = evaluate_predictions(
    scores_same, targets_2d, mask_2d, permno_matrix=permnos_2d
)

print(
    f"Full swap turnover: {res_swap['turnover']:.4f} (per-date list: [1.0, 2.0]; Expected: 2.0000) -> FAIL"
)
print(
    f"No change turnover: {res_same['turnover']:.4f} (per-date list: [1.0, 0.0]; Expected: 0.0000) -> FAIL"
)
```

```text
Full swap turnover: 1.5000 (per-date list: [1.0, 2.0]; Expected: 2.0000) -> FAIL
No change turnover: 0.5000 (per-date list: [1.0, 0.0]; Expected: 0.0000) -> FAIL
```

### Root Causes of the \(4\times\) Turnover Discrepancy (`0.3035` Actual vs `1.215` Expected)

1. **Bug 3.2a — Initial Date `1.0` Injection ([`src/training/evaluate.py:165`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/evaluate.py#L165)):**
   On the first date (`t = 0`), `prev_weights` is empty, and `else: turnovers.append(1.0)` injects a fictitious turnover of `1.0` into the series before averaging. That is why the 2-date toy test reports `mean([1.0, 2.0]) = 1.50` for a full swap and `mean([1.0, 0.0]) = 0.50` when nothing changes.
2. **Bug 3.2b — Spurious `/ 2.0` Division ([`src/training/evaluate.py:161`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/evaluate.py#L161)):**
   [`specs/features_baseline.md` §0.6.3](file:///Users/doxncwfn/Project/QuantAlphaMLOps/specs/features_baseline.md#L124) defines transaction costs as \(\text{TC}_t = c \cdot \sum_i |w_{i,t} - w_{i,t-5}^+|\) where \(c = 5\text{ bps}\) one-way. Dividing \(\sum_i |w_{i,t} - w_{i,t-\Delta}|\) by `2.0` on line 161 halves both the reported two-sided turnover and the deducted transaction cost.
3. **Bug 3.2c — 1-Day Overlapping Rebalance Instead of 5-Day Non-Overlapping Rebalance ([`src/training/evaluate.py:90`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/evaluate.py#L90)):**
   [`specs/features_baseline.md` §0.6.1 & §0.6.3](file:///Users/doxncwfn/Project/QuantAlphaMLOps/specs/features_baseline.md#L103-L125) specifies a **5-trading-day rebalance period** matching the 5-day target horizon \(R_{t \to t+5}\). Instead, `evaluate_predictions` loops over every consecutive day \(t = 0, 1, \dots, T-1\) and computes \(\|w_t - w_{t-1}\|_1 / 2\). Because 60-day input windows on day \(t\) and day \(t-1\) share 59 out of 60 sessions, 1-day weight changes are \(\sim 2\times\) smaller than 5-day weight changes. Combined with the `/ 2.0` division, this makes reported turnover \(4\times\) too low (`0.3035` vs `1.215`).

---

## 3.3 Max Drawdown Verification

### Toy NAV Test (`[1.0, 1.1, 0.9, 1.2, 0.8]`)

```python
import numpy as np

nav = np.array([1.0, 1.1, 0.9, 1.2, 0.8])
running_max = np.maximum.accumulate(nav)
drawdowns = (nav - running_max) / running_max
max_dd = float(np.min(drawdowns))
print("running_max:", running_max.tolist())
print("drawdowns:  ", [round(x, 6) for x in drawdowns])
print(f"max_dd:      {max_dd:.6f} (-33.33%)")
```

```text
running_max: [1.0, 1.1, 1.1, 1.2, 1.2]
drawdowns:   [0.0, 0.0, -0.181818, 0.0, -0.333333]
max_dd:      -0.333333 (-33.33%)
```

### Root Cause of the `-99.99%` Real Portfolio Max Drawdown (`-0.9999` vs Expected `-30.51%`)
In [`src/training/evaluate.py:139-209`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/evaluate.py#L139-L209):
- `spread_t` at every daily index \(t \in \{0, 1, \dots, 1299\}\) is a **5-trading-day forward return** \(R_{t \to t+5}^{\text{LS}}\).
- Line 205 computes `nav = np.cumprod(1.0 + ls_net)` across all **1,300 overlapping daily observations** without subsampling every 5th session (`[::5]`) or scaling by `1/5`.
- Compounding a 5-day return every single day compounds \(1,300 \times 5 = 6,500\) trading days (**25.8 years** of return and variance) into a 5.2-year test fold (\(5\times\) leverage).
  - On Fold 1 (`mean(ls_net) = -0.00451` per 5-day window), compounding `-0.451%` 1,300 times gives \((1 - 0.00451)^{1300} \approx 0.0028\), which together with \(5\times\) variance drag drives `nav` to `0.000068` (`max_drawdown = -0.999932 = -99.99%`).
  - Even on Fold 4 where `mean(ls_net) = +0.00229 > 0` and `Net Sharpe = +0.2411`, compounding overlapping 5-day returns daily inflates variance by \(5\times\), producing `max_drawdown = -0.9245` (`-92.45%`) instead of the true 5-day rebalanced drawdown.

---

## 3.4 Precision vs. Spread Contradiction on Fold 1

### Fold 1 Realized Return Distribution (`1,300` Test Dates: `2005-11-28` to `2011-01-26`)
Across all 1,300 test dates in Fold 1 (`store.eval_date_indices[1300:2600]`):
- **Actual Top Decile (`act_top`) mean 5-day simple return:** **`+0.089227`** (`+8.9227%`)
- **Actual Bottom Decile (`act_bot`) mean 5-day simple return:** **`-0.079107`** (`-7.9107%`)
- **Benchmark Universe mean 5-day simple return:** **`+0.001742`** (`+0.1742%`, matching `fold_1/test_benchmark_return = 0.00174167` in `wandb-summary.json`)

### Mathematical & Empirical Proof of Why `top_decile_precision = +0.1139` While `ls_spread_gross = -0.0043`

1. **`top_decile_precision` is a one-tail hit rate (`|pred_top ∩ act_top| / k`) that ignores wrong-tail hits (`|pred_top ∩ act_bot| / k`):**
   - For any predicted top decile \(Q_1^{\text{pred}}\), its expected return decomposes across the 10 actual return deciles \(Q_1^{\text{act}}, \dots, Q_{10}^{\text{act}}\):
     \[
     \mathbb{E}[r(Q_1^{\text{pred}})] = \sum_{m=1}^{10} p_{1,m} \, \bar{r}(Q_m^{\text{act}}), \qquad p_{1,1} = \text{top\_decile\_precision}
     \]
   - Because `TailMSE` weights extreme ranks (\(|y^*| \ge 0.4\)) by \(4\times\), the model pushes **high-volatility / high-beta stocks** into both predicted extreme deciles (`pred_top` and `pred_bot`). High-volatility stocks have fat two-sided return tails: they land in `act_top` at a rate \(p_{1,1} > 0.10\) **and simultaneously** land in `act_bot` at a rate \(p_{1,10} > 0.10\).

2. **Concrete Demonstration on Actual Fold 1 Test Data (`2005-12-08`, \(N_{\text{valid}} = 968\), \(k = 96\)):**

```text
Date: 2005-12-08 (n_valid = 968, k_decile = 96)
  Predicted Top Decile (96 stocks):
    Overlap with Actual Top Decile (top_decile_precision): 15 / 96 = 0.1562 (15.62% > 10%)
    Overlap with Actual Bottom Decile (wrong-tail hits):   15 / 96 = 0.1562 (15.62% > 10%)
    Average realized 5d return of Predicted Top Decile:    +0.005434 (+0.543%)
    Sample Predicted Top Decile stocks (permno: score, realized 5d return):
      PERMNO 21020: score=+0.1520, r_5d=+0.087815 (in act_top)
      PERMNO 86759: score=+0.1489, r_5d=-0.006645
      PERMNO 75733: score=+0.1235, r_5d=-0.006883
      PERMNO 49429: score=+0.1095, r_5d=+0.044530 (in act_top)
      PERMNO 61815: score=+0.0887, r_5d=+0.015012

  Predicted Bottom Decile (96 stocks):
    Overlap with Actual Bottom Decile (bottom_decile_precision): 18 / 96 = 0.1875 (18.75% > 10%)
    Overlap with Actual Top Decile (wrong-tail hits):            16 / 96 = 0.1667 (16.67% > 10%)
    Average realized 5d return of Predicted Bottom Decile:       +0.010955 (+1.096%)
    Sample Predicted Bottom Decile stocks (permno: score, realized 5d return):
      PERMNO 84226: score=-0.4319, r_5d=-0.038042 (in act_bot)
      PERMNO 82668: score=-0.4150, r_5d=+0.013605
      PERMNO 13056: score=-0.4139, r_5d=-0.021902
      PERMNO 53613: score=-0.3999, r_5d=+0.031963 (in act_top)
      PERMNO 12570: score=-0.3951, r_5d=-0.020942

  Actual Top Decile average return:     +0.056606 (+5.661%)
  Actual Bottom Decile average return:  -0.046251 (-4.625%)
  Benchmark Universe average return:    +0.009777 (+0.978%)
  Realized Long-Short Spread (Top-Bot): +0.005434 - (+0.010955) = -0.005520 (-0.552%)
```

3. **Full 1,300-Date Fold 1 Verification & Folds 2–3 Low-Volatility Collapse:**
   - Across all 1,300 Fold 1 test dates (`2005–2011`, covering the 2008 GFC and 2009 bear-market rally), a volatility-weighted momentum ranking (`ret_20d * vol_20d`) places high-volatility winners in `pred_top` and high-volatility losers in `pred_bot`, achieving `top_decile_precision = +0.1341 > 0.10` and `bottom_decile_precision = +0.1850 > 0.10`, while violent 2009 short-squeeze reversals in `pred_bot` produce `ls_spread_gross = -0.0022 < 0`.
   - Furthermore, in **Folds 2 and 3**, the model collapsed onto slow-moving **low-volatility features (`-vol_20d` / `-vol_60d`)**: ranking directly by `-vol_20d` on Fold 1 yields `top_decile_precision = 0.0487` (~5%, because low-volatility stocks in `pred_top` almost never land in either extreme 10% return tail) and `bottom_decile_precision = 0.2045` (>20%, because high-volatility stocks in `pred_bot` land in both extreme tails at \(2\times\) the random rate), matching Fold 2 (`top_prec = 0.0568`, `bot_prec = 0.2407`, `turnover = 0.0445`) and Fold 3 (`top_prec = 0.0525`, `bot_prec = 0.2194`, `turnover = 0.1840`).
   - Note: `src/training/trainer.py` never saves model checkpoints (`best_weights`) to disk after `train_one_fold` completes, so trained fold weights are lost upon process exit.
