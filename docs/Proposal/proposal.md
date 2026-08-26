# An End-to-End MLOps Pipeline for Cross-Sectional Equity Ranking under Data Constraints: Engineering Validation and Ablation Study.

## I. Motivation and Limitations

While deep learning architectures show theoretical promise in equity forecasting, transitioning these models to deployable pipelines requires robust MLOps frameworks, ranking-aware objectives, and strict validation protocols. This project develops an end-to-end engineering pipeline to evaluate a hybrid temporal-attention model under rigid computational constraints.

**Explicit Limitations:**  
The data universe is the intersection of 413 S&P 500 equities possessing complete, continuous daily split- and dividend-adjusted OHLCV history from 2020-01-01 onward. No attempt is made to reconstruct delisted or merged names; therefore, a structural survivorship bias is present and acknowledged. Intra-series missing bars or trading suspensions are strictly forward-filled to prevent look-ahead bias. Additionally, sector definitions rely on static, current GICS Level 1 assignments, acting as an acknowledged approximation.

Consequently, all financial metrics serve solely to validate mechanical pipeline health and must not be interpreted as deployable alpha. The data layer is decoupled, allowing a survivorship-free point-in-time database to be substituted without altering the core pipeline code.

## II. Phase 1: Signal Generation and MLOps Validation

This phase establishes a point-in-time data pipeline, trains the hybrid architecture against strict baselines, and utilizes Explainable AI to characterize logic shifts across volatility regimes.

**Target Residualization & Z-Scoring:**  
The target is a 5-day forward excess return residualized against a multi-factor risk model to isolate asset-specific price-volume dynamics:

$$
y_{i,t} = r_{i,t+5} - \sum_{k} \hat{\beta}_{i,k,t} f_{k,t+5}
$$

Returns ($r$) are total returns adjusted for splits and dividends. Daily Fama-French factor returns (Market, SMB, HML) are obtained from the Kenneth French Data Library. The 5-day forward factor returns are compounded daily:  
$f_{k,t+5} = \prod_{s=1}^{5}(1+f_{k,t+s})-1$.

The Volatility factor is constructed as the return differential between the bottom and top quintiles of trailing 60-day realized volatility, computed point-in-time. Exposures ($\hat{\beta}$) are estimated using strictly data from $t-126$ to $t$.

The target is then cross-sectionally z-scored per date $t$:  
$\tilde{y}_{i,t} = \frac{y_{i,t} - \mu_t}{\sigma_t}$.

**Architecture Specifications:**  
The network accepts input tensors of shape $[Batch\_Dates, 413, 60, 15]$. The LSTM is applied independently per stock, treating the stock dimension as part of the batch; no cross-sectional information enters the network until the Transformer layer.

The 2-layer LSTM's final hidden state is concatenated with a learned 32-dimensional static Stock ID embedding. This feeds a 2-layer, 4-head Transformer encoder (feedforward dimension 128) operating across the cross-section.

> **Note:** The performance difference between models with and without the Stock ID embedding will be reported as a primary ablation finding to explicitly monitor for survivor-universe memorization.

**Training & Capacity Ablation:**  
The optimizer is AdamW with a cosine learning rate schedule and a batch size of 32 dates. The current experimental baseline uses hidden size 256; the formal capacity ablation searches over $\{64, 128, 256\}$ within the nested validation loop. Dropout is treated as a continuous hyperparameter searched over $[0.1, 0.5]$ (with an exploratory run initialization of 0.1912).

**Hybrid AlphaLoss Formulation:**  
The objective function combines MSE with a pairwise hinge loss penalizing rank inversions. Let $P_t$ and $Q_t$ represent the top and bottom 10% of predicted scores on date $t$. The ranking loss is computed separately for each date $t$ and averaged across dates (deciles are never pooled across dates):

$$
L_{\text{rank},t} = \frac{1}{\vert{}P_t\vert{}\vert{}Q_t\vert{}} \sum_{i \in P_t} \sum_{j \in Q_t} \max\left(0,\; \alpha - (\tilde{y}_{i,t} - \tilde{y}_{j,t})\right)
$$

The margin $\alpha \in \{0.1, 0.25\}$ is in z-score units. The total loss is $L_{\text{total},t} = L_{\text{rank},t} + \lambda L_{\text{MSE},t}$, with $\lambda \in \{0.1, 0.5, 1.0\}$ selected in the inner walk-forward loop.

**Validation Protocol:**  
Evaluation utilizes an expanding 5-fold Purged Walk-Forward (approx. 12-month test windows, 5-day purge, 20-day embargo). Optuna hyperparameter optimization is performed exclusively on the inner training folds. The outer test folds remain completely untouched until after the architecture and hyperparameters are firmly locked.

**Regime-Conditioned XAI Diagnostics:**  
Data is segmented into High/Low regimes based on the trailing 60-day median of market-wide realized volatility, computed point-in-time. PyTorch Integrated Gradients are computed on a 5000-sample subset. LightGBM surrogate fidelity is evaluated via Spearman rank correlation between neural network and surrogate outputs. Attributions are interpreted strictly as descriptive logic shifts.

# III. Phase 2: Constrained Portfolio Construction

This phase translates the out-of-sample predictions from the locked final walk-forward test folds of Phase 1 into a simulated portfolio using convex optimization.

**Convex Optimization Engine:**  
Out-of-sample predictions drive a single convex objective utilizing Ledoit-Wolf covariance shrinkage ($\Sigma$):

$$
\max_{w} w^T\alpha - \lambda w^T\Sigma w - \gamma TC(w, w_{\text{prev}})
$$

**Transaction Costs (TCA) & Constraints:**  
To accurately capture non-linear scale dynamics, transaction costs integrate both a linear spread term and a quadratic market impact term:

$$
TC(w,w_{\text{prev}}) = \sum_i \left[ c_{\text{spread},i} \vert{}\Delta w_i\vert{} + c_{\text{impact},i} \left(\frac{\Delta w_i}{ADV_i}\right)^2 \right]
$$

An explicit turnover constraint $\vert{}\vert{}w - w_{\text{prev}}\vert{}\vert{}_1 \le \tau$ is enforced.

**Strict Risk Neutralization:**

- Dollar neutrality is enforced as a hard equality constraint ($\sum_i w_i = 0$).
- Gross leverage bounds ($\vert{}\vert{}w\vert{}\vert{}_1 \le 2.0$) and individual position limits ($-0.05 \le w_i \le 0.05$) are applied.
- Factor neutrality uses inequality constraints ($\vert{}X^T w\vert{} \le 0.1$) for the exact Fama-French factors (Market, SMB, HML) and the Volatility factor defined in Phase 1.
- Sector constraints utilize static GICS Level 1 dummy variables.

**Engineering Validation (Paper Trading):**  
The end-to-end pipeline is deployed to a brokerage API. Paper trading results are entirely excluded from financial conclusions, acting solely as validation of order routing and system orchestration.

# IV. Engineering KPIs and Secondary / Non-Inferential Diagnostics

All statistical diagnostics are reported as Mean $\pm$ Standard Deviation across the five outer test folds. Baseline models (Naive Momentum, ElasticNet, LightGBM, Pure MLP, Pure LSTM, Pure Transformer) are evaluated on exactly the same folds and dates.

**Engineering Targets:**

- Feature computation throughput $< 180$ seconds for the full 6-year panel
- Inference latency $< 50$ms per 413-stock batch
- Fully versioned experiment tracking with pinned dependencies and seed control (acknowledging standard GPU nondeterminism)

**Statistical Diagnostics:**

- Out-of-Sample Purged Mean IC
- IC Information Ratio (ICIR)
- % positive IC periods
- Newey-West adjusted t-statistics
- A Deflated Sharpe Ratio (or Probability of Backtest Overfitting) will be reported to rigorously control for multiple testing bias

**Secondary / Non-Inferential Economic Diagnostics:**  
Acknowledging the strict limitations of the dataset, these metrics serve only to proxy execution friction mechanics:

- Long-short decile spread monotonicity
- Annualized turnover
- Post-cost Sharpe Ratio
- The impact of turnover/spread costs on net returns
