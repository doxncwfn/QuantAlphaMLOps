# PiT Universe Construction and Model Input Window Specification

## 1. Purpose and Scope

This specification defines:

1. how the historical point-in-time (PiT) Russell 1000 universe is constructed from the available annual constituent snapshots;
2. how daily security observation status is interpreted;
3. how temporary missing observations are distinguished from terminal security disappearance;
4. how the fixed historical input window for the LSTM + attention + cross-sectional Transformer is constructed;
5. how a dynamically changing number of securities is handled without forcing the economic universe to a fixed size; and
6. how these components interact without introducing survivorship bias or look-ahead leakage.

The fundamental principle is:

> **PiT membership, market-data availability, model-input eligibility, and future-target availability are separate concepts and must never be represented by a single "stock exists" flag.**

---

# 2. Data Contract and Historical Universe Assumption

## 2.1 Annual Russell 1000 snapshots

Because exact daily Russell 1000 constituent histories are unavailable for the entire study period, the project uses annual historical Russell 1000 constituent snapshots recovered from available sources.

Each snapshot \(S_y\) represents the Russell 1000 constituent universe for approximately one annual interval:

\[
P_y=[30/06/y,\ 30/06/(y+1))
\]

or the corresponding project-defined trading-day boundaries.

Therefore, for every trading date \(t\in P_y\):

\[
\boxed{U_t=S_y}
\]

where \(U_t\) is the historical PiT Russell 1000 membership universe.

This is an **annual-snapshot approximation**, not a claim that the actual Russell 1000 membership was literally unchanged every day during that year.

The approximation is a deliberate consequence of the available historical data and must be documented as such.

---

# 3. Security Membership

For security \(i\) and trading date \(t\), define the PiT membership indicator:

\[
M_{i,t}=
\begin{cases}
1,&i\in U_t\\
0,&i\notin U_t
\end{cases}
\]

Thus:

\[
\boxed{
M_{i,t}=1
\iff
i\text{ belongs to the annual Russell snapshot assigned to }t
}
\]

Membership is determined from the historical constituent snapshot and security-identity resolution.

It is **not inferred from whether an OHLCV observation happens to exist on an individual day**.

---

# 4. Observation Availability

For every security-date pair, independently define:

\[
O_{i,t}=
\begin{cases}
1,&\text{a valid market observation is available}\\
0,&\text{no valid market observation is available}
\end{cases}
\]

The key distinction is:

\[
\boxed{
M_{i,t}\neq O_{i,t}
}
\]

A security can therefore have:

\[
M_{i,t}=1,\qquad O_{i,t}=0.
\]

This means:

> The security is a member of the PiT universe, but a valid market observation is unavailable on that date.

It does **not** automatically mean that the security was removed from the Russell 1000.

---

# 5. Interpretation of Security Disappearance

Under the project's data assumption, if a security is observed as part of the universe and then permanently disappears from the subsequent daily observations, the disappearance is treated as a **terminal security event**.

We deliberately do not attempt to distinguish whether the underlying economic mechanism was:

- bankruptcy,
- acquisition,
- merger,
- exchange-related delisting,
- ticker retirement,
- corporate restructuring,
- or another form of terminal disappearance.

For this project:

\[
\boxed{
\text{Permanent disappearance from observed market history}
=
\text{terminal event}
}
\]

The exact economic reason is outside the scope of PiT universe construction.

---

# 6. Temporary Missing Observation vs. Terminal Disappearance

This distinction is mandatory.

### Temporary missing observation

Example:

\[
O_{i,t-1}=1,\quad
O_{i,t}=0,\quad
O_{i,t+1}=1
\]

The security disappears for one observation and subsequently returns.

Interpretation:

\[
\boxed{\text{temporary data/market-observation gap}}
\]

The security remains a member:

\[
M_{i,t}=1.
\]

It must **not** be treated as delisted.

---

### Terminal disappearance

Example:

\[
O_{i,t-1}=1
\]

followed by no valid observations thereafter, subject to the project's terminality detection rules.

Interpretation:

\[
\boxed{\text{terminal security event}}
\]

The security remains historically valid in every prior PiT universe in which it was a member.

Its disappearance must **not retroactively remove it from earlier universes**.

---

# 7. Example of Terminal Disappearance

Suppose:

```text
05/31:
1000 observed Russell securities

06/01:
990 observed securities
```

and ten securities from 05/31 never subsequently return.

Then:

\[
D_{05/31\rightarrow06/01}
=
U_{05/31}\setminus U_{06/01}
\]

contains those ten terminally disappearing securities.

For every such security \(i\):

\[
M_{i,05/31}=1
\]

and therefore it remains a valid member of the 05/31 PiT universe.

The fact that it disappears on 06/01 must not cause:

\[
M_{i,05/31}
\]

to become zero.

---

# 8. No Retroactive Survivorship Filtering

A security that subsequently disappears must not be removed from historical samples merely because it cannot produce a conventional future observation.

Specifically, the following operation is prohibited:

\[
U_t
\rightarrow
U_t\setminus
\{\text{securities that later disappear}\}.
\]

That would introduce future information into the historical universe and potentially create survivorship bias.

The correct principle is:

\[
\boxed{
\text{Historical membership is determined at }t,
\text{ not by what happens after }t.
}
\]

---

# 9. PiT Universe vs. Model-Input Universe

The PiT universe is the economic/historical universe:

\[
\boxed{U_t}
\]

The model-input universe is a subset determined by data availability:

\[
\boxed{E_t\subseteq U_t}
\]

where \(E_t\) is the set of securities eligible to produce a valid model input at prediction date \(t\).

This distinction is fundamental.

A security may satisfy:

\[
i\in U_t
\]

but fail to satisfy:

\[
i\in E_t
\]

because its historical feature window contains too much missing data.

This does **not** mean it was not a Russell 1000 constituent.

---

# 10. Fixed Input Window

The model uses a fixed lookback length:

\[
\boxed{L=60}
\]

trading sessions.

For prediction date \(t\), define the global market trading calendar:

\[
\mathcal{D}
=
(d_1,d_2,\ldots,d_T).
\]

If \(t=d_j\), then the model input window is:

\[
\boxed{
W_t=
\{d_{j-L+1},\ldots,d_j\}
}
\]

or equivalently:

\[
\boxed{
W_t=[t-59,\ldots,t]
}
\]

in trading-session notation.

The window therefore contains exactly 60 **market trading sessions**, not 60 valid observations for an individual security.

---

# 11. Do NOT Shift the Window to Obtain 60 Valid Observations

This rule is explicit.

Suppose:

```text
t-59  observed
t-58  observed
...
t-31  observed
t-30  missing
t-29  observed
...
t     observed
```

The model window remains:

\[
[t-59,\ldots,t].
\]

We do **not** move the beginning to \(t-60\) merely to obtain 60 valid observations.

We do **not** search backward until 60 valid observations have been found.

Therefore, the following is prohibited:

\[
W_{i,t}^{*}
=
\{\text{previous 60 valid observations of }i\}.
\]

That would cause different securities to represent different amounts of market time.

For example, one security could have:

\[
60\text{ observations}=60\text{ trading days}
\]

while another could have:

\[
60\text{ observations}=70\text{ trading days}.
\]

The LSTM would then receive sequences with identical tensor length but different temporal meaning.

---

# 12. Why the Fixed Trading-Session Window Is Required

The temporal dimension of the model must have a consistent interpretation.

For every security:

\[
X_{i,t}
=
[x_{i,d_{j-59}},
x_{i,d_{j-58}},
\ldots,
x_{i,d_j}]
\]

represents:

> the security's market history over the same 60 global trading sessions ending at \(t\).

This is essential for the LSTM and temporal-attention mechanism to learn meaningful temporal patterns.

---

# 13. Missing Observations Inside the 60-Day Window

Define the observation mask:

\[
Q_{i,d}=
\begin{cases}
1,&O_{i,d}=1\\
0,&O_{i,d}=0
\end{cases}
\]

For a 60-session window:

\[
Q_{i,t}^{(60)}
=
[
Q_{i,d_{j-59}},
\ldots,
Q_{i,d_j}
].
\]

Define the number of missing observations:

\[
\boxed{
m_{i,t}
=
\sum_{d\in W_t}(1-O_{i,d})
}
\]

where:

\[
0\le m_{i,t}\le60.
\]

Example:

\[
m_{i,t}=1
\]

means exactly one market observation is missing within the 60-session window.

---

# 14. Missingness Does Not Automatically Mean Sample Rejection

A temporary missing observation does not automatically invalidate the entire model sample.

Instead, define a configurable missingness tolerance:

\[
\boxed{M_{\max}}
\]

and require:

\[
\boxed{
m_{i,t}\le M_{\max}
}
\]

for model-input eligibility.

Therefore:

\[
\boxed{
E_t=
\left\{
i\in U_t:
m_{i,t}\le M_{\max}
\right\}
}
\]

The exact value of \(M_{\max}\) should be determined from data-quality analysis and sensitivity experiments rather than silently hard-coded as a universal truth.

A clean baseline experiment may use:

\[
M_{\max}=0,
\]

while a robustness experiment may allow a small number of missing sessions.

---

# 15. Missing Data Must Remain Explicit

Missing observations must not be silently converted into genuine observations.

The model-input pipeline should retain an observation mask:

\[
Q_{i,t}\in\{0,1\}^{60}.
\]

Thus the model conceptually receives:

\[
\boxed{
(X_{i,t},Q_{i,t})
}
\]

rather than only \(X_{i,t}\).

This allows the model or preprocessing layer to distinguish:

> observed value

from:

> value that required missing-data treatment.

---

# 16. Missing-Value Treatment

If a model implementation requires a numerically complete tensor, missing feature values may be imputed using a predefined **leakage-safe** method.

The imputation procedure must satisfy:

\[
\boxed{
\text{Imputation at }d
\text{ may not use information unavailable at }d.
}
\]

Future observations must never be used to repair an earlier point.

The missingness mask \(Q_{i,d}\) must be retained even after imputation.

Therefore:

\[
\text{imputation}
\neq
\text{pretending the observation was genuine}.
\]

The precise imputation method is a separate preprocessing specification and is not part of PiT membership construction.

---

# 17. Do Not Blindly Forward-Fill All Features

Different features have different semantics.

For example:

- price,
- volume,
- return,
- volatility,
- RSI,
- Bollinger Bands,
- other derived indicators

should not automatically receive identical forward-fill treatment.

Any imputation strategy must be defined per feature class and validated for leakage.

The PiT layer itself must preserve the original missingness information.

---

# 18. Newly Entering Securities

A security does **not** need to have been a Russell 1000 constituent for all 60 previous sessions in order to enter the model.

Eligibility is based on:

\[
i\in U_t
\]

plus sufficient valid market-data history.

Therefore, if a stock becomes a Russell constituent at \(t\) but has been publicly traded for years, its pre-membership market history may be used to construct:

\[
X_{i,t}.
\]

We do **not** impose:

\[
i\in U_{t-59},\ldots,U_t.
\]

This is intentional.

The LSTM models the security's historical market behavior, while the PiT universe determines whether the security belongs to the cross-section being modeled at prediction date \(t\).

---

# 19. Input Tensor Shape

For security \(i\) at prediction date \(t\):

\[
\boxed{
X_{i,t}\in\mathbb{R}^{L\times F}
}
\]

with:

\[
L=60.
\]

For the cross-section:

\[
\boxed{
X_t\in\mathbb{R}^{N_t\times60\times F}
}
\]

where:

\[
\boxed{
N_t=|E_t|
}
\]

and \(N_t\) is allowed to change from day to day.

This is the correct economic representation.

---

# 20. Dynamic Number of Securities Is Not a PiT Problem

The fact that:

\[
N_t\ne N_{t+1}
\]

does not justify modifying the PiT universe.

The PiT universe remains historically faithful:

\[
U_t.
\]

The model-input universe may vary naturally:

\[
E_t.
\]

The model implementation must therefore support a variable cross-sectional dimension.

---

# 21. Batch Construction for the Transformer

A standard dense tensor requires a common tensor shape within a batch.

For a batch of \(B\) prediction dates:

\[
t_1,\ldots,t_B
\]

define:

\[
N_{\max}
=
\max_{b=1,\ldots,B}|E_{t_b}|.
\]

Construct:

\[
\boxed{
\tilde X
\in
\mathbb{R}^{B\times N_{\max}\times60\times F}
}
\]

by padding cross-sections smaller than \(N_{\max}\).

A corresponding stock-level mask is required:

\[
\boxed{
A\in\{0,1\}^{B\times N_{\max}}
}
\]

where:

\[
A_{b,j}=
\begin{cases}
1,&\text{real security}\\
0,&\text{padding}
\end{cases}
\]

The Transformer must prevent padded positions from participating in attention.

---

# 22. Padding Is a Computational Device, Not an Economic Security

A padding position is **not** a fake stock.

It must not:

- receive a ranking;
- contribute to the loss;
- participate in attention;
- enter portfolio construction;
- affect cross-sectional normalization;
- be interpreted as a Russell constituent.

Therefore:

\[
\boxed{
\text{padding exists only to satisfy tensor-shape requirements.}
}
\]

---

# 23. Cross-Sectional Attention

After the temporal encoder, each security produces a fixed-dimensional representation:

\[
h_{i,t}
=
TemporalEncoder(X_{i,t},Q_{i,t})
\]

where:

\[
h_{i,t}\in\mathbb{R}^{D}.
\]

Stacking the eligible securities gives:

\[
H_t
=
[h_{1,t},\ldots,h_{N_t,t}]
\]

with:

\[
\boxed{
H_t\in\mathbb{R}^{N_t\times D}.
}
\]

The cross-sectional Transformer operates on this dimension:

\[
\boxed{
Z_t
=
Transformer(H_t,A_t).
}
\]

Therefore the model can naturally perform attention across the actual securities available in \(E_t\).

---

# 24. Final Model Flow

The complete data flow is:

\[
\boxed{
\text{Annual Russell Snapshot}
\rightarrow
U_t
\rightarrow
\text{60-session window}
\rightarrow
E_t
\rightarrow
X_t
\rightarrow
LSTM
\rightarrow
Temporal Attention
\rightarrow
Cross-sectional Transformer
\rightarrow
Ranking Head
}
\]

More explicitly:

```text
Annual Russell 1000 snapshot
              │
              ▼
       PiT universe U_t
              │
              ▼
     Current-date members
              │
              ▼
     Construct fixed 60-
      trading-session window
              │
              ▼
   Evaluate observation mask
              │
              ▼
     m(i,t) <= M_max ?
          /          \
        yes           no
         │             │
         ▼             ▼
   model candidate   exclude
         │
         ▼
   X_i,t : 60 × F
   Q_i,t : 60
         │
         ▼
       LSTM
         │
         ▼
 Temporal attention
         │
         ▼
 H_t : N_t × D
         │
         ▼
Cross-sectional Transformer
     + padding mask
         │
         ▼
     ranking scores
         │
       ┌─┴─┐
       ▼   ▼
    Top k% Bottom k%
```

---

# 25. Future Disappearance and the Prediction Target

Terminal disappearance must be handled separately from input construction.

For a prediction date \(t\), define a future terminal-event indicator:

\[
\boxed{
D_{i,t}^{(H)}
=
\mathbb{1}
\left[
\text{security }i\text{ terminates within the next }H
\text{ trading sessions}
\right].
}
\]

For the project's five-session prediction horizon:

\[
D_{i,t}^{(5)}.
\]

This variable may be used for separate event analysis or a future auxiliary task.

However:

\[
\boxed{
D_{i,t}^{(5)}
\text{ must not automatically be converted into a numerical return such as }-100\%.
}
\]

The current common dataset does not provide a universally reliable delisting-return measure across the entire 2000–2026 period.

Therefore, terminal disappearance and five-day investment return must remain separate concepts unless a defensible total-return/delisting-return methodology is introduced later.

---

# 26. Target Eligibility Is Separate From Input Eligibility

Define:

\[
E_t^{input}
\]

as securities with sufficiently valid 60-session input histories.

Separately define:

\[
E_t^{target}
\]

as securities for which the selected future target can be validly calculated.

Thus:

\[
\boxed{
E_t^{target}\subseteq E_t^{input}
}
\]

is possible, but not necessarily guaranteed.

A security may have a perfectly valid input at \(t\) and subsequently terminate before \(t+5\).

That does not invalidate its historical input.

It creates a target-definition problem that must be handled by the target-construction layer.

---

# 27. Leakage Rules

The following rules are mandatory.

### Rule 1 — No future membership leakage

The model input at \(t\) may not use information from \(t+1\) onward to determine whether a security belongs to \(E_t\).

### Rule 2 — No future data imputation

Missing values at \(d\le t\) cannot be repaired using observations after \(d\) unless the procedure is explicitly justified as an information-available-at-\(t\) transformation.

### Rule 3 — No future survival filtering

A security must not be removed from \(U_t\) because it subsequently disappears.

### Rule 4 — No variable-length historical compression

Do not search backward for 60 valid observations.

### Rule 5 — No artificial universe stabilization

Do not force \(N_t=1000\) by adding non-members or retaining future members.

### Rule 6 — Padding must not affect model output

Padding positions must be masked from attention and loss computation.

---

# 28. Data Layers

The implementation should maintain the following conceptual layers.

### Layer 1 — Historical membership

\[
\boxed{U_t}
\]

Answers:

> Who belongs to the Russell 1000 PiT universe on date \(t\)?

### Layer 2 — Observation status

\[
\boxed{O_{i,t}}
\]

Answers:

> Do we have a valid market observation for security \(i\) on date \(t\)?

### Layer 3 — Input eligibility

\[
\boxed{E_t^{input}}
\]

Answers:

> Can security \(i\) produce a sufficiently valid 60-session model input at \(t\)?

### Layer 4 — Target eligibility

\[
\boxed{E_t^{target}}
\]

Answers:

> Can the selected future target be calculated for security \(i\) at \(t\)?

### Layer 5 — Model tensor

\[
\boxed{
X_t\in\mathbb{R}^{N_t\times60\times F}
}
\]

Answers:

> What exactly is presented to the LSTM + attention + Transformer?

These layers must remain independently auditable.

---

# 29. Final Definitions

The final specification therefore uses the following definitions.

### PiT membership

\[
\boxed{
M_{i,t}=1
\iff
i\in S_y,\quad t\in P_y
}
\]

### Observation

\[
\boxed{
O_{i,t}=1
\iff
\text{valid market observation exists for }i,t
}
\]

### Missingness count

\[
\boxed{
m_{i,t}
=
\sum_{d\in W_t}(1-O_{i,d})
}
\]

### Model-input eligibility

\[
\boxed{
E_t^{input}
=
\{i\in U_t:m_{i,t}\le M_{\max}\}
}
\]

### Fixed temporal window

\[
\boxed{
W_t=[d_{j-59},\ldots,d_j]
}
\]

for prediction date \(t=d_j\).

### Per-security input

\[
\boxed{
X_{i,t}\in\mathbb{R}^{60\times F}
}
\]

### Cross-sectional input

\[
\boxed{
X_t\in\mathbb{R}^{N_t\times60\times F},
\quad
N_t=|E_t^{input}|
}
\]

### Batched representation

\[
\boxed{
\tilde X
\in
\mathbb{R}^{B\times N_{\max}\times60\times F}
}
\]

with an explicit padding/attention mask.

### Terminal event

\[
\boxed{
D_{i,t}^{(5)}
=
\mathbb{1}
[
i\text{ terminates within five future trading sessions}
]
}
\]

which is kept separate from the ordinary future-return/ranking target.

---

# 30. Final Design Decision

The project therefore adopts the following principle:

\[
\boxed{
\textbf{Fixed temporal window + dynamic cross-sectional universe}
}
\]

not:

\[
\text{fixed 60 observations + dynamic temporal span}
\]

and not:

\[
\text{fixed 1,000 stocks + artificial membership}.
\]

In practical terms:

> **Every prediction date uses the same 60 global trading sessions as its temporal window. Every security currently belonging to the PiT Russell universe is evaluated against that window. Temporary missing observations are represented explicitly and handled through a predefined missingness tolerance. Securities that permanently disappear remain valid historical members up to their terminal date and are never retroactively removed. The resulting model-input cross-section is allowed to vary in size. Padding and an attention mask are used only at batch-construction time to make the variable-size cross-section compatible with the LSTM + temporal-attention + cross-sectional-Transformer implementation.**

This is the canonical PiT/input-window specification for the project.