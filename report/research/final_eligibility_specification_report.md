# Authoritative PiT Dataset Specification & Target Definition

**Project:** Russell 1000 Point-in-Time Quantitative ML Dataset  
**Target Architecture:** Dynamic Cross-Sectional Ranking Model ($L = 60$ Trading Sessions, $H = 5$ Sessions)  
**Specification Lock Timestamp:** 2026-10-03 03:25:39  

---

## 1. Executive Summary & Status Lock

> **VERDICT: READY_FOR_FEATURE_ENGINEERING = YES**

The dataset specification, target definition, target reconciliation, and observation semantics are **100% locked**. Downstream feature engineering may proceed under the formal rules specified in this document.

### Locked Core Architecture
- **Lookback Window ($W_t$):** Fixed 60 global US trading sessions ($d_{t-59} \dots d_t$). No backward search, no sequence compression.
- **Universe ($U_t$):** Annual Russell 1000 Point-in-Time constituents assigned to snapshot $U_t$.
- **Input Candidate Ceiling ($M_{\max}$):** Hard input threshold **$M_{\max} = 2$** ($m_{i,t} \le 2$). Retention rate = **99.6771%**.
- **Target Horizon:** Forward 5-session log return $R_{i,t}^{(5)} = \log(P_{i,t+5}/P_{i,t})$ percentile-ranked within $T_t$.
- **Cross-Sectional Target Population ($T_t$):** $T_t = \{i \in U_t : C^{\text{input}}_{i,t} = 1 \land C^{\text{target}}_{i,t} = 1\}$. Future index membership at $t+5$ is NOT used.

---

## 2. Complete Mathematical Pipeline

$$
U_t \xrightarrow{\text{Constituent}} W_t \xrightarrow{\text{60-Session Window}} C^{\text{input}}_{i,t} \xrightarrow{\text{Input Candidate}} C^{\text{target}}_{i,t} \xrightarrow{\text{Target Candidate}} T_t \xrightarrow{\text{Cross-Section}} C^{\text{sample}}_{i,t} \xrightarrow{\text{Sample}} R_{i,t}^{(5)} \xrightarrow{\text{5d Return}} p_{i,t}
$$

### Formal Mathematical Definitions

1. **Global Trading Session Axis ($D$):** $D = \{d_0, d_1, \dots, d_T\}$ representing US market calendar sessions.
2. **Input Window ($W_t$):** $W_t = \{d_{t-59}, \dots, d_t\}$ for prediction date $t$.
3. **Raw Observation Availability ($O_{i,d} \in \{0, 1\}$):** $O_{i,d} = 1 \iff$ valid market price recorded at date $d$.
4. **Input Candidate ($C^{\text{input}}_{i,t} \in \{0, 1\}$):**
$$C^{\text{input}}_{i,t} = 1 \iff (i \in U_t) \land (O_{i,t} = 1) \land (d_{t-59} \ge d_{i, \text{first\_obs}}) \land (d_t \le d_{i, \text{last\_obs}}) \land (m_{i,t} \le 2)$$

5. **Target Candidate ($C^{\text{target}}_{i,t} \in \{0, 1\}$):**
$$C^{\text{target}}_{i,t} = 1 \iff (O_{i,t+5} = 1) \land (P_{i,t+5} > 0) \land (P_{i,t} > 0)$$

6. **Cross-Sectional Target Set ($T_t$):** $T_t = \{i \in U_t : C^{\text{input}}_{i,t} = 1 \land C^{\text{target}}_{i,t} = 1\}$.
7. **Final Supervised Sample Flag ($C^{\text{sample}}_{i,t} \in \{0, 1\}$):** $C^{\text{sample}}_{i,t} = C^{\text{input}}_{i,t} \land C^{\text{target}}_{i,t}$.
8. **Target Percentile Label ($p_{i,t} \in [0, 1]$):** Normalized percentile rank of $R_{i,t}^{(5)}$ across $i \in T_t$.

---

## 3. Reconciled Target Population Partition ($N = 6,573,171$)

| Target Status Category | Operational Definition | Count | Percentage |
|-----------------------|------------------------|------:|-----------:|
| **`TARGET_VALID`** | $O_{i,t+5} = 1$ and valid return computed | 6,561,869 | 99.8280% |
| **`TARGET_UNAVAILABLE_DISAPPEARED`** | Delisted or gapped before $t+5$ | 6,782 | 0.1032% |
| **`TARGET_DATASET_END_EDGE`** | $t+5 \ge T$ (last 5 dates of dataset) | 4,520 | 0.0688% |
| **`TARGET_INVALID_PRICE`** | $P \le 0$ or null price at $t$ or $t+5$ | 0 | 0.0000% |
| **TOTAL** | **All Mutually Exclusive Target Categories** | **6,573,171** | **100.0000%** |

```text
AUTOMATED ASSERTION: N(input_candidates) == sum(target_status_categories) [PASSED]
```

---

## 4. Observation Mask Semantics & Feature Independence

Define the raw observation mask tensor $Q_{i,t} \in \{0, 1\}^{60}$:
$$Q_{i,t,j} = O_{i,d_{t-59+j}} \quad \text{for } j = 0, \dots, 59$$

### Strict Mask Rules (LOCKED):
1. **$Q_{i,t,j} = 1$** indicates genuine observed market OHLCV on session $d_{t-59+j}$.
2. **$Q_{i,t,j} = 0$** indicates unobserved/missing session.
3. **Feature Imputation Rule:** Downstream feature transformations (forward fill, EMA, rolling std) may impute numerical feature values, but **MUST NEVER MODIFY $Q_{i,t,j}$**.
4. **Volume Rule:** Missing raw volume ($O=0$) is **NOT** zero volume ($V=0$). Imputation policies are defined independently per feature family during feature engineering.

---

## 5. Machine-Checkable Validation Assertions

The dataset construction pipeline enforces 8 machine-checkable assertion families:

```python
# 1. Window Length Assertion
assert len(W_t) == 60 and W_t[-1] == prediction_date

# 2. Prediction Date Observation Assertion
assert C_input == 1 -> O(i, t) == 1

# 3. Missingness Ceiling Assertion
assert C_input == 1 -> missing_count <= 2

# 4. Target Validation Assertion
assert C_sample == 1 -> (C_input == 1 and C_target == 1)

# 5. Cross-Sectional Membership Assertion
assert target_rank is not None -> i in T_t

# 6. Ranking Bounds Assertion
assert 0.0 <= percentile_rank <= 1.0

# 7. Leakage Prevention Assertion
assert feature_dates <= t and target_dates >= t+1

# 8. Target Partition Exhaustiveness Assertion
assert total_input_candidates == sum(target_partition_counts.values())
```

---

## 6. Locked Decision Status Matrix

| Pipeline Component | Status | Locked Specification |
|--------------------|--------|----------------------|
| **60-Session Window ($W_t$)** | `LOCKED` | $W_t = [d_{t-59} \dots d_t]$ on US trading calendar. No sequence compression. |
| **PiT Membership ($U_t$)** | `LOCKED` | Annual Russell 1000 constituent snapshot. Missing data does not alter membership. |
| **Observation Mask ($Q_{i,t}$)** | `LOCKED` | $Q_{i,t,j} = O_{i,d_{t-59+j}}$. Imputation never modifies $Q$. |
| **Input Candidate Ceiling ($M_{\max}$)** | `LOCKED` | $M_{\max} = 2$ ($m_{i,t} \le 2$). Retention = 99.6771%. |
| **Cross-Sectional Target Set ($T_t$)** | `LOCKED` | $T_t = \{i \in U_t : C^{\text{input}}_{i,t}=1 \land C^{\text{target}}_{i,t}=1\}$. Ex-post index membership not used. |
| **Target Variable ($p_{i,t}$)** | `LOCKED` | Normalized percentile rank of 5-session forward log return $R_{i,t}^{(5)}$ within $T_t$. |
| **Target Reconciliation** | `LOCKED` | 6,561,869 Valid + 6,782 Disappeared + 4,520 Dataset Edge = 6,573,171 Candidates. |
| **US Trading Calendar** | `LOCKED` | 6,598 valid US trading sessions (2000-06-30 to 2026-09-25). |
| **Imputation & Feature Pipeline** | `READY` | Feature-specific rules to be implemented in feature engineering phase. |

