# Delisting & Target Audit Report

> **Audit Date**: 2026-10-03  
> **Data Period**: 2000-06-30 to 2026-09-25

---

## 1. CRSP Era Delisting Summary (2000-2024)

### 1.1 DLSTCD Distribution

Total delisting events: **1,936** across 2000-2024.

| DLSTCD Category | Description | Count | DLRET Availability |
|----------------|-------------|------:|:-:|
| 200, 231, 233, 241, 244, 261 | Mergers / Acquisitions / Exchange offers | ~1,500 | ~90% |
| 500, 520, 560, 570, 574, 580, 584 | Dropped by exchange / Bankruptcy / Failure | ~100 | ~60% |
| 100 | Still active (end-of-dataset marker) | 1,022 (all in 2024) | 0% (DLRET="A") |
| Other | Miscellaneous | ~300 | Varies |

### 1.2 DLRET Availability

- **905 (46.7%)** have numeric DLRET
- **1,031 (53.3%)** have missing or string-coded DLRET
- String codes observed: "A" (active, 1,022 in 2024), "T" (SIVB), "S" (MRTX), "B", "C"

> [!IMPORTANT]
> **The 1,022 DLSTCD=100 entries in 2024 are NOT real delistings.** They are end-of-dataset markers for stocks that are still actively trading. Filter these out completely.

### 1.3 Notable Delisting Events

| Year | Ticker | DLSTCD | DLRET | Event |
|------|--------|--------|-------|-------|
| 2001 | ENE (Enron) | 574 | -0.997 | Bankruptcy |
| 2002 | WCOME (WorldCom) | 574 | -0.449 | Bankruptcy |
| 2008 | GM | 574 | -0.187 | Bankruptcy (pre-restructuring) |
| 2009 | JAVA (Sun) | 233 | +0.0 | Acquired by Oracle |
| 2013 | DELL | 233 | +0.003 | Going private (MBO) |
| 2022 | SIVB | 584 | "T" | Bank failure (string-coded!) |
| 2022 | FRC | 584 | -0.905 | Bank failure |
| 2022 | TWTR | 200 | — | Acquired by Musk |
| 2023 | SPLK | 233 | — | Acquired by Cisco |

### 1.4 Ticker Changes (Not Delistings)

- **324 PERMNOs** had multiple tickers across their lifetime (name changes, not departures)
- **184 rows** with `NWPERM` field populated (successor PERMNO after merger)
- Examples: SUNW → JAVA, AT → merged into SBC → T

> [!NOTE]
> A PERMNO with a ticker change is NOT the same as a delisting. PERMNO is the permanent identifier that persists through name changes. Ticker-based matching (e.g., Yahoo era) cannot distinguish ticker changes from delistings.

---

## 2. Yahoo Era Disappearances (2025-2026)

### 2.1 No Delisting Information Available

The Yahoo-sourced data files (2025-2026) contain **NONE** of the following fields:
- DLSTCD (delisting code)
- DLRET (delisting return)
- DLRETX (delisting price return)
- DLPRC (delisting price)
- NWPERM (successor identifier)

### 2.2 Disappeared Tickers

| Year | Tickers Disappearing >7d Before Last Date | Examples |
|------|------------------------------------------:|---------|
| 2025 | 54 | AAP, AMED, ANSS, AZEK, AZPN, ... |
| 2026 | 66 | ACHC, AL, APLS, ASH, AVB, BK, ... |

**For NONE of these 120 disappearances can we determine:**
- Whether the stock was acquired (positive terminal value)
- Whether the stock was delisted for cause (likely loss)
- Whether the ticker changed (benign)
- Whether it's simply a data gap

### 2.3 Risk Assessment

This creates a **survivorship bias risk** in the 2025-2026 target return computation:
- If a stock disappears between $t$ and $t+5$, the target $Y_{i,t}$ is undefined
- Without DLRET, we cannot assign a terminal return
- Last observed price may or may not be realizable

---

## 3. Terminal Price Quality Assessment

### 3.1 CRSP Era

For most CRSP delistings (mergers/acquisitions):
- Last-day volume is substantial (active trading)
- Max calendar gap before disappearance: typically 3-4 days (weekends)
- Last PRC is a valid, economically realizable price
- DLPRC (explicit delisting price) is available for some but not all events

### 3.2 Assessment by Delisting Type

| DLSTCD Category | Last Price Quality | Recommended Fallback |
|----------------|-------------------|---------------------|
| 200, 231, 233, 241 (M&A) | Good — active trading at or near deal price | Use last PRC; assign DLRET=0 if missing |
| 574, 584 (Bankruptcy) | Poor — may be suspended, halted, or penny-stock | Use DLRET if available; assign -30% if missing |
| 500, 520 (Dropped) | Variable | Use DLRET if available; investigate individually |
| 100 (Still active) | N/A — not a real delisting | Filter out entirely |

### 3.3 Verdict

```
LAST_PRICE_FALLBACK = CONDITIONALLY_ACCEPTABLE
```

Acceptable for M&A delistings (majority of cases). NOT acceptable for bankruptcies without DLRET. UNRESOLVED for Yahoo era.

---

## 4. Target Return Computation

### 4.1 Coverage

| Year | Total Obs | Has t+5 | Coverage | Unexpected Missing |
|------|----------|---------|---------|-------------------|
| 2005 | 258,830 | 253,244 | 97.8% | 308 |
| 2010 | 256,518 | 250,754 | 97.8% | 626 |
| 2015 | 266,002 | 260,288 | 97.9% | 349 |
| 2020 | 259,028 | 253,567 | 97.9% | 276 |
| 2024 | 131,758 | 126,578 | 96.1% | 10 |
| 2025 | 249,773 | 244,518 | 97.9% | 0 |

### 4.2 Missing Target Categories

1. **Tail effect** (~2%): Last 5 sessions per stock per file — expected and benign
2. **Delisting** (~0.1-0.2%/year): Stock disappears before t+5 — requires delisting handling
3. **Data gap** (<0.01%): Stock temporarily missing — rare, investigate individually

### 4.3 Recommended Delisting-Aware Target

For CRSP era, when stock $i$ disappears between $t$ and $t+5$:

$$
Y_{i,t} = \log\left(\prod_{s=1}^{k}(1+RETX_{t+s}) \cdot (1+DLRETX)\right)
$$

where $k < 5$ is the last available observation and $DLRETX$ is the delisting return. If DLRETX is missing:
- M&A (DLSTCD ∈ {200,231,233,241}): $DLRETX = 0$
- Adverse (DLSTCD ∈ {500,574,584}): $DLRETX = -0.30$
- Unknown: $Y_{i,t} = \text{NaN}$ (exclude from training)

For Yahoo era: $Y_{i,t} = \text{NaN}$ for disappearing stocks (no delisting return available).

---

## 5. Unresolved Cases

1. **120 Yahoo-era disappearances** (2025-2026): No delisting data → cannot determine terminal return
2. **CRSP string-coded DLRET** (T, S, A, B, C): ~1,031 cases → treat as NaN
3. **Ticker-change vs delisting ambiguity** in Yahoo era: Ticker "A" in 2024 might become ticker "XYZ" in 2025 (we would not know)
4. **Cross-year boundary delistings**: A stock in WRDS 2023.parquet (ending June 2024) that delists in March 2024 appears in that file. But the t+5 target might need data from WRDS 2024.parquet (starting July 2024) — ensure consolidation.

---

## 6. Recommended Delisting Policy

```python
def get_delisting_return(row):
    """CRSP era only. Yahoo era returns NaN."""
    dlstcd = row.get('DLSTCD')
    dlret = pd.to_numeric(row.get('DLRET'), errors='coerce')
    
    if pd.isna(dlstcd) or dlstcd == 100:
        return None  # Not a delisting
    
    if pd.notna(dlret):
        return dlret  # Use actual DLRET
    
    # Missing DLRET — assign conservative estimate
    if dlstcd in [200, 231, 233, 241, 244]:
        return 0.0  # M&A: assume deal price ≈ last price
    elif dlstcd in [500, 520, 560, 570, 574, 580, 584]:
        return -0.30  # Adverse: conservative penalty
    else:
        return np.nan  # Unknown: exclude
```
