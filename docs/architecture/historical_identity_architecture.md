# Historical Security Identity Resolution Architecture

## 1. Architectural Principles

In historical quantitative research, equity tickers are volatile pointers rather than immutable identities. The same ticker string frequently refers to different corporate entities at different points in time (ticker reuse), while corporate name changes and share class variations introduce false identity fragmentation.

The Historical Security Identity Resolver implements a strictly non-mutating, point-in-time evidence hierarchy:

```text
Ticker Spell (ticker, start_date, end_date)
    ↓
Representative Date Strategy (Trading Session Midpoint + Boundary Fallback)
    ↓
Massive Point-in-Time Reference API (Primary)
    ↓
SEC EDGAR CIK & Entity Token Corroboration (Secondary)
    ↓
OpenFIGI Temporal Verification (Fallback Guard)
    ↓
5-Tier Canonical Identity Decision
    ↓
Common-Stock Research Universe Classification
```

---

## 2. The 5-Tier Canonical Identity Hierarchy

Every ticker spell is resolved deterministically into one of five distinct tiers:

| Tier | Name | Evidence Requirements | `is_canonical` | Security ID Format |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Authoritative FIGI** | Point-in-time Share-Class FIGI returned directly by Massive for `(ticker, rep_date)`. | `True` | `BBG00...` (12-char FIGI) |
| **Tier 2** | **Corroborated CIK Match** | Massive CIK corroboration via SEC EDGAR company list matching distinctive name tokens. | `True` | `BBG00...` (Confirmed FIGI) |
| **Tier 3** | **Provisional Issuer Scope** | Confirmed CIK without a confirmed share-class FIGI. Non-canonical namespace. | `False` | `PROVISIONAL_CIK_<CIK>_<TICKER>_<HASH>` |
| **Tier 4** | **Verified Massive Empty** | Massive PIT API returned an empty active instrument set for `(ticker, rep_date)`. | `False` | `UNRESOLVED_<TICKER>_<SEQ>_<HASH>` |
| **Tier 5** | **Offline Pending** | Backfill pending execution or offline queue item. | `False` | `UNRESOLVED_<TICKER>_<SEQ>_<HASH>` |

### Non-Negotiable Invariants:
1. **Zero False Identity Merges**: Spells belonging to different corporate entities sharing the same ticker never receive the same `security_id`.
2. **Provisional Scope Isolation**: A `PROVISIONAL_CIK` identifier is scoped to `(CIK, ticker, start_date)` to prevent collapsing multi-class or multi-spell securities into a single issuer ID.
3. **Deterministic Unresolved IDs**: Every unresolved spell receives a distinct, hash-stabilized identifier.
4. **Unknown Security Exclusions**: Instruments with `UNKNOWN` or unverified security type are never promoted to `research_universe_status = INCLUDE`.

---

## 3. Representative Date Selection & Drift Detection

For each spell $[t_{\text{start}}, t_{\text{end}}]$ across contiguous trading sessions:

1. **Level 1 (Midpoint)**: The midpoint trading session $t_{\text{mid}} = \text{session}[(i_{\text{start}} + i_{\text{end}}) // 2]$ is queried first.
2. **Level 2 (Boundary Fallback)**: If $t_{\text{mid}}$ yields empty or weak evidence, boundary sessions $t_{\text{start}}$ and $t_{\text{end}}$ are checked to recover valid historical metadata.
3. **Level 3 (Identity Drift Detection)**: If boundary points return conflicting CIKs or FIGIs, the spell is flagged with `drift_detected = True` for interval partitioning.
