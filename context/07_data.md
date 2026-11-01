# 07 · Data collection and universe

**Status:** data up to 2024-12-31 done and verified; 2025–2026 partly processed. **Owner:** Hào

## Why not the full market

In theory the universe should contain every stock that ever existed. That is not done, for two reasons:
- **Data limits:** collecting the full history of every stock that ever traded is very hard and expensive.
- **Most stocks are noise:** among the very large number of listed securities, only some reflect the real economy and trader behaviour. Most of the rest are noise or negligible.

## How the universe is built

1. Each year, take the **Russell 1000** constituents of that year.
2. Union the yearly lists into one **master list** of every stock that was ever a member.
3. Collect the **full lifetime history** of every stock in the master list, including the years before it joined and after it left the index.

Why this works:
- **No survivorship bias:** stocks that were later acquired, went bankrupt or were delisted stay in the data.
- **Far fewer stocks** than the whole market, while still capturing its main characteristics.

This universe is treated as an **approximation of the true universe**.

## Current state

| Item | State |
|---|---|
| Russell 1000 lists | 2000–2026, one per year. **The 2003 file is an exact copy of 2002**, so a real 2003 list is missing |
| Master list | 2,931 securities (CRSP PERMNO) |
| Price history to 2024-12-31 | CRSP via WRDS, full lifetime, 18.7M daily rows; delisting returns included; integrity checks passed |
| 2025–2026 | Prices from yfinance / massive / tiingo, cleaned and mapped; delisting causes looked up by hand for 86 stocks. **Not yet joined with CRSP** and not yet in the warehouse silver layer |
| Warehouse | `../../warehouse/`: bronze (raw) + silver (`security`, `security_history`, `prices_daily`, `delisting`, `russell_constituent`, view `v_prices`). One shareable DuckDB file |

## Known limitations

- **Membership by year is approximate.** Lists are yearly snapshots, and the snapshot date varies (2023 looks like a late-2023 snapshot). Reconstitution happens at the end of June, twice a year from 2026.
- 116 rows of the Russell lists have no matched security: PDF junk, new 2025–26 listings without a CRSP ID, the 2003 copy, and a few stocks missing from CRSP.
- **CRSP data is licensed:** share privately with team members who have WRDS access only, and never commit it to git.

Details: `../../warehouse/SPEC.md` and `../../pipeline/README.md`.
