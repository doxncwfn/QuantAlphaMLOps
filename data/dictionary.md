# WRDS / CRSP Daily Stock Data Dictionary

> **Dataset**: CRSP US Daily Stock Database (Wharton Research Data Services / WRDS)  
> **Target Universe**: Russell 1000 Historical Constituents (`/data/Russell 1000/WRDS/`)  
> **Primary Key**: `(PERMNO, date)`  
> **Total Variables**: 63  
> **Formats Available**: CSV (`.csv`), Apache Parquet (`.parquet`)

---

## 1. Quick Data Conventions & Critical Notes

Before analyzing or modeling this dataset, take note of the following standard CRSP conventions:

1. **Negative Prices (`PRC`, `OPENPRC`, `BIDLO`, `ASKHI`, `DLPRC`)**:
   - A **negative sign** in a price field indicates that **no closing trade occurred** on that trading day.
   - The absolute value $| \text{PRC} |$ represents the **bid/ask midpoint average**.
   - A value of `0.0` indicates that neither a closing trade price nor a bid/ask quote was available.
2. **Special Return Codes (`RET`, `RETX`, `DLRET`, `DLRETX`)**:
   - While usually numeric returns, CRSP uses specific character codes when trading conditions prevent normal calculation:
     - `'B'`: No valid price or quote available on day $t-1$.
     - `'C'`: No valid price or quote available on day $t$.
     - `'T'`: Trading halted by exchange.
     - `'S'`: Trading suspended.
     - `'A'`: Not traded.
   - Numeric missing values are coded as negative values: `-55.0`, `-66.0`, `-77.0`, `-88.0`, `-99.0` (see variable descriptions for meanings).
3. **Adjustment Factors (`FACPR`, `FACSHR`, `CFACPR`, `CFACSHR`)**:
   - To compute split-adjusted prices historically: $\text{Price}_{\text{adj}}(t) = \text{PRC}(t) / \text{CFACPR}(t)$.
   - To compute split-adjusted shares: $\text{Shares}_{\text{adj}}(t) = \text{SHROUT}(t) \times \text{CFACSHR}(t)$.
4. **Volume Units (`VOL`)**:
   - Reported in individual single shares for daily data. Nasdaq reports exact shares traded; NYSE/AMEX data sources historically rounded volumes to the nearest 100 shares.
5. **Shares Outstanding (`SHROUT`)**:
   - Recorded in **thousands** of shares. Multiply by $1,000$ for the actual share count.

---

## 2. Master Variable Summary Table

| Column                                                                | Type    | Category         | Brief Description                                        |
| :-------------------------------------------------------------------- | :------ | :--------------- | :------------------------------------------------------- |
| [`PERMNO`](#permno--crsp-permanent-issue-identifier)                  | Integer | Identifier       | CRSP Permanent Security Issue Number (Primary Key)       |
| [`date`](#date--trading-date)                                         | Date    | Date             | Trading Date (`YYYYMMDD` or `YYYY-MM-DD`)                |
| [`COMNAM`](#comnam--company-name)                                     | Char    | Identifier       | Company Name (32-character maximum description)          |
| [`TICKER`](#ticker--exchange-ticker-symbol)                           | Char    | Identifier       | Exchange Ticker Symbol                                   |
| [`TSYMBOL`](#tsymbol--trading-ticker-symbol)                          | Char    | Identifier       | Exchange Trading Symbol (includes share suffixes)        |
| [`CUSIP`](#cusip--cusip-identifier)                                   | Char    | Identifier       | Latest 8-character CUSIP identifier                      |
| [`NCUSIP`](#ncusip--historical-cusip-identifier)                      | Char    | Identifier       | Historical CUSIP identifier for the observation          |
| [`PERMCO`](#permco--crsp-permanent-company-number)                    | Float   | Identifier       | CRSP Permanent Company Number                            |
| [`ISSUNO`](#issuno--nasdaq-issue-number)                              | Float   | Identifier       | Nasdaq Issue Number assigned by NASD                     |
| [`SHRCLS`](#shrcls--share-class)                                      | Char    | Identifier       | Share Class designation (e.g. "A" for Class A)           |
| [`NAMEENDT`](#nameendt--last-date-of-name)                            | Date    | Date             | End Date of security's name history structure            |
| [`SHRENDDT`](#shrenddt--shares-outstanding-observation-end-date)      | Date    | Date             | Shares Observation End Date                              |
| [`NEXTDT`](#nextdt--date-of-next-available-information)               | Date    | Date             | Date of next available price information                 |
| [`DCLRDT`](#dclrdt--declaration-date)                                 | Date    | Date             | Distribution Declaration Date                            |
| [`RCRDDT`](#rcrddt--record-date)                                      | Date    | Date             | Distribution Record Date                                 |
| [`PAYDT`](#paydt--payment-date)                                       | Date    | Date             | Distribution Payment Date                                |
| [`DLPDT`](#dlpdt--date-of-delisting-payment)                          | Date    | Date             | Date of Delisting Payment                                |
| [`SHRCD`](#shrcd--share-code)                                         | Float   | Classification   | 2-digit Share Code describing security & company type    |
| [`SHRFLG`](#shrflg--share-flag)                                       | Float   | Classification   | Share source flag (0 = regular source, 1 = distribution) |
| [`SHROUT`](#shrout--number-of-shares-outstanding)                     | Float   | Shares           | Number of publicly held shares (in thousands)            |
| [`SICCD`](#siccd--standard-industrial-classification-code)            | Float   | Industry         | Historical Standard Industrial Classification (SIC) code |
| [`HSICCD`](#hsiccd--header-standard-industrial-classification-code)   | Float   | Industry         | Header (latest non-zero) SIC code                        |
| [`HSICMG`](#hsicmg--header-sic-major-group)                           | Float   | Industry         | First 2 digits of `HSICCD` (Major Group)                 |
| [`HSICIG`](#hsicig--header-sic-industry-group)                        | Float   | Industry         | First 3 digits of `HSICCD` (Industry Group)              |
| [`NAICS`](#naics--north-american-industry-classification-system-code) | Char    | Industry         | 6-digit North American Industry Classification code      |
| [`EXCHCD`](#exchcd--exchange-code)                                    | Float   | Exchange         | Listing Exchange Code                                    |
| [`HEXCD`](#hexcd--header-exchange-code)                               | Float   | Exchange         | Header (latest valid) Exchange Code (1, 2, or 3)         |
| [`PRIMEXCH`](#primexch--primary-exchange)                             | Char    | Exchange         | Primary trading exchange single-letter code              |
| [`TRDSTAT`](#trdstat--trading-status)                                 | Char    | Status           | Trading Status code (`A`ctive, `H`alted, `S`uspended)    |
| [`SECSTAT`](#secstat--security-status)                                | Char    | Status           | Security Status code (`R`egular, `W`hen-issued, etc.)    |
| [`TRTSCD`](#trtscd--traits-code)                                      | Float   | Status           | 1-digit code describing trading status of issue          |
| [`NMSIND`](#nmsind--nasdaq-national-market-indicator)                 | Float   | Status           | Nasdaq National Market tier classification code          |
| [`MMCNT`](#mmcnt--market-maker-count)                                 | Float   | Trading          | Count of registered market makers for the issue          |
| [`NSDINX`](#nsdinx--nasd-index)                                       | Float   | Classification   | NASD internal business description category code         |
| [`PRC`](#prc--closing-price-or-bidask-average)                        | Decimal | Price            | Closing price (positive) or Bid/Ask average (negative)   |
| [`OPENPRC`](#openprc--price-open)                                     | Decimal | Price            | Open price (first trade price after market open)         |
| [`ASKHI`](#askhi--ask-or-high-price)                                  | Decimal | Price            | High trade price or closing ask price                    |
| [`BIDLO`](#bidlo--bid-or-low-price)                                   | Decimal | Price            | Low trade price or closing bid price                     |
| [`BID`](#bid--closing-bid)                                            | Decimal | Price            | Closing bid price quotation                              |
| [`ASK`](#ask--closing-ask)                                            | Decimal | Price            | Closing ask price quotation                              |
| [`VOL`](#vol--share-volume)                                           | Decimal | Volume           | Total number of shares traded on that date               |
| [`NUMTRD`](#numtrd--number-of-trades)                                 | Integer | Volume           | Number of trades recorded on Nasdaq                      |
| [`RET`](#ret--holding-period-return)                                  | Decimal | Return           | Daily holding period return (with dividends)             |
| [`RETX`](#retx--holding-period-return-without-dividends)              | Decimal | Return           | Daily holding period return (without dividends)          |
| [`DLSTCD`](#dlstcd--delisting-code)                                   | Float   | Delisting        | 3-digit code providing reason for delisting              |
| [`DLPRC`](#dlprc--delisting-price)                                    | Float   | Delisting        | Delisting trade price or bid/ask quote                   |
| [`DLAMT`](#dlamt--amount-after-delisting)                             | Float   | Delisting        | Ending value/payments used in delisting return           |
| [`DLRET`](#dlret--delisting-return)                                   | Float   | Delisting        | Delisting return after security ceases trading           |
| [`DLRETX`](#dlretx--delisting-return-without-dividends)               | Float   | Delisting        | Delisting return excluding ordinary dividends            |
| [`NWPERM`](#nwperm--new-crsp-permanent-number)                        | Float   | Delisting        | Acquiring company PERMNO following merger/exchange       |
| [`DISTCD`](#distcd--distribution-code)                                | Float   | Corporate Action | 4-digit code describing type & method of distribution    |
| [`DIVAMT`](#divamt--dividend-cash-amount)                             | Float   | Corporate Action | Cash dividend or distribution dollar amount per share    |
| [`FACPR`](#facpr--factor-to-adjust-price)                             | Float   | Corporate Action | Factor to adjust price for distributions & splits        |
| [`FACSHR`](#facshr--factor-to-adjust-shares-outstanding)              | Float   | Corporate Action | Factor to adjust shares for distributions & splits       |
| [`CFACPR`](#cfacpr--cumulative-factor-to-adjust-price)                | Float   | Corporate Action | Cumulative factor to adjust historical price             |
| [`CFACSHR`](#cfacshr--cumulative-factor-to-adjust-shares-outstanding) | Float   | Corporate Action | Cumulative factor to adjust historical shares            |
| [`ACPERM`](#acperm--acquiring-permno)                                 | Float   | Corporate Action | PERMNO of acquiring security linked to distribution      |
| [`ACCOMP`](#accomp--acquiring-permco)                                 | Float   | Corporate Action | PERMCO of acquiring company linked to distribution       |
| [`vwretd`](#vwretd--value-weighted-return-with-dividends)             | Float   | Index Return     | Value-weighted market return (including dividends)       |
| [`vwretx`](#vwretx--value-weighted-return-without-dividends)          | Float   | Index Return     | Value-weighted market return (excluding dividends)       |
| [`ewretd`](#ewretd--equal-weighted-return-with-dividends)             | Float   | Index Return     | Equal-weighted market return (including dividends)       |
| [`ewretx`](#ewretx--equal-weighted-return-without-dividends)          | Float   | Index Return     | Equal-weighted market return (excluding dividends)       |
| [`sprtrn`](#sprtrn--return-on-the-sp-composite-index)                 | Float   | Index Return     | Daily return on the S&P 500 Composite Index              |

---

## 3. Detailed Variable Specifications

### Security & Company Identifiers

#### `PERMNO` — CRSP Permanent Issue Identifier

- **Variable Name**: `PERMNO`
- **Type**: Integer
- **Description**: A unique permanent five-digit identifier assigned by CRSP to each listed security issue. Unlike tickers, CUSIPs, or company names, `PERMNO` never changes during the security's trading history, nor is it ever reassigned to another security after delisting. It is the primary issue-level key for survivorship-bias-free tracking.

#### `PERMCO` — CRSP Permanent Company Number

- **Variable Name**: `PERMCO`
- **Type**: Float / Integer
- **Description**: A unique permanent company-level identifier assigned by CRSP to all companies issuing securities on the CRSP file. Permanent for all securities issued by the firm regardless of corporate name changes or reorganizations.
- **Coding Rules**:
  - `PERMCO < 20,000`: Inherited from the Nasdaq-assigned Company Number when the firm had an issue trading on The Nasdaq Stock Market.
  - `PERMCO >= 20,000`: Assigned directly by CRSP.

#### `COMNAM` — Company Name

- **Variable Name**: `COMNAM`
- **Type**: Char (32 characters max)
- **Description**: Company name description allocated by CRSP. Preference is given to spellings and abbreviations in Standard & Poor's CUSIP Directory. When description exceeds 32 characters, standardized CRSP abbreviations are used.

#### `TICKER` — Exchange Ticker Symbol

- **Variable Name**: `TICKER`
- **Type**: Char
- **Description**: Common exchange ticker symbol. The combination of ticker, exchange, and date uniquely identifies a security. Tickers are 1 to 3 characters for NYSE and AMEX securities, or 4 to 5 characters for Nasdaq securities.
- **Nasdaq 5th Character Suffixes**:
  Nasdaq trading tickers have 4 base characters and an optional 5th character suffix indicating issue type or temporary status:

  | Suffix | Definition                            |
  | :----- | :------------------------------------ |
  | `A`    | Class A                               |
  | `B`    | Class B                               |
  | `F`    | Companies incorporated outside the US |
  | `S`    | Shares of Beneficial Interest         |
  | `U`    | Unit                                  |
  | `V`    | When-issued                           |
  | `Y`    | American Depository Receipt (ADR)     |
  | `Z`    | Miscellaneous common issues           |

> [!NOTE]
> When Nasdaq adds two suffixes to a 4-letter base ticker, the 4th letter is dropped to maintain the 5-character limit (e.g. `ABCD` + `A` + `F` becomes `ABCAF`). Ticker fields may be blank for Nasdaq securities that stopped trading prior to the mid-1980s, and for NYSE securities prior to July 1962.

#### `TSYMBOL` — Trading Ticker Symbol

- **Variable Name**: `TSYMBOL`
- **Type**: Char
- **Description**: The official trading symbol listed by exchanges and consolidated quote systems. Includes all temporary values, share classes, and share type suffixes without punctuation (no periods). Data available from 2002-01-02 onwards for NYSE/AMEX, and from 1982-11-01 onwards for Nasdaq.

#### `CUSIP` — CUSIP Identifier

- **Variable Name**: `CUSIP`
- **Type**: Char (8 characters)
- **Description**: The latest 8-character Committee on Uniform Security Identification Procedures (CUSIP) identifier for the security through the end of the file. Assigned by the CUSIP Service Bureau (Standard & Poor's / American Bankers Association).
  - Characters 1–6 identify the corporate issuer (alphabetical sequence).
  - Characters 7–8 identify the specific issue.
  - Securities domiciled outside the US/Canada use CINS (CUSIP International Numbering System) where the first character is an alpha country code.
- **CRSP Dummy CUSIPs**:
  - `***99*9*`: Dummy issuer number (chars 1–6) and dummy issue number (chars 7–8) assigned by CRSP when no official CUSIP exists.
  - `******9*`: Real issuer number with a dummy issue number assigned by CRSP.

#### `NCUSIP` — Historical CUSIP Identifier

- **Variable Name**: `NCUSIP`
- **Type**: Char (8 characters)
- **Description**: The historical CUSIP identifier of the security at the specific time of observation. While `CUSIP` records the latest identifier, `NCUSIP` tracks changes across corporate renamings, reorganizations, and capital restructurings. Blank if the structure predates the CUSIP Bureau.

#### `ISSUNO` — Nasdaq Issue Number

- **Variable Name**: `ISSUNO`
- **Type**: Float / Integer
- **Description**: Unique integer assigned by the National Association of Securities Dealers (NASD) to each listed security on the Nasdaq Stock Market. Differentiates multiple securities issued by the same firm. Set to `0` if unknown. If an NYSE/AMEX security previously traded on Nasdaq, this holds the latest Nasdaq issue number.

#### `SHRCLS` — Share Class

- **Variable Name**: `SHRCLS`
- **Type**: Char
- **Description**: Share class designation (e.g. `"A"` for Class A common stock). Left-justified, padded with spaces, and generally blank for companies with a single class of common stock.

---

### Dates & Timing

#### `date` — Trading Date

- **Variable Name**: `date`
- **Type**: Date / Integer (`YYYY-MM-DD` or `YYYYMMDD`)
- **Description**: The date on which the stock market transaction or quote took place. Forms the composite primary key alongside `PERMNO`.

#### `NAMEENDT` — Last Date of Name

- **Variable Name**: `NAMEENDT`
- **Type**: Date (`YYYYMMDD`)
- **Description**: The last effective date of a security's name history record. Set to the day preceding the `Name Effective Date` of the next name record, the end of the stock data series, or the delisting date of the last name record.

#### `SHRENDDT` — Shares Outstanding Observation End Date

- **Variable Name**: `SHRENDDT`
- **Type**: Date (`YYYYMMDD`)
- **Description**: The last effective date for a given shares outstanding (`SHROUT`) observation. Set to the latest date prior to the observation date of the next record, or the delisting date for the final record.

#### `NEXTDT` — Date of Next Available Information

- **Variable Name**: `NEXTDT`
- **Type**: Date (`YYYYMMDD`)
- **Description**: The date of a security's post-delisting price (`DLPRC`). Set to `0` if the security is still actively trading, if final value is determined by a distribution, or if the value is unknown after suspension/delisting. Set to one trading day after delist date if the security was determined worthless.

#### `DCLRDT` — Declaration Date

- **Variable Name**: `DCLRDT`
- **Type**: Date (`YYYYMMDD`)
- **Description**: The date on which the board of directors declared a dividend or distribution. Set to `0` if declaration date cannot be found.

#### `RCRDDT` — Record Date

- **Variable Name**: `RCRDDT`
- **Type**: Date (`YYYYMMDD`)
- **Description**: The date by which a shareholder must be officially registered on the company's stock transfer books to be eligible for a dividend or distribution. For mergers/liquidations where the firm ceased to exist, set equal to the date of last price.

#### `PAYDT` — Payment Date

- **Variable Name**: `PAYDT`
- **Type**: Date (`YYYYMMDD`)
- **Description**: The date on which distribution checks are mailed or payments made. Set to `0` if unavailable. For mergers/liquidations where the firm ceased to exist, set equal to the date of last price.

#### `DLPDT` — Date of Delisting Payment

- **Variable Name**: `DLPDT`
- **Type**: Date (`YYYYMMDD`)
- **Description**: The effective date of any distribution or payment after delisting used in calculating delisting returns. Set to `0` if no post-delisting payments occurred.

---

### Share Classification & Industry Coding

#### `SHRCD` — Share Code

- **Variable Name**: `SHRCD`
- **Type**: Float / Integer (2-digit code)
- **Description**: Describes the security type and company domicile structure:
  - **1st Digit (Security Type)**:
    | Code | Definition |
    | :--- | :--- |
    | `1` | Ordinary Common Shares |
    | `2` | Certificates |
    | `3` | ADRs (American Depository Receipts) |
    | `4` | SBIs (Shares of Beneficial Interest) |
    | `7` | Units (Depository Units, LP Units, etc.) |
  - **2nd Digit (Detailed Characteristics)**:
    | Code | Definition |
    | :--- | :--- |
    | `0` | Securities not further defined |
    | `1` | Securities which need not be further defined (Standard US Ordinary) |
    | `2` | Companies incorporated outside the US |
    | `3` | Americus Trust Components (Primes and Scores) |
    | `4` | Closed-end funds |
    | `5` | Closed-end funds incorporated outside the US |
    | `8` | Real Estate Investment Trusts (REITs) |

  _(Example: `SHRCD = 11` denotes standard US common stock; `SHRCD = 14` denotes common shares of a closed-end fund; `SHRCD = 18` denotes a REIT)._

#### `SHRFLG` — Share Flag

- **Variable Name**: `SHRFLG`
- **Type**: Float / Integer
- **Description**: Integer flag indicating the provenance of the shares outstanding value:
  - `0`: Extracted directly from primary filing sources.
  - `1`: Derived algorithmically by CRSP from a distribution event.

#### `SHROUT` — Number of Shares Outstanding

- **Variable Name**: `SHROUT`
- **Type**: Float (recorded in **thousands**)
- **Description**: Number of publicly traded common shares outstanding as of the observation date. Multiply by $1,000$ to obtain total shares outstanding.

#### `SICCD` — Standard Industrial Classification Code

- **Variable Name**: `SICCD`
- **Type**: Float / Integer (4-digit integer between 100 and 9999)
- **Description**: US Government Standard Industrial Classification code:
  - Digits 1–2: Major Group.
  - Digits 1–3: Industry Group.
  - All 4 digits: Specific Industry.
  - Missing codes are recorded as `0`.

#### `HSICCD` — Header Standard Industrial Classification Code

- **Variable Name**: `HSICCD`
- **Type**: Float / Integer
- **Description**: The most recent valid, non-zero SIC code assigned to the security across its history. Set to `0` if no SIC code was ever reported.

#### `HSICMG` — Header SIC Major Group

- **Variable Name**: `HSICMG`
- **Type**: Float / Integer
- **Description**: The first 2 digits of `HSICCD`, representing the overarching economic major group (e.g. `28` for Chemicals and Allied Products).

#### `HSICIG` — Header SIC Industry Group

- **Variable Name**: `HSICIG`
- **Type**: Float / Integer
- **Description**: The first 3 digits of `HSICCD`, identifying the specific 3-digit industry subgroup.

#### `NAICS` — North American Industry Classification System Code

- **Variable Name**: `NAICS`
- **Type**: Char (up to 6 digits)
- **Description**: Hierarchical industry code established by OMB in 1997 to supersede the SIC system across North America. Available in CRSP from August 24, 2001 onwards. Blank for earlier or unknown observations.

---

### Exchange & Trading Status

#### `EXCHCD` — Exchange Code

- **Variable Name**: `EXCHCD`
- **Type**: Float / Integer
- **Description**: Exchange on which the security is currently listed:

| Code | Definition                                  |
| :--- | :------------------------------------------ |
| `-2` | Halted by NYSE or AMEX                      |
| `-1` | Suspended by NYSE, AMEX, or Nasdaq          |
| `0`  | Not trading on NYSE, AMEX, or Nasdaq        |
| `1`  | New York Stock Exchange (NYSE)              |
| `2`  | American Stock Exchange (AMEX)              |
| `3`  | The Nasdaq Stock Market                     |
| `4`  | Arca Stock Market                           |
| `5`  | Mutual Funds (quoted by Nasdaq)             |
| `10` | Boston Stock Exchange                       |
| `13` | Chicago Stock Exchange                      |
| `16` | Pacific Stock Exchange                      |
| `17` | Philadelphia Stock Exchange                 |
| `19` | Toronto Stock Exchange                      |
| `20` | Over-The-Counter (Non-Nasdaq Dealer Quotes) |
| `31` | When-issued trading on NYSE                 |
| `32` | When-issued trading on AMEX                 |
| `33` | When-issued trading on Nasdaq               |

#### `HEXCD` — Header Exchange Code

- **Variable Name**: `HEXCD`
- **Type**: Float / Integer
- **Description**: Displays the most recent primary exchange listed for the security. Valid values are `1` (NYSE), `2` (AMEX), or `3` (Nasdaq). Regional exchanges are not represented.

#### `PRIMEXCH` — Primary Exchange

- **Variable Name**: `PRIMEXCH`
- **Type**: Char (1 character)
- **Description**: Single-character code identifying the security's primary listing venue:

| Code | Exchange                       |
| :--- | :----------------------------- |
| `N`  | New York Stock Exchange (NYSE) |
| `A`  | American Stock Exchange (AMEX) |
| `Q`  | The Nasdaq Stock Market        |
| `R`  | Arca                           |
| `X`  | Other Exchange                 |

#### `TRDSTAT` — Trading Status

- **Variable Name**: `TRDSTAT`
- **Type**: Char (1 character)
- **Description**: Single-character trading status indicator:
  - `A`: Active trading.
  - `H`: Halted by exchange.
  - `S`: Suspended from trading.
  - `X`: Unknown status.

#### `SECSTAT` — Security Status

- **Variable Name**: `SECSTAT`
- **Type**: Char (1 character)
- **Description**: Describes the delivery status of the security:
  - `R`: Regular Way delivery.
  - `W`: When Issued trading.
  - `E`: Ex-Distributed.
  - `Q`: Bankruptcy proceedings.

#### `TRTSCD` — Traits Code

- **Variable Name**: `TRTSCD`
- **Type**: Float / Integer
- **Description**: 1-digit code describing the issue's market structure traits:
  - `0`: Unknown
  - `1`: Active
  - `2`: Trading with only one market maker
  - `3`: Suspended
  - `4`: Inactive
  - `5`: Delisted

#### `NMSIND` — Nasdaq National Market Indicator

- **Variable Name**: `NMSIND`
- **Type**: Float / Integer
- **Description**: Indicates membership status in The Nasdaq National Market (now Global Market):
  - `0`: Unknown or unavailable
  - `1`: Nasdaq SmallCap (pre-June 15, 1992)
  - `2`: Nasdaq National Market
  - `3`: Nasdaq SmallCap (post-June 15, 1992)
  - `4`: Capital Market (post-July 1, 2006)
  - `5`: Global Market (post-July 1, 2006)
  - `6`: Global Select Market (post-July 1, 2006)

#### `MMCNT` — Market Maker Count

- **Variable Name**: `MMCNT`
- **Type**: Float / Integer
- **Description**: The number of registered market makers competing in the issue on Nasdaq. Missing in December 1982 for small companies and throughout February 1986 due to NASD source limitations.

#### `NSDINX` — NASD Index Classification

- **Variable Name**: `NSDINX`
- **Type**: Float / Integer
- **Description**: NASD internal business classification index:
  - `0`: Unknown
  - `1`: No index
  - `2`: Industrial company
  - `3`: Bank
  - `4`: Other financial institution
  - `5`: Insurance company
  - `6`: Transportation company
  - `7`: Utility company

---

### Prices, Volume & Quotes

#### `PRC` — Closing Price or Bid/Ask Average

- **Variable Name**: `PRC`
- **Type**: Decimal / Float
- **Description**: Closing trade price or negative bid/ask average on the trading day:
  - $\text{PRC} > 0$: An actual closing trade occurred at price $\text{PRC}$.
  - $\text{PRC} < 0$: **No trade occurred**. The absolute value $|\text{PRC}|$ is the midpoint bid/ask average:
    $$|\text{PRC}| = \frac{\text{BID} + \text{ASK}}{2}$$
  - $\text{PRC} = 0$: Neither price nor quote was available.

#### `OPENPRC` — Price Open

- **Variable Name**: `OPENPRC`
- **Type**: Decimal / Float
- **Description**: First trade price executed after the market opens. Available for NYSE, AMEX, and Nasdaq securities from June 15, 1992 onwards (and for NYSE between Dec 1925 and June 1962).

#### `ASKHI` — Ask or High Price

- **Variable Name**: `ASKHI`
- **Type**: Decimal / Float
- **Description**: The highest trading price of the day. If no trades occurred, contains the closing ask price (with a negative sign indicating a quote rather than a trade). Set to `0` if unavailable.

#### `BIDLO` — Bid or Low Price

- **Variable Name**: `BIDLO`
- **Type**: Decimal / Float
- **Description**: The lowest trading price of the day. If no trades occurred, contains the closing bid price (with a negative sign indicating a quote rather than a trade). Set to `0` if unavailable.

#### `BID` — Closing Bid Price

- **Variable Name**: `BID`
- **Type**: Decimal / Float
- **Description**: Closing bid price quotation available daily for NYSE, AMEX, and Nasdaq:
  - **Nasdaq**: Uses inside quotations (highest bid across all market makers at 4:00 PM Eastern). Available for National Market from Nov 1, 1982 and all Nasdaq from June 15, 1992.
  - **NYSE/AMEX**: Represents the last representative bid quote before market close. Set to `0` from 1992 onwards when quotes were deemed unrepresentative (e.g. penny bids with double-price asks from off-market dealers). Continuous series available from Dec 28, 1992.

#### `ASK` — Closing Ask Price

- **Variable Name**: `ASK`
- **Type**: Decimal / Float
- **Description**: Closing ask price quotation available daily for NYSE, AMEX, and Nasdaq:
  - **Nasdaq**: Uses inside quotations (lowest ask across all market makers at 4:00 PM Eastern).
  - **NYSE/AMEX**: Represents the last representative ask quote before market close. Set to `0` from 1992 onwards when determined unrepresentative. Continuous series available from Dec 28, 1992.

#### `VOL` — Share Volume

- **Variable Name**: `VOL`
- **Type**: Decimal / Float
- **Description**: Number of shares traded during the day across all exchanges.
  - Nasdaq reports exact shares (e.g. `12,345`).
  - NYSE/AMEX data sources historically rounded volumes to hundreds of shares (e.g. `12,300`).
  - Set to `-99` if missing. A value of `0` indicates no trades occurred on that day.

#### `NUMTRD` — Number of Trades

- **Variable Name**: `NUMTRD`
- **Type**: Integer
- **Description**: Total count of transactions executed on the Nasdaq Stock Market for the security on the date. Set to `99` or `-99` if unavailable. Reported for Nasdaq National Market since Nov 1, 1982, and all Nasdaq since June 15, 1992. (Not available for NYSE/AMEX).

---

### Holding Period & Delisting Returns

#### `RET` — Holding Period Return (With Dividends)

- **Variable Name**: `RET`
- **Type**: Decimal / Float
- **Description**: Total return from holding the security from trading day $t'$ to day $t$:
  $$r(t) = \frac{p(t) \cdot f(t) + d(t)}{p(t')} - 1$$
  where:
  - $t'$ = date of last available price prior to $t$ (usually $t-1$, but up to 10 trading periods back).
  - $p(t)$ = closing price or bid/ask midpoint at time $t$.
  - $d(t)$ = dividend or cash adjustment at time $t$.
  - $f(t)$ = price adjustment factor (`FACPR`) at time $t$.
- **Special Missing Return Codes**:
  - `-66.0`: More than 10 trading periods elapsed between price dates without trading.
  - `-77.0`: Not trading on current exchange at time $t$.
  - `-88.0`: No return calculation possible (out of active array bounds).
  - `-99.0`: Missing return due to missing price at time $t$.
  - `'B'`: Return missing because no valid quote existed on day $t-1$.
  - `'C'`: Return missing because no valid quote existed on day $t$.

#### `RETX` — Holding Period Return Without Dividends

- **Variable Name**: `RETX`
- **Type**: Decimal / Float
- **Description**: Capital appreciation return excluding ordinary dividends. Computed with the same formula as `RET`, but setting cash dividend $d(t) = 0$.

#### `DLSTCD` — Delisting Code

- **Variable Name**: `DLSTCD`
- **Type**: Float / Integer (3-digit code)
- **Description**: Specifies whether an issue continues trading or defines the exact reason for delisting:
  - **Major Categories (1st Digit)**:
    - `100`: Active (still trading).
    - `200`: Mergers (merged with another firm).
    - `300`: Exchanges (exchanged for another security).
    - `400`: Liquidations.
    - `500`: Dropped by exchange (failure to meet listing requirements, non-payment of fees).
    - `600`: Expirations.
    - `900`: Domestic companies that became foreign.
  - Digits 2 and 3 provide granular sub-reasons (e.g. delisting codes `470`, `480` for issues pending research).

#### `DLPRC` — Delisting Price

- **Variable Name**: `DLPRC`
- **Type**: Float
- **Description**: The price of the security on the date specified in `NEXTDT`. Positive values indicate trade prices; negative values indicate bid/ask averages. Set to `0` if `NEXTDT = 0`.

#### `DLAMT` — Amount After Delisting

- **Variable Name**: `DLAMT`
- **Type**: Float
- **Description**: The post-delisting per-share payout or value used to compute the delisting return (`DLRET`). Represents either a trade price found on another market or the sum of cash/stock payments in a merger or liquidation.

#### `DLRET` — Delisting Return (Total)

- **Variable Name**: `DLRET`
- **Type**: Float
- **Description**: Total return realized after a security delists, comparing total post-delisting value against the last active trading price.
- **Missing Delisting Return Codes**:
  - `-55.0`: No sources available to establish delisting value, or issue is pending research.
  - `-66.0`: More than 10 trading periods between last price and first post-delist trade.
  - `-88.0`: Security is still active (not delisted).
  - `-99.0`: Security trades on a new exchange, but no price sources are currently linked.
- **Valuation Rules**:
  - If evidence confirms shareholders received nothing, stock is deemed worthless and `DLRET = -1.0` (-100% loss).
  - Delisting codes `470` and `480` default to `-55.0`.

#### `DLRETX` — Delisting Return Without Dividends

- **Variable Name**: `DLRETX`
- **Type**: Float
- **Description**: Delisting return excluding ordinary cash dividends paid between the last trading date and the post-delist payment date.

#### `NWPERM` — New CRSP Permanent Number

- **Variable Name**: `NWPERM`
- **Type**: Float / Integer
- **Description**: The `PERMNO` of the acquiring company's stock when an issue ceases trading due to a merger or share exchange. Acts as a forward pointer enabling continuous survivorship tracking across corporate acquisitions. Set to `0` if inapplicable or non-merger.

---

### Distributions, Adjustments & Corporate Actions

#### `DISTCD` — Distribution Code

- **Variable Name**: `DISTCD`
- **Type**: Float / Integer (4-digit code)
- **Description**: 4-digit code identifying corporate distribution events:
  - **1st Digit (Event Type)**:
    | Code | Meaning |
    | :--- | :--- |
    | `1` | Ordinary dividend |
    | `2` | Liquidating dividend |
    | `3` | Exchanges and reorganizations |
    | `4` | Subscription rights |
    | `5` | Stock splits and stock dividends |
    | `6` | Issuance notation (change in shares outstanding) |
    | `7` | General information announcement for dropped issues |
  - **2nd Digit (Payment Method)**:
    | Code | Meaning |
    | :--- | :--- |
    | `0` | Unknown / unencoded |
    | `1` | Unspecified or not applicable |
    | `2` | Cash (US Dollars) |
    | `3` | Cash (foreign currency converted to USD) |
    | `4` | Cash (Canadian Dollars, converted to USD) |
    | `5` | Same issue of common stock |
    | `6` | Units including common stock |
    | `7` | Issue of different common stock tracked on file |
    | `8` | Other property |

#### `DIVAMT` — Dividend Cash Amount

- **Variable Name**: `DIVAMT`
- **Type**: Float
- **Description**: Per-share US dollar cash amount of the distribution (dividends, spin-offs, cash merger consideration, or liquidation proceeds).

#### `FACPR` — Factor to Adjust Price

- **Variable Name**: `FACPR`
- **Type**: Float
- **Description**: Multiplier adjustment applied to historical prices following distributions to ensure equivalent pre- and post-event comparability:
  1. **Cash Dividends**: `FACPR = 0`.
  2. **Mergers / Full Liquidations** (security disappears): `FACPR = -1.0`.
  3. **Stock Splits and Stock Dividends**:
     $$\text{FACPR} = \frac{s(t) - s(t')}{s(t')} = \frac{s(t)}{s(t')} - 1$$
     where $s(t)$ is shares outstanding after the event and $s(t')$ is shares before.
  4. **Spin-offs and Rights Offerings**:
     $$\text{FACPR} = \frac{\text{DIVAMT}}{P(t)}$$
     where $P(t)$ is the ex-distribution stock price.

#### `FACSHR` — Factor to Adjust Shares Outstanding

- **Variable Name**: `FACSHR`
- **Type**: Float
- **Description**: Adjustment factor for shares outstanding. Equals `FACPR` for standard splits and stock dividends. Set to `0` for spin-offs. For subscription rights, equals the reciprocal of the subscription ratio.

#### `CFACPR` — Cumulative Factor to Adjust Price

- **Variable Name**: `CFACPR`
- **Type**: Float
- **Description**: The cumulative product of price adjustment factors from the current date forward to the end of the time series:
  $$\text{CFACPR}(t) = \prod_{\tau > t} (1 + \text{FACPR}(\tau))$$
  To obtain the split-adjusted price:
  $$\text{Price}_{\text{adj}}(t) = \frac{\text{PRC}(t)}{\text{CFACPR}(t)}$$

#### `CFACSHR` — Cumulative Factor to Adjust Shares Outstanding

- **Variable Name**: `CFACSHR`
- **Type**: Float
- **Description**: Cumulative product of share adjustment factors:
  $$\text{CFACSHR}(t) = \prod_{\tau > t} (1 + \text{FACSHR}(\tau))$$
  To obtain split-adjusted shares:
  $$\text{Shares}_{\text{adj}}(t) = \text{SHROUT}(t) \times \text{CFACSHR}(t)$$

#### `ACPERM` — Acquiring PERMNO

- **Variable Name**: `ACPERM`
- **Type**: Float / Integer
- **Description**: The `PERMNO` of an associated security received in a spin-off, stock exchange, or merger. Set to a number $< 1000$ if inapplicable or unknown.

#### `ACCOMP` — Acquiring PERMCO

- **Variable Name**: `ACCOMP`
- **Type**: Float / Integer
- **Description**: The `PERMCO` of the parent or acquiring firm linked to a distribution or cash merger payout. Set to `0` if unknown or not tracked by CRSP.

---

### Market Indices & Benchmark Returns

#### `vwretd` — Value-Weighted Return (With Dividends)

- **Variable Name**: `vwretd`
- **Type**: Float
- **Description**: Combined daily return on a capitalization-weighted market portfolio of all NYSE, AMEX, and Nasdaq stocks, **including** all distributions (ADRs excluded).

#### `vwretx` — Value-Weighted Return (Without Dividends)

- **Variable Name**: `vwretx`
- **Type**: Float
- **Description**: Combined daily return on a capitalization-weighted market portfolio of all NYSE, AMEX, and Nasdaq stocks, **excluding** dividend distributions (ADRs excluded).

#### `ewretd` — Equal-Weighted Return (With Dividends)

- **Variable Name**: `ewretd`
- **Type**: Float
- **Description**: Combined daily return on an equal-weighted market portfolio across NYSE, AMEX, and Nasdaq stocks, **including** distributions (includes ADRs).

#### `ewretx` — Equal-Weighted Return (Without Dividends)

- **Variable Name**: `ewretx`
- **Type**: Float
- **Description**: Combined daily return on an equal-weighted market portfolio across NYSE, AMEX, and Nasdaq stocks, **excluding** distributions (includes ADRs).

#### `sprtrn` — Return on the S&P Composite Index

- **Variable Name**: `sprtrn`
- **Type**: Float
- **Description**: Daily return on the Standard & Poor's 500 Composite Index:
  $$\text{SPRTRN}(t) = \frac{\text{SPINDX}(t)}{\text{SPINDX}(t-1)} - 1$$
  where $\text{SPINDX}(t)$ is the closing level of the S&P 500 Composite Index on date $t$.
