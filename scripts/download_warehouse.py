import os
import sys
import time
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv(".env")
token = os.environ.get("MOTHERDUCK_API_KEY", "")
if not token:
    print("ERROR: MOTHERDUCK_API_KEY not found in .env", flush=True)
    sys.exit(1)

target = Path("data/warehouse/russell1000.duckdb")
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    target.unlink()

print("Connecting to MotherDuck...", flush=True)
md = duckdb.connect(f"md:stock_data?motherduck_token={token}", read_only=True)

print(f"Opening local warehouse {target}...", flush=True)
local = duckdb.connect(str(target))

schemas = ["bronze", "silver"]
for s in schemas:
    local.execute(f"CREATE SCHEMA IF NOT EXISTS {s};")

tables = [
    ("bronze", "_manifest"),
    ("bronze", "stopped_2526"),
    ("bronze", "master_list"),
    ("bronze", "russell_lists"),
    ("bronze", "prices_2526"),
    ("bronze", "crsp_daily"),
    ("silver", "security"),
    ("silver", "russell"),
    ("silver", "security_history"),
    ("silver", "market_factor"),
    ("silver", "delisting"),
    ("silver", "prices_daily"),
]

for schema, tbl in tables:
    t0 = time.time()
    print(f"[{schema}.{tbl}] Counting rows...", end="", flush=True)
    count = md.execute(f'SELECT count(*) FROM "{schema}"."{tbl}"').fetchone()[0]
    print(f" {count:,} rows. Transferring via parquet stream...", flush=True)

    tmp_parquet = target.parent / f"_tmp_{schema}_{tbl}.parquet"
    if tmp_parquet.exists():
        tmp_parquet.unlink()

    # Export to parquet directly from MotherDuck to avoid large in-memory Arrow buffers
    md.execute(
        f'COPY "{schema}"."{tbl}" TO \'{tmp_parquet}\' (FORMAT PARQUET, COMPRESSION ZSTD);'
    )
    # Load into local DuckDB
    local.execute(
        f'CREATE TABLE "{schema}"."{tbl}" AS SELECT * FROM \'{tmp_parquet}\';'
    )

    tmp_parquet.unlink()
    dt = time.time() - t0
    print(f"[{schema}.{tbl}] Done in {dt:.1f}s.", flush=True)

print("Recreating silver.v_prices view...", flush=True)
local.execute("""
CREATE OR REPLACE VIEW silver.v_prices AS 
SELECT 
    p.*, 
    (p."close" / p.cfacpr) AS adj_close, 
    CASE  
        WHEN (((p.ret IS NULL) AND (d.dlret IS NOT NULL))) THEN (d.dlret) 
        ELSE (((1 + p.ret) * (1 + COALESCE(d.dlret, 0))) - 1) 
    END AS ret_total 
FROM silver.prices_daily AS p 
LEFT JOIN silver.delisting AS d 
    ON (((p.security_id = d.security_id) AND (p.date = d.delist_date)));
""")

md.close()
local.close()

print("Verifying target...", flush=True)
test_con = duckdb.connect(str(target), read_only=True)
print(test_con.sql("SELECT count(*), min(date), max(date) FROM silver.v_prices;"))
test_con.close()
print("All tables downloaded successfully!", flush=True)
