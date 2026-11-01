# data/

Local data. Everything here except this README and `warehouse/.gitkeep` is git-ignored. Never commit data: it is CRSP data licensed through WRDS.

```
data/
  warehouse/russell1000.duckdb   the warehouse: daily prices, delistings, universe (download, see below)
  datasets/<name>.duckdb         dataset tables built by `qam build-dataset` / `qam run` (created automatically)
```

The paths come from `configs/env/local.yaml`.

## Getting the warehouse (onboarding)

The warehouse is shared on MotherDuck as a **restricted share**. Only team members covered by the WRDS licence get access.

1. **Get access.** Ask Hào to grant your MotherDuck account access to the share, and to send you the share URL (`md:_share/...`).
2. **Create a MotherDuck token** (MotherDuck UI → Settings → Access tokens) and put it in your shell environment:
   ```bash
   export motherduck_token=...   # never commit it, never put it in a config file
   ```
3. **Download the database** into this folder, from the repository root and with the project's Python environment active:
   ```bash
   python - <<'EOF'
   import duckdb
   con = duckdb.connect("data/warehouse/russell1000.duckdb")   # new local file, attached as "russell1000"
   con.execute("ATTACH '<share URL>' AS src")                  # reads motherduck_token from the environment
   con.execute("COPY FROM DATABASE src TO russell1000")        # physical copy of schemas, tables and views
   con.close()
   EOF
   ```
   If the file already exists, delete it first. `COPY FROM DATABASE` needs an empty target.
4. **Check it:**
   ```bash
   python -c "import duckdb; print(duckdb.connect('data/warehouse/russell1000.duckdb', read_only=True).sql('SELECT count(*), min(date), max(date) FROM silver.v_prices'))"
   ```
5. **Build the dataset tables:** `qam build-dataset configs/experiments/baseline_hybrid.yaml`, or `qam run ...`, which builds as needed.

## Updating

When the warehouse on MotherDuck changes, delete the local file and download it again (step 3). The dataset tables depend on the warehouse file's size and modification time, so the next build creates new table versions on its own. `qam gc CONFIG --yes` removes versions that no run uses.

## Rules

- Keep the MotherDuck share **restricted** to named, WRDS-licensed team members. Never make it unrestricted or send the share URL elsewhere.
- Do not copy the warehouse or dataset files to other places (shared drives, chat, public buckets).
