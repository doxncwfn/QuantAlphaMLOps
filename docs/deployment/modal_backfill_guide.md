# V3 Full Massive PIT Backfill — Modal Deployment & Operations Guide

## 1. Architectural Overview

The **V3 Full Massive Point-in-Time (PIT) Backfill** executes the historical security identity evidence collection remotely on Modal across all **43,757 ticker spells** in `data/universe/spells.csv`.

```text
                                  Modal Cloud
┌─────────────────────────────────────────────────────────────────────────────┐
│ Modal App: v3-massive-backfill                                              │
│                                                                             │
│   Worker Pool (9 Concurrent Threads pinned 1:1 to Modal Secret Keys)        │
│   ├── Worker 1 ──> MASSIVE_API_KEY_1 (pacing: 12.1s independent lock)      │
│   ├── Worker 2 ──> MASSIVE_API_KEY_2 (pacing: 12.1s independent lock)      │
│   │   ...                                                                   │
│   └── Worker 9 ──> MASSIVE_API_KEY_9 (pacing: 12.1s independent lock)      │
│                                                                             │
│   Thread-Safe Work Queue (queue.Queue)                                      │
│   ├── Level 1: Primary Trading Session Midpoint Lookup                      │
│   ├── Level 2: Boundary Fallback (Start/End date recovery)                  │
│   └── Level 3: Within-Spell Drift Detection                                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Mounted at /modal_data
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Persistent Modal Volume: v3-massive-backfill                                │
│ ├── cache/massive/       (atomic JSON writes via .tmp + rename)             │
│ ├── checkpoints/         (checkpoint_*.parquet saved every 1,000 spells)    │
│ ├── manifests/           (massive_manifest.parquet, modal_manifest.json)    │
│ ├── telemetry/           (worker_telemetry.parquet, telemetry_summary.json) │
│ └── logs/                (v3_modal_backfill.log, v3_modal_summary.md)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼ modal volume get / modal_retrieve.py
┌─────────────────────────────────────────────────────────────────────────────┐
│ Local Repository (Post-Backfill Step B)                                     │
│ ├── data/identity/cache/massive/                                            │
│ ├── data/manifests/v3/                                                      │
│ └── python3 src/identity/v3/resolver.py (Generates candidate datasets)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Guarantees
- **Zero Production Overwrites**: The Modal deployment operates strictly on evidence acquisition. It never modifies production identity tables (`data/identity/security_master.parquet` etc.) or `data/universe/spells.csv`.
- **Key-Slot Isolation**: Each worker thread is strictly bound to its assigned key slot (`WORKER_1` $\leftrightarrow$ `MASSIVE_API_KEY_1`, etc.).
- **Independent Rate Limiting**: Each worker enforces its own 12.1-second minimum interval via private thread locks, avoiding unnecessary serialization.
- **Atomic Cache Writes**: Responses are written to a `.tmp` file in the same directory, flushed, fsynced, and renamed via `os.replace` to prevent partial writes.
- **Persistent Checkpointing & Resumability**: Checkpoints are saved every 1,000 spells and committed to the persistent Modal Volume. If interrupted, restarting resumes from the last completed spell without re-querying.

---

## 2. Prerequisites & Configuration

### Modal Authentication
Ensure your Modal client is authenticated and target workspace is selected:
```bash
modal profile current
```

### Modal Secrets
The job uses the existing Modal Secret named `massive`:
```bash
modal secret list
```
The secret must contain keys: `MASSIVE_API_KEY_1` through `MASSIVE_API_KEY_9`.

### Persistent Modal Volume
The Modal Volume `v3-massive-backfill` is automatically created on first launch if missing:
```bash
modal volume list
```

---

## 3. Validation & Testing Modes

### Step 1: Execute Dry Run (0 API Calls)
Validates image building, volume mounting, secret visibility (masked), input hash, worker slot configuration, and mock atomic cache/checkpoint writes:
```bash
modal run src/identity/v3/modal_backfill.py --dry-run
```
Expected output:
```json
{
  "status": "DRY_RUN_PASSED",
  "configured_keys": 9,
  "spells_sha256": "5fc79a37cdc341cf7b10a7017ccd75cf7001aa1b91e5dd6501c829b746191bf1",
  "total_spells": 43757,
  "cache_items": 1336
}
```

### Step 2: Execute 9-Key Live Smoke Test (18 Queries)
Validates live HTTP requests across all 9 workers (2 queries per worker), rate limiter pacing, telemetry recording, and atomic persistent cache writes:
```bash
modal run src/identity/v3/modal_backfill.py --smoke-test
```
Expected output:
```json
{
  "status": "PARTIAL",
  "configured_key_count": 9,
  "worker_count": 9,
  "spells_accounted_for": 18,
  "live_requests": 6,
  "cache_hits": 18,
  "failed_requests": 0
}
```

---

## 4. Production Backfill Execution

### Recommended Command (Detached Mode)
Run the backfill in detached background mode so execution continues even if your terminal closes or computer sleeps:
```bash
modal run --detach src/identity/v3/modal_backfill.py
```

### Foreground Mode (Streaming Logs to Terminal)
If you prefer to stream live progress directly in your terminal:
```bash
modal run src/identity/v3/modal_backfill.py
```

---

## 5. Progress Monitoring & Heartbeat

### View Running Modal Apps
```bash
modal app list
```

### Stream Live Logs from Remote App
```bash
modal app logs <APP_ID>
```
*(Replace `<APP_ID>` with the ID printed upon launch, e.g., `ap-ptQVcy47HUoe93nGnT2DIG`)*.

### Heartbeat Output Format
Every 30–60 seconds, the remote job logs a structured progress heartbeat:
```text
------------------------------------------------------------
V3 MODAL MASSIVE PIT BACKFILL HEARTBEAT
Completed       : 12,340 / 43,757 (28.20%)
Remaining       : 31,417
Cache Hits      : 963
Live Requests   : 11,377
Empty Results   : 45
HTTP 429s       : 0
Failures        : 0
Observed Rate   : 85.20 req/min
Active Workers  : 9 / 9
PLANNING_ESTIMATE ETA: 6.14 hours remaining
------------------------------------------------------------
```

---

## 6. Interruption & Resumption Procedure

If the Modal container restarts, is stopped manually, or experiences network disruption, simply re-run the exact same command:
```bash
modal run --detach src/identity/v3/modal_backfill.py
```
The engine automatically:
1. Discovers existing `checkpoint_*.parquet` files on `/modal_data/checkpoints/`.
2. Reconstructs the set of completed `spell_id`s.
3. Skips all previously completed spells and queries only the remaining work.
4. Preserves all existing cache files and continues seamlessly.

---

## 7. Artifact Retrieval (Local Sync)

After the backfill finishes, synchronize all artifacts from the Modal Volume back to your local repository:

### Using the Automated Retrieval CLI:
```bash
python3 src/identity/v3/modal_retrieve.py
```
This automatically retrieves and places:
- Persistent Cache $\to$ `data/identity/cache/massive/`
- Master Manifests $\to$ `data/manifests/v3/`
- Telemetry $\to$ `log/modal_telemetry/`
- Execution Logs $\to$ `log/`
- Checkpoints $\to$ `data/manifests/v3/checkpoints/`

### Manual Retrieval Commands:
```bash
modal volume get v3-massive-backfill cache/massive data/identity/cache/massive --force
modal volume get v3-massive-backfill manifests data/manifests/v3 --force
modal volume get v3-massive-backfill logs log --force
```

---

## 8. Local Step B: Candidate Dataset Resolution

Once the Massive PIT evidence is synced locally, generate the candidate datasets:
```bash
PYTHONPATH=. python3 src/identity/v3/resolver.py
```
This produces all 8 candidate artifacts in `data/identity/candidates/v3/` and `data/universe/candidates/v3/` with full provenance, leaving production tables untouched.

---

## 9. Troubleshooting

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| `Missing keys in secret 'massive'` | Secret is not deployed or keys have different names | Check `modal secret list` and ensure keys are named `MASSIVE_API_KEY_1`..`9`. |
| `Spells SHA-256 mismatch` | `spells.csv` was altered locally | Restore original `data/universe/spells.csv` from git (`git checkout data/universe/spells.csv`). |
| `HTTP 429 Rate Limited` | Transient provider throttle | The worker automatically handles 429 using `Retry-After` header and exponential backoff with jitter. |
| Container Timeout | 24-hour Modal hard limit reached | Re-run `modal run --detach src/identity/v3/modal_backfill.py` to resume remaining spells. |
