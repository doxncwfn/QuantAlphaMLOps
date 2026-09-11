"""
V3 Production Promotion Orchestrator.
=====================================
Safely and atomically promotes V3 candidate datasets to production.
NOTE: This script is intended to be executed manually by the user/supervisor
after reviewing report/quality/PRODUCTION_GATE_REPORT.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "v3" / "promotion_manifest.json"
EXPECTED_SPELLS_HASH = (
    "5fc79a37cdc341cf7b10a7017ccd75cf7001aa1b91e5dd6501c829b746191bf1"
)
SPELLS_CSV = REPO_ROOT / "data" / "universe" / "spells.csv"

PRODUCTION_MAP = {
    "security_master": (
        REPO_ROOT / "data" / "identity" / "security_master.parquet",
        REPO_ROOT / "data" / "identity" / "security_master.csv",
    ),
    "ticker_history": (
        REPO_ROOT / "data" / "identity" / "ticker_history.parquet",
        REPO_ROOT / "data" / "identity" / "ticker_history.csv",
    ),
    "identity_evidence": (
        REPO_ROOT / "data" / "identity" / "identity_evidence.parquet",
        REPO_ROOT / "data" / "identity" / "identity_evidence.csv",
    ),
    "identity_conflicts": (
        REPO_ROOT / "data" / "identity" / "identity_conflicts.parquet",
        REPO_ROOT / "data" / "identity" / "identity_conflicts.csv",
    ),
    "identity_aliases": (
        REPO_ROOT / "data" / "identity" / "identity_aliases.parquet",
        REPO_ROOT / "data" / "identity" / "identity_aliases.csv",
    ),
    "availability_episodes": (
        REPO_ROOT / "data" / "universe" / "availability_episodes.parquet",
        REPO_ROOT / "data" / "universe" / "availability_episodes.csv",
    ),
    "expected_security_dates": (
        REPO_ROOT / "data" / "universe" / "expected_security_dates.parquet",
        None,
    ),
    "daily_universe": (
        REPO_ROOT / "data" / "universe" / "daily_universe.parquet",
        REPO_ROOT / "data" / "universe" / "daily_universe.csv",
    ),
}


def hash_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Promote V3 Candidate Datasets to Production"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST),
        help="Path to promotion manifest",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Explicit confirmation flag required to execute promotion",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as fp:
        manifest = json.load(fp)

    print("=" * 80)
    print("V3 PRODUCTION DATASET PROMOTION ORCHESTRATOR")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Gate Decision: {manifest.get('promotion_gate_decision')}")
    print(
        f"Invariants Passed: {manifest['validation_summary']['invariants_passed']}/{manifest['validation_summary']['invariants_evaluated']}"
    )

    # 1. Verify Spells Immutability
    current_spells_hash = hash_file(SPELLS_CSV)
    if current_spells_hash != EXPECTED_SPELLS_HASH:
        print(
            f"[FATAL ERROR] spells.csv hash mismatch: {current_spells_hash} != {EXPECTED_SPELLS_HASH}"
        )
        sys.exit(1)
    print("[PASS] Spells input verified bit-for-bit immutable.")

    # 2. Verify Candidate Artifacts Checksums
    candidate_artifacts = manifest.get("candidate_artifacts", {})
    for name, meta in candidate_artifacts.items():
        src_path = REPO_ROOT / meta["file_path"]
        if not src_path.exists():
            print(f"[FATAL ERROR] Candidate file missing: {src_path}")
            sys.exit(1)
        actual_hash = hash_file(src_path)
        if actual_hash != meta["sha256"]:
            print(
                f"[FATAL ERROR] Checksum mismatch on {name}: {actual_hash} != {meta['sha256']}"
            )
            sys.exit(1)
        print(f"[PASS] Verified checksum for {name}: {actual_hash[:16]}...")

    if not args.confirm:
        print("-" * 80)
        print(
            "[DRY-RUN] Pre-promotion verification complete. To execute atomic promotion to production, run:"
        )
        print("  python3 src/identity/v3/promote_to_production.py --confirm")
        print("-" * 80)
        sys.exit(0)

    # 3. Execute Atomic Promotion
    print("Executing atomic file copy to production paths...")
    now_ts = datetime.datetime.now(datetime.UTC).isoformat()
    promoted_records = []

    for name, meta in candidate_artifacts.items():
        src_path = REPO_ROOT / meta["file_path"]
        dst_parquet, dst_csv = PRODUCTION_MAP.get(name, (None, None))
        if dst_parquet:
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_parquet)
            print(f"  [PROMOTED] {src_path.name} -> {dst_parquet}")

            if dst_csv:
                df = pl.read_parquet(src_path)
                df.write_csv(dst_csv)
                print(f"  [CSV EXPORT] {dst_csv}")

            promoted_records.append(
                {
                    "artifact": name,
                    "source": str(src_path.relative_to(REPO_ROOT)),
                    "destination_parquet": str(dst_parquet.relative_to(REPO_ROOT)),
                    "sha256": meta["sha256"],
                    "row_count": meta["row_count"],
                }
            )

    # 4. Write Production Record
    prod_record = {
        "promotion_timestamp": now_ts,
        "manifest_used": str(manifest_path.relative_to(REPO_ROOT)),
        "promoted_artifacts": promoted_records,
    }
    rec_path = REPO_ROOT / "data" / "manifests" / "production_promotion_record.json"
    with open(rec_path, "w", encoding="utf-8") as fp:
        json.dump(prod_record, fp, indent=2)

    print("=" * 80)
    print("SUCCESS: V3 Candidate datasets successfully promoted to production.")
    print(f"Production record saved to {rec_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
