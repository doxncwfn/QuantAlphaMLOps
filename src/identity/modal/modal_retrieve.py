"""
V3 Modal Artifact Retrieval Tool.
=================================
Safely synchronizes persistent artifacts from the Modal Volume 'v3-massive-backfill'
back to the local repository filesystem.

Retrieves:
1. Massive API Cache: /cache/massive -> data/identity/cache/massive/
2. Master Manifests:  /manifests/    -> data/manifests/v3/
3. Telemetry:         /telemetry/    -> log/modal_telemetry/
4. Execution Logs:    /logs/         -> log/
5. Checkpoints:       /checkpoints/  -> data/manifests/v3/checkpoints/
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VOLUME_NAME = "v3-massive-backfill"

LOCAL_CACHE_DIR = REPO_ROOT / "data" / "identity" / "cache" / "massive"
LOCAL_MANIFESTS_DIR = REPO_ROOT / "data" / "manifests" / "v3"
LOCAL_LOGS_DIR = REPO_ROOT / "log"
LOCAL_TELEMETRY_DIR = REPO_ROOT / "log" / "modal_telemetry"
LOCAL_CHECKPOINTS_DIR = REPO_ROOT / "data" / "manifests" / "v3" / "checkpoints"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("modal_retrieve")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def run_volume_get(remote_path: str, local_dest: Path, logger: logging.Logger) -> bool:
    """Executes `modal volume get` for a specific remote path and flattens nested folder."""
    local_dest.mkdir(parents=True, exist_ok=True)
    cmd = ["modal", "volume", "get", VOLUME_NAME, remote_path, str(local_dest), "--force"]
    logger.info("Executing: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if res.stdout.strip():
            logger.info(res.stdout.strip())

        # If modal created a nested subdirectory (e.g. local_dest / remote_folder_name), flatten it
        nested_dir = local_dest / Path(remote_path).name
        if nested_dir.exists() and nested_dir.is_dir() and nested_dir != local_dest:
            for item in nested_dir.iterdir():
                dest_item = local_dest / item.name
                if dest_item.exists() and dest_item.is_dir():
                    shutil.rmtree(dest_item)
                elif dest_item.exists():
                    dest_item.unlink()
                shutil.move(str(item), str(local_dest))
            shutil.rmtree(nested_dir)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to retrieve %s: %s\nStderr: %s", remote_path, exc, exc.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Retrieve V3 Backfill Artifacts from Modal Volume")
    parser.add_argument("--volume", type=str, default=VOLUME_NAME, help="Modal Volume name")
    parser.add_argument("--cache-only", action="store_true", help="Retrieve only the Massive API cache")
    parser.add_argument("--manifests-only", action="store_true", help="Retrieve only manifests and reports")
    parser.add_argument("--all", action="store_true", default=True, help="Retrieve all artifacts")
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("=" * 80)
    logger.info("V3 MODAL ARTIFACT RETRIEVAL TOOL")
    logger.info("=" * 80)
    logger.info("Modal Volume: %s", args.volume)
    logger.info("Local Repo Root: %s", REPO_ROOT)

    retrieval_plan = []
    if args.cache_only:
        retrieval_plan = [("cache/massive", LOCAL_CACHE_DIR)]
    elif args.manifests_only:
        retrieval_plan = [
            ("manifests", LOCAL_MANIFESTS_DIR),
            ("logs", LOCAL_LOGS_DIR)
        ]
    else:
        retrieval_plan = [
            ("manifests", LOCAL_MANIFESTS_DIR),
            ("logs", LOCAL_LOGS_DIR),
            ("telemetry", LOCAL_TELEMETRY_DIR),
            ("cache/massive", LOCAL_CACHE_DIR),
            ("checkpoints", LOCAL_CHECKPOINTS_DIR),
        ]

    success_count = 0
    for remote_p, local_p in retrieval_plan:
        logger.info("Retrieving '%s' -> '%s'...", remote_p, local_p)
        ok = run_volume_get(remote_p, local_p, logger)
        if ok:
            success_count += 1

    logger.info("-" * 80)
    logger.info("RETRIEVAL SUMMARY: %d / %d artifact sets synchronized successfully.",
                success_count, len(retrieval_plan))
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
