"""Verify checkpoint, metadata, and result JSON provenance and integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "loss_type",
    "lam_hub",
    "device",
    "dtype",
    "lr",
    "weight_decay",
    "warmup_epochs",
    "min_checkpoint_epoch",
    "seed",
    "batch_size",
    "git_sha",
    "target_hash",
)

REQUIRED_CKPT_KEYS: tuple[str, ...] = (
    "fold_idx",
    "best_epoch",
    "best_val_ic",
    "model_state_dict",
    "history",
)

LEGACY_PREFIXES: tuple[str, ...] = (
    "v1.5_",
    "baseline_1a",
    "cheap_fixes",
    "full_diagnosis",
    "s1_ablation",
)


def compute_store_target_hash(store: Any) -> str:
    """Compute canonical SHA-256 over sorted date_targets float32 arrays."""
    hasher = hashlib.sha256()
    date_targets = getattr(store, "date_targets", {})
    if isinstance(date_targets, dict):
        arrays = (date_targets[k] for k in sorted(date_targets.keys()))
    else:
        arrays = date_targets
    for arr in arrays:
        hasher.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def _is_hex64(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in value)


def _is_legacy_name(path: Path) -> bool:
    return any(path.name.startswith(prefix) for prefix in LEGACY_PREFIXES)


def verify_checkpoint(
    pt_path: Path,
    expected_target_hash: str | None = None,
) -> dict[str, Any]:
    """Validate one .pt checkpoint and its companion .metadata.json."""
    issues: list[str] = []
    try:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(pt_path),
            "status": "INVALID",
            "issues": [f"torch.load failed: {exc}"],
        }

    if not isinstance(ckpt, dict):
        return {
            "path": str(pt_path),
            "status": "INVALID",
            "issues": ["checkpoint payload is not a dict"],
        }

    missing_ckpt = [k for k in REQUIRED_CKPT_KEYS if k not in ckpt]
    if missing_ckpt:
        issues.append(f"missing ckpt keys: {missing_ckpt}")

    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        issues.append("missing or empty model_state_dict")
    else:
        non_finite = [
            k
            for k, v in state.items()
            if isinstance(v, torch.Tensor)
            and v.is_floating_point()
            and not torch.isfinite(v).all()
        ]
        if non_finite:
            issues.append(f"non-finite weights in: {non_finite[:5]}")

    meta_path = pt_path.with_suffix(".metadata.json")
    if not meta_path.exists():
        if not issues and _is_legacy_name(pt_path):
            return {
                "path": str(pt_path),
                "status": "LEGACY_PRE_AUDIT",
                "issues": ["missing companion .metadata.json (pre-audit artifact)"],
                "best_epoch": ckpt.get("best_epoch"),
                "best_val_ic": ckpt.get("best_val_ic"),
            }
        issues.append(f"missing metadata file: {meta_path.name}")
        return {"path": str(pt_path), "status": "INVALID", "issues": issues}

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        issues.append(f"invalid metadata JSON: {exc}")
        return {"path": str(pt_path), "status": "INVALID", "issues": issues}

    if not isinstance(meta, dict):
        issues.append("metadata JSON is not a dict")
        return {"path": str(pt_path), "status": "INVALID", "issues": issues}

    missing_meta = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
    if missing_meta:
        issues.append(f"missing metadata keys: {missing_meta}")

    git_sha = meta.get("git_sha")
    if not isinstance(git_sha, str) or len(git_sha.strip()) < 7:
        issues.append(f"invalid git_sha: {git_sha!r}")

    target_hash = meta.get("target_hash")
    if not _is_hex64(target_hash):
        issues.append(f"invalid target_hash format: {target_hash!r}")
    elif expected_target_hash is not None and target_hash != expected_target_hash:
        issues.append(
            f"target_hash mismatch: got {target_hash[:12]}..., expected {expected_target_hash[:12]}..."
        )

    status = "INVALID" if issues else "VALID"
    return {
        "path": str(pt_path),
        "status": status,
        "issues": issues,
        "git_sha": git_sha,
        "target_hash": target_hash,
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_ic": ckpt.get("best_val_ic"),
    }


def verify_result_json(
    json_path: Path,
    expected_target_hash: str | None = None,
) -> dict[str, Any]:
    """Validate one cached diagnostic/result JSON for parseability and provenance."""
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(json_path),
            "status": "INVALID",
            "issues": [f"invalid JSON: {exc}"],
        }

    if not isinstance(payload, dict):
        return {
            "path": str(json_path),
            "status": "INVALID",
            "issues": ["top-level JSON is not a dict"],
        }

    has_prov = "git_sha" in payload and "target_hash" in payload
    if not has_prov:
        if _is_legacy_name(json_path):
            return {
                "path": str(json_path),
                "status": "LEGACY_PRE_AUDIT",
                "issues": ["pre-audit JSON without git_sha/target_hash"],
            }
        return {
            "path": str(json_path),
            "status": "INVALID",
            "issues": ["missing git_sha and target_hash"],
        }

    issues: list[str] = []
    if not isinstance(payload["git_sha"], str) or len(payload["git_sha"].strip()) < 7:
        issues.append(f"invalid git_sha: {payload['git_sha']!r}")
    if not _is_hex64(payload["target_hash"]):
        issues.append(f"invalid target_hash: {payload['target_hash']!r}")
    elif (
        expected_target_hash is not None
        and payload["target_hash"] != expected_target_hash
    ):
        issues.append("target_hash mismatch")

    return {
        "path": str(json_path),
        "status": "INVALID" if issues else "VALID",
        "issues": issues,
        "git_sha": payload.get("git_sha"),
        "target_hash": payload.get("target_hash"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify model checkpoints and result JSON provenance."
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("models/checkpoints")
    )
    parser.add_argument("--results-dir", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--warehouse-path", type=Path, default=Path("data/warehouse/russell1000.duckdb")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--expected-target-hash", type=str, default=None)
    parser.add_argument("--skip-target-hash", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if legacy or zero valid checkpoints exist.",
    )
    args = parser.parse_args(argv)

    expected_hash = args.expected_target_hash
    if expected_hash is None and not args.skip_target_hash:
        from src.data.loader import FeatureDataStore

        store = FeatureDataStore(
            warehouse_path=args.warehouse_path, cache_dir=args.cache_dir
        )
        expected_hash = compute_store_target_hash(store)
        print(f"Live store target_hash: {expected_hash}")

    pt_files = (
        sorted(args.checkpoint_dir.glob("*.pt")) if args.checkpoint_dir.exists() else []
    )
    json_files = (
        sorted(args.results_dir.glob("*.json")) if args.results_dir.exists() else []
    )

    ckpt_reports = [verify_checkpoint(p, expected_hash) for p in pt_files]
    json_reports = [verify_result_json(p, expected_hash) for p in json_files]
    all_reports = ckpt_reports + json_reports

    counts: dict[str, int] = {"VALID": 0, "LEGACY_PRE_AUDIT": 0, "INVALID": 0}
    for rep in all_reports:
        counts[rep["status"]] = counts.get(rep["status"], 0) + 1
        detail = (
            "; ".join(rep["issues"])
            if rep["issues"]
            else f"git={rep.get('git_sha', '')[:8]}"
        )
        print(f"[{rep['status']}] {rep['path']} ({detail})")

    print(
        f"Summary: total={len(all_reports)} | VALID={counts['VALID']} | "
        f"LEGACY_PRE_AUDIT={counts['LEGACY_PRE_AUDIT']} | INVALID={counts['INVALID']}"
    )

    if counts["INVALID"] > 0:
        return 1
    if args.strict:
        valid_ckpts = sum(1 for r in ckpt_reports if r["status"] == "VALID")
        if counts["LEGACY_PRE_AUDIT"] > 0 or valid_ckpts == 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
