from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.check_env import (
    check_device_parity,
    check_packages,
    parse_pinned_requirements,
)
from scripts.check_env import (
    main as check_env_main,
)
from scripts.verify_artifacts import (
    REQUIRED_METADATA_KEYS,
    verify_checkpoint,
    verify_result_json,
)
from scripts.verify_artifacts import (
    main as verify_main,
)


def _write_valid_ckpt(ckpt_path: Path, target_hash: str) -> None:
    ckpt = {
        "fold_idx": 1,
        "best_epoch": 1,
        "best_val_ic": 0.02,
        "model_state_dict": {"w": torch.ones(4)},
        "history": [{"epoch": 1, "train_loss": 0.1}],
    }
    torch.save(ckpt, ckpt_path)
    meta = {k: "val" for k in REQUIRED_METADATA_KEYS}
    meta.update(
        {
            "lam_hub": 0.0,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "warmup_epochs": 15,
            "min_checkpoint_epoch": 15,
            "seed": 42,
            "batch_size": 8,
            "git_sha": "abcdef1",
            "target_hash": target_hash,
        }
    )
    ckpt_path.with_suffix(".metadata.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


def test_verify_checkpoint_valid_and_hash_mismatch(tmp_path: Path) -> None:
    expected = "a" * 64
    ckpt_path = tmp_path / "fold1.pt"
    _write_valid_ckpt(ckpt_path, expected)

    ok_rep = verify_checkpoint(ckpt_path, expected_target_hash=expected)
    assert ok_rep["status"] == "VALID"

    bad_rep = verify_checkpoint(ckpt_path, expected_target_hash="b" * 64)
    assert bad_rep["status"] == "INVALID"
    assert any("target_hash mismatch" in issue for issue in bad_rep["issues"])


def test_verify_legacy_strict_vs_non_strict(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpts"
    res_dir = tmp_path / "res"
    ckpt_dir.mkdir()
    res_dir.mkdir()

    legacy_pt = ckpt_dir / "v1.5_s1a_fold1.pt"
    torch.save(
        {
            "fold_idx": 1,
            "best_epoch": 1,
            "best_val_ic": 0.01,
            "model_state_dict": {"w": torch.ones(2)},
            "history": [],
        },
        legacy_pt,
    )
    legacy_json = res_dir / "v1.5_s1a_fold_1.json"
    legacy_json.write_text(json.dumps({"variant": "s1a"}), encoding="utf-8")

    assert verify_checkpoint(legacy_pt)["status"] == "LEGACY_PRE_AUDIT"
    assert verify_result_json(legacy_json)["status"] == "LEGACY_PRE_AUDIT"

    rc_non_strict = verify_main(
        [
            "--checkpoint-dir",
            str(ckpt_dir),
            "--results-dir",
            str(res_dir),
            "--skip-target-hash",
        ]
    )
    assert rc_non_strict == 0

    rc_strict = verify_main(
        [
            "--checkpoint-dir",
            str(ckpt_dir),
            "--results-dir",
            str(res_dir),
            "--skip-target-hash",
            "--strict",
        ]
    )
    assert rc_strict == 1


def test_verify_strict_passes_with_valid_only(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpts"
    res_dir = tmp_path / "res"
    ckpt_dir.mkdir()
    res_dir.mkdir()

    expected = "f" * 64
    _write_valid_ckpt(ckpt_dir / "post_audit_fold1.pt", expected)
    rc = verify_main(
        [
            "--checkpoint-dir",
            str(ckpt_dir),
            "--results-dir",
            str(res_dir),
            "--expected-target-hash",
            expected,
            "--strict",
        ]
    )
    assert rc == 0


def test_check_env_pins_and_skip_data() -> None:
    pins = dict(parse_pinned_requirements(Path("pyproject.toml")))
    assert "torchsort" in pins and pins["torchsort"] == "0.1.10"
    ok, _ = check_packages(Path("pyproject.toml"))
    assert ok
    assert check_env_main(["--skip-data", "--device", "cpu"]) == 0


def test_option_b_split_parity_gate() -> None:
    src = Path("scripts/check_env.py").read_text(encoding="utf-8")
    assert "model.eval()" not in src, (
        "check_device_parity must keep train() mode for cuDNN RNN backward"
    )
    cpu_ok, cpu_lines = check_device_parity("cpu")
    assert cpu_ok
    assert any("max|dL_kernel|=" in ln for ln in cpu_lines)
    if torch.backends.mps.is_available():
        assert check_env_main(["--skip-data", "--check-mps-parity"]) == 0
