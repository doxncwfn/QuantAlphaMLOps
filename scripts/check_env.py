"""Pre-flight environment, data, dependency, and single-batch compute check."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

import duckdb
import numpy as np
import torch

from scripts.verify_artifacts import compute_store_target_hash
from src.losses.combined import CombinedRankingLoss
from src.losses.decile_mono import DecileMonotonicityLoss
from src.models.architecture import BaselineArchitectureY

EXPECTED_FEATURES_DAILY_ROWS = 10_745_931


def parse_pinned_requirements(pyproject_path: Path) -> list[tuple[str, str]]:
    """Extract (package_name, exact_version) pins from pyproject.toml."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    raw_specs: list[str] = list(data.get("project", {}).get("dependencies", []))
    raw_specs.extend(
        data.get("project", {}).get("optional-dependencies", {}).get("dev", [])
    )
    pins: list[tuple[str, str]] = []
    seen: set[str] = set()
    for spec in raw_specs:
        m = re.match(r"^\s*([A-Za-z0-9_.-]+)==([^\s;#]+)", spec)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            pins.append((m.group(1), m.group(2)))
    return pins


def check_packages(pyproject_path: Path) -> tuple[bool, list[str]]:
    """Verify installed package versions match pyproject.toml == pins."""
    pins = parse_pinned_requirements(pyproject_path)
    lines: list[str] = []
    ok = True
    for pkg, expected in pins:
        try:
            installed = importlib.metadata.version(pkg)
            matched = installed == expected
            tag = "OK" if matched else "MISMATCH"
            if not matched:
                ok = False
            lines.append(f"  {tag}: {pkg}=={installed} (expected {expected})")
        except importlib.metadata.PackageNotFoundError:
            ok = False
            lines.append(f"  MISSING: {pkg} (expected {expected})")
    return ok, lines


def check_duckdb_and_cache(
    warehouse_path: Path, cache_dir: Path
) -> tuple[bool, list[str]]:
    """Verify DuckDB warehouse row counts, SQL NULL / IEEE NaN invariants, and npz cache."""
    lines: list[str] = []
    if not warehouse_path.exists():
        return False, [f"  FAIL: missing warehouse {warehouse_path}"]

    con = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        row_count, t5_null, t5_nan, ys_nan = con.execute(
            """
            SELECT
                COUNT(*)::BIGINT,
                COUNT(*) FILTER (WHERE target_5d IS NULL)::BIGINT,
                COUNT(*) FILTER (WHERE isnan(target_5d))::BIGINT,
                COUNT(*) FILTER (WHERE isnan(y_star))::BIGINT
            FROM silver.features_daily
            """
        ).fetchone()
    finally:
        con.close()

    ok = True
    if row_count != EXPECTED_FEATURES_DAILY_ROWS:
        ok = False
        lines.append(f"  FAIL: row_count={row_count} != {EXPECTED_FEATURES_DAILY_ROWS}")
    else:
        lines.append(f"  OK: silver.features_daily rows={row_count}")

    if t5_nan != 0 or ys_nan != 0:
        ok = False
        lines.append(f"  FAIL: IEEE NaNs found target_5d={t5_nan}, y_star={ys_nan}")
    else:
        lines.append(f"  OK: zero IEEE NaNs (target_5d SQL NULL={t5_null})")

    npz_path = cache_dir / "silver_features_daily_store.npz"
    if not npz_path.exists():
        ok = False
        lines.append(f"  FAIL: missing cache {npz_path}")
        return ok, lines

    from src.data.loader import FeatureDataStore

    store = FeatureDataStore(warehouse_path=warehouse_path, cache_dir=cache_dir)
    n_eval = len(store.eval_date_indices)
    target_hash = compute_store_target_hash(store)
    if n_eval == 0 or not np.isfinite(store.grid).all():
        ok = False
        lines.append(f"  FAIL: cache invalid (eval_dates={n_eval})")
    else:
        lines.append(
            f"  OK: cache eval_dates={n_eval}, npz={npz_path.stat().st_size / (1024**2):.1f}MB, "
            f"target_hash={target_hash[:16]}..."
        )
    return ok, lines


def _make_synthetic_batch(
    device: torch.device, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    b, k, t = 2, 32, 10
    x_stock = torch.randn(b, k, t, 19, generator=rng).to(device)
    x_ctx = torch.randn(b, t, 11, generator=rng).to(device)
    mask = torch.ones(b, k, dtype=torch.bool, device=device)
    target_5d = (torch.randn(b, k, generator=rng) * 0.05).to(device)
    y_norm = torch.randn(b, k, generator=rng).to(device)
    y_star = torch.linspace(-0.45, 0.45, k).unsqueeze(0).expand(b, k).clone().to(device)
    return x_stock, x_ctx, mask, target_5d, y_norm, y_star


@contextlib.contextmanager
def _mps_safe_determinism(device_type: str):
    """Temporarily set warn_only=True when strict deterministic mode hits unsupported MPS ops."""
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    changed = device_type == "mps" and prev_det and not prev_warn
    if changed:
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        if changed:
            torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)


def check_compute_smoke(device: torch.device) -> tuple[bool, list[str]]:
    """Run a 1-batch forward/backward check with both loss functions on target device."""
    lines: list[str] = []
    ok = True
    with _mps_safe_determinism(device.type):
        for loss_name, criterion in [
            ("combined", CombinedRankingLoss().to(device)),
            (
                "decile_mono",
                DecileMonotonicityLoss(K=4, tau_sort=0.01, lam_hub=0.0).to(device),
            ),
        ]:
            torch.manual_seed(42)
            model = BaselineArchitectureY(
                n_stock_features=19,
                n_context_features=11,
                d_model=64,
                lstm_hidden=64,
                lstm_layers=1,
                context_hidden=16,
                n_heads=4,
                tfm_layers=1,
                ff_dim=128,
                dropout=0.0,
            ).to(device)
            x_s, x_c, mask, t5, y_norm, y_star = _make_synthetic_batch(device, seed=11)
            model.zero_grad(set_to_none=True)
            scores = model(x_s, x_c, mask)
            loss_out = criterion(scores, t5, y_star, mask, y_norm)
            loss = loss_out["total"] if isinstance(loss_out, dict) else loss_out[0]
            loss.backward()
            grad_norms = [
                float(p.grad.detach().norm())
                for p in model.parameters()
                if p.grad is not None and torch.isfinite(p.grad).all()
            ]
            n_params = sum(1 for _ in model.parameters())
            finite_ok = (
                bool(torch.isfinite(loss))
                and len(grad_norms) == n_params
                and sum(grad_norms) > 0.0
            )
            if not finite_ok:
                ok = False
            lines.append(
                f"  {'OK' if finite_ok else 'FAIL'}: {device.type} {loss_name} loss={float(loss.detach()):.6f}, "
                f"finite_grads={len(grad_norms)}/{n_params}"
            )
    return ok, lines


def check_device_parity(target_device: str = "cuda") -> tuple[bool, list[str]]:
    """Compare CPU vs accelerator (cuda/mps) via Option B split parity gate.

    Gate thresholds (fp32 deterministic pass):
      (i)   loss-kernel forward parity on identical scores: max|dL_kernel| < 1e-5
      (ii)  end-to-end forward parity: max|dL| < 1e-5 (combined) or < 1e-4 (decile_mono)
      (iii) backward L2 grad-norm parity: | ||g_dev||_2 - ||g_cpu||_2 | / ||g_cpu||_2 < 1e-3
    """
    if target_device == "cuda" and not torch.cuda.is_available():
        return True, ["  SKIP: CUDA unavailable on this host"]
    if target_device == "mps" and not torch.backends.mps.is_available():
        return True, ["  SKIP: MPS unavailable on this host"]

    lines: list[str] = []
    ok = True
    for loss_name, crit_factory, dl_tol in [
        ("combined", lambda: CombinedRankingLoss(), 1e-5),
        (
            "decile_mono",
            lambda: DecileMonotonicityLoss(K=4, tau_sort=0.01, lam_hub=0.0),
            1e-4,
        ),
    ]:
        grad_vecs: dict[str, torch.Tensor] = {}
        losses: dict[str, float] = {}
        scores_cpu_ref: torch.Tensor | None = None
        for dev_str in ("cpu", target_device):
            with _mps_safe_determinism(dev_str):
                dev = torch.device(dev_str)
                torch.manual_seed(42)
                model = BaselineArchitectureY(
                    n_stock_features=19,
                    n_context_features=11,
                    d_model=64,
                    lstm_hidden=64,
                    lstm_layers=1,
                    context_hidden=16,
                    n_heads=4,
                    tfm_layers=1,
                    ff_dim=128,
                    dropout=0.0,
                ).to(dev)
                # Keep train() mode: dropout=0.0 above and cuDNN forbids RNN backward in eval().
                crit = crit_factory().to(dev)
                x_s, x_c, mask, t5, y_norm, y_star = _make_synthetic_batch(dev, seed=19)
                model.zero_grad(set_to_none=True)
                scores = model(x_s, x_c, mask)
                if dev_str == "cpu":
                    scores_cpu_ref = scores.detach().clone()
                out = crit(scores, t5, y_star, mask, y_norm)
                loss = out["total"] if isinstance(out, dict) else out[0]
                loss.backward()
                losses[dev_str] = float(loss.detach().cpu())
                grad_vecs[dev_str] = torch.cat(
                    [
                        p.grad.detach().cpu().reshape(-1)
                        for p in model.parameters()
                        if p.grad is not None
                    ]
                )
        dev_target = torch.device(target_device)
        crit_ref = crit_factory().to(dev_target)
        _, _, m_ref, t5_ref, yn_ref, ys_ref = _make_synthetic_batch(dev_target, seed=19)
        with torch.no_grad():
            assert scores_cpu_ref is not None
            out_same = crit_ref(
                scores_cpu_ref.to(dev_target), t5_ref, ys_ref, m_ref, yn_ref
            )
            loss_same = float(
                (out_same["total"] if isinstance(out_same, dict) else out_same[0])
                .detach()
                .cpu()
            )
        dl_kernel = abs(losses["cpu"] - loss_same)
        dl = abs(losses["cpu"] - losses[target_device])
        norm_cpu = float(grad_vecs["cpu"].norm())
        norm_dev = float(grad_vecs[target_device].norm())
        rel_gnorm = abs(norm_dev - norm_cpu) / max(norm_cpu, 1e-12)
        elem_max = float((grad_vecs["cpu"] - grad_vecs[target_device]).abs().max())
        passed = dl_kernel < 1e-5 and dl < dl_tol and rel_gnorm < 1e-3
        if not passed:
            ok = False
        lines.append(
            f"  {'OK' if passed else 'FAIL'}: cpu<->{target_device} {loss_name} "
            f"max|dL_kernel|={dl_kernel:.3e} (<1e-5), max|dL|={dl:.3e} (<{dl_tol:.0e}), "
            f"rel_dNorm={rel_gnorm:.3e} (<1e-3), info_max|dGrad|={elem_max:.3e}"
        )
    if target_device == "cuda":
        dev = torch.device("cuda")
        torch.manual_seed(42)
        bf_model = BaselineArchitectureY(
            n_stock_features=19,
            n_context_features=11,
            d_model=64,
            lstm_hidden=64,
            lstm_layers=1,
            context_hidden=16,
            n_heads=4,
            tfm_layers=1,
            ff_dim=128,
            dropout=0.0,
        ).to(dev)
        x_s, x_c, mask, t5, y_norm, y_star = _make_synthetic_batch(dev, seed=19)
        bf_model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            scores = bf_model(x_s, x_c, mask)
        loss_bf = DecileMonotonicityLoss(K=4, tau_sort=0.01, lam_hub=0.0).to(dev)(
            scores.float(), t5, y_star, mask, y_norm
        )[0]
        loss_bf.backward()
        bf_ok = bool(torch.isfinite(loss_bf.detach())) and all(
            torch.isfinite(p.grad).all()
            for p in bf_model.parameters()
            if p.grad is not None
        )
        if not bf_ok:
            ok = False
        lines.append(
            f"  {'OK' if bf_ok else 'FAIL'}: cuda bfloat16+float32_loss autocast finite smoke"
        )
    return ok, lines


def check_cuda_parity() -> tuple[bool, list[str]]:
    """Compare CPU vs CUDA parity under Option B split gate."""
    return check_device_parity("cuda")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight check for runtime, data, and GPU/CPU compute."
    )
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument(
        "--warehouse-path", type=Path, default=Path("data/warehouse/russell1000.duckdb")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "mps", "cuda"]
    )
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--check-cuda-parity", action="store_true")
    parser.add_argument("--check-mps-parity", action="store_true")
    args = parser.parse_args(argv)

    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    py_ok = sys.version_info >= (3, 11)
    print(f"[Python] {py_ver} ({'OK' if py_ok else 'FAIL < 3.11'})")

    pkg_ok, pkg_lines = check_packages(args.pyproject)
    print("[Packages]")
    print("\n".join(pkg_lines))

    data_ok = True
    if not args.skip_data:
        data_ok, data_lines = check_duckdb_and_cache(
            args.warehouse_path, args.cache_dir
        )
        print("[Data]")
        print("\n".join(data_lines))

    if args.device is not None:
        dev = torch.device(args.device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    print(
        f"[Hardware] cuda={torch.cuda.is_available()}, mps={torch.backends.mps.is_available()}, selected={dev.type}"
    )
    smoke_ok, smoke_lines = check_compute_smoke(dev)
    print("\n".join(smoke_lines))

    parity_ok = True
    if args.check_cuda_parity:
        c_ok, parity_lines = check_cuda_parity()
        parity_ok = parity_ok and c_ok
        print("[CUDA Parity]")
        print("\n".join(parity_lines))
    if args.check_mps_parity:
        m_ok, mps_lines = check_device_parity("mps")
        parity_ok = parity_ok and m_ok
        print("[MPS Parity]")
        print("\n".join(mps_lines))

    all_ok = py_ok and pkg_ok and data_ok and smoke_ok and parity_ok
    print(f"ENV_CHECK={'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
