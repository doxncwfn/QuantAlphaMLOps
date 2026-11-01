"""Run directory and tracking.

Every run writes runs/<UTC timestamp>_<name>_<hash6>/:
    config.resolved.yaml   the exact config after the env overlay and CLI time parameters
    meta.json              git sha, dirty flag, config hash, dataset table hashes, versions, times, status
    metrics.json           every logged scalar, grouped by prefix (e.g. fold_1/val_ic, test/mean_ic)
    run.log                standard logging output
    predictions.parquet    written by the train stage

The local run directory is the source of truth. W&B is optional and only receives
params and scalar metrics, never tables or data.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from qam.config import RunConfig

log = logging.getLogger("qam")

_PACKAGES = ("qam", "numpy", "pandas", "polars", "duckdb", "torch", "pytorch-lightning", "pydantic")


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {"python": platform.python_version()}
    for name in _PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def _flatten(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), value, out)
    else:
        out[prefix] = obj


class RunContext:
    def __init__(self, cfg: RunConfig, run_dir: Path | None = None):
        self.cfg = cfg
        if run_dir is None:
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            base = cfg.paths.runs_root / f"{stamp}_{cfg.name}_{cfg.hash()[:6]}"
            run_dir, n = base, 1
            while run_dir.exists():  # two runs of the same config within one second
                run_dir, n = base.with_name(f"{base.name}_{n}"), n + 1
            run_dir.mkdir(parents=True)
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name
        self._metrics_path = self.run_dir / "metrics.json"
        self._meta_path = self.run_dir / "meta.json"
        self._handler: logging.Handler | None = None
        self._wandb = None

    # ------------------------------------------------------------ lifecycle

    def __enter__(self) -> "RunContext":
        self._attach_log_file()
        if not (self.run_dir / "config.resolved.yaml").exists():
            with (self.run_dir / "config.resolved.yaml").open("w") as fh:
                yaml.safe_dump(self.cfg.to_plain(), fh, sort_keys=False)
        root = Path(__file__).resolve().parents[2]
        self.update_meta(
            run_id=self.run_id,
            config_hash=self.cfg.hash(),
            git_sha=_git(root, "rev-parse", "HEAD"),
            git_dirty=bool(_git(root, "status", "--porcelain")),
            versions=_versions(),
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            status="running",
        )
        self._start_wandb()
        log.info("run %s started in %s", self.run_id, self.run_dir)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        status = "failed" if exc_type else "finished"
        if exc_type:
            log.error("run %s failed", self.run_id, exc_info=(exc_type, exc, tb))
        self.update_meta(status=status, finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        if self._wandb is not None:
            self._wandb.finish(exit_code=1 if exc_type else 0)
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler.close()

    def _attach_log_file(self) -> None:
        handler = logging.FileHandler(self.run_dir / "run.log")
        handler.setFormatter(logging.Formatter(f"%(asctime)s %(levelname)s [{self.run_id}] %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
        self._handler = handler

    # ------------------------------------------------------------ recording

    def meta(self) -> dict[str, Any]:
        return json.loads(self._meta_path.read_text()) if self._meta_path.exists() else {}

    def update_meta(self, **fields: Any) -> None:
        meta = self.meta()
        meta.update(fields)
        self._meta_path.write_text(json.dumps(meta, indent=2, default=str))

    def log_metrics(self, values: dict[str, Any], prefix: str = "", step: int | None = None) -> None:
        scalars: dict[str, float] = {}
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"metric {key!r} must be a scalar number, got {type(value).__name__}")
            scalars[f"{prefix}/{key}" if prefix else key] = float(value)
        stored = json.loads(self._metrics_path.read_text()) if self._metrics_path.exists() else {}
        stored.update(scalars)
        self._metrics_path.write_text(json.dumps(stored, indent=2, sort_keys=True))
        if self._wandb is not None:
            self._wandb.log(scalars, step=step)

    def metrics(self) -> dict[str, float]:
        return json.loads(self._metrics_path.read_text()) if self._metrics_path.exists() else {}

    # ------------------------------------------------------------ W&B

    def _start_wandb(self) -> None:
        wb = self.cfg.tracking_spec.wandb
        if not wb.enabled or wb.mode == "disabled":
            return
        import wandb  # optional dependency: pip install qam[tracking]

        params: dict[str, Any] = {}
        _flatten("", self.cfg.to_plain(), params)
        params.pop("paths.warehouse", None)
        params.pop("paths.data_root", None)
        params.pop("paths.runs_root", None)
        self._wandb = wandb.init(
            project=wb.project,
            entity=wb.entity,
            mode=wb.mode,
            name=self.run_id,
            dir=str(self.run_dir),
            config=params,
        )

    def lightning_loggers(self, fold: str) -> list[Any]:
        from pytorch_lightning.loggers import CSVLogger

        loggers: list[Any] = [CSVLogger(save_dir=str(self.run_dir), name="lightning", version=fold)]
        if self._wandb is not None:
            from pytorch_lightning.loggers import WandbLogger

            loggers.append(WandbLogger(experiment=self._wandb, prefix=fold, log_model=False))
        return loggers
