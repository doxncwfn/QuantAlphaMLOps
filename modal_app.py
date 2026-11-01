"""Run the platform on Modal. Thin wrapper: all logic lives in qam.pipeline.

    modal run modal_app.py --config configs/experiments/baseline_hybrid.yaml [--start 2010-01-01] [--end 2020-12-31]

Parameters come from the spec; only the period can be given here (it wins over the spec, with a warning).

Paths come from configs/env/modal.yaml (volume "quant-data-vol" mounted at /mnt/data).
Either the warehouse file is on the volume (tables are built as needed), or only the dataset
DuckDB file is: then the newest stored table versions built from the same config and
component code are used, and a change that needs a rebuild fails with a clear error.
CRSP-derived data on a third-party cloud: check the WRDS terms before uploading it.
"""

from pathlib import Path

import modal

root = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(root / "requirements.txt"))
    .add_local_dir(str(root / "src"), remote_path="/root/src")
    .add_local_dir(str(root / "configs"), remote_path="/root/configs")
    .add_local_file(str(root / "pyproject.toml"), remote_path="/root/pyproject.toml")
)

app = modal.App("quant-model-training")
data_vol = modal.Volume.from_name("quant-data-vol")


@app.function(image=image, gpu="A10G", timeout=86400, volumes={"/mnt/data": data_vol}, secrets=[modal.Secret.from_name("wandb-secret")])
def run_pipeline(config: str, start: str = "", end: str = "") -> str:
    import datetime as dt
    import sys

    sys.path.insert(0, "/root/src")
    from qam.config import load_config
    from qam.pipeline import run

    times = {"start": dt.date.fromisoformat(start) if start else None, "end": dt.date.fromisoformat(end) if end else None}
    cfg = load_config(f"/root/{config}", env="modal", times=times)
    ctx = run(cfg)
    data_vol.commit()
    return str(ctx.run_dir)


@app.local_entrypoint()
def main(config: str = "configs/experiments/baseline_hybrid.yaml", start: str = "", end: str = ""):
    print("run directory:", run_pipeline.remote(config, start, end))
