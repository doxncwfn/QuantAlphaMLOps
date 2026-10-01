from pathlib import Path

import modal

src_dir = Path(__file__).parent
root_dir = src_dir.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(root_dir / "requirements.txt"))
    .add_local_dir(str(src_dir), remote_path="/root/src")
)

app = modal.App("quant-model-training")

data_vol = modal.Volume.from_name("quant-data-vol")


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/mnt/data": data_vol},
    secrets=[modal.Secret.from_name("wandb-secret")],  # Doxncwfn's Modal Workspace
    # secrets=[modal.Secret.from_name("doxncwfn-secret")] # Ltrol's Modal Workspace
)
def run_training():
    import sys
    from pathlib import Path

    import yaml

    sys.path.append("/root")
    sys.path.append("/root/src")

    cfg_path = Path("/root/config/config.yaml")
    if not cfg_path.exists():
        cfg_path = Path("config/config.yaml")

    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    cfg["data_dir"] = "/mnt/data/db"
    cfg["fama_french_dir"] = "/mnt/data/factors"
    cfg["ckpt_dir"] = "/mnt/data/checkpoints"

    import src.train

    src.train.main(config=cfg)


@app.function(image=image, gpu="A100", timeout=14400, volumes={"/mnt/data": data_vol})
def run_xai():
    import sys
    from pathlib import Path

    import yaml

    sys.path.append("/root")
    sys.path.append("/root/src")

    cfg_path = Path("/root/config/config.yaml")
    if not cfg_path.exists():
        cfg_path = Path("config/config.yaml")

    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    cfg["data_dir"] = "/mnt/data/db"
    cfg["fama_french_dir"] = "/mnt/data/factors"
    cfg["ckpt_dir"] = "/mnt/data/checkpoints"  # Doxncwfn's Modal Workspace
    # cfg['ckpt_dir'] = '/mnt/data/artifacts/checkpoints' # Ltrol's Modal Workspace

    from src.evaluation.xai import run_xai_diagnostics

    run_xai_diagnostics(
        oos_parquet_path="/mnt/data/checkpoints/oos_predictions_3.parquet",
        config=cfg,
    )


@app.local_entrypoint(name="train")
def train_main():
    print("🚀 Launching Quant GPU training job on Modal (A10G)...")
    run_training.remote()
    print("✅ Training complete.")


@app.local_entrypoint(name="xai")
def xai_main():
    print("🚀 Launching XAI Diagnostics on Modal...")
    run_xai.remote()
    print("✅ XAI execution complete.")
