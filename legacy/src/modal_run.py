import modal
from pathlib import Path

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
    secrets=[modal.Secret.from_name("wandb-secret")] # Doxncwfn's Modal Workspace
    # secrets=[modal.Secret.from_name("doxncwfn-secret")] # Ltrol's Modal Workspace
)
def run_training():
    import sys
    sys.path.append("/root")
    sys.path.append("/root/src")

    import config
    config.CONFIG['data_dir'] = '/mnt/data/db'
    config.CONFIG['fama_french_dir'] = '/mnt/data/factors'
    config.CONFIG['ckpt_dir'] = '/mnt/data/checkpoints'

    import src.train
    src.train.main()

@app.function(
    image=image,
    gpu="A100",
    timeout=14400,
    volumes={"/mnt/data": data_vol}
)
def run_xai():
    import sys
    sys.path.append("/root")
    sys.path.append("/root/src")

    import config
    config.CONFIG['data_dir'] = '/mnt/data/db'
    config.CONFIG['fama_french_dir'] = '/mnt/data/factors'
    config.CONFIG['ckpt_dir'] = '/mnt/data/checkpoints' # Doxncwfn's Modal Workspace
    # config.CONFIG['ckpt_dir'] = '/mnt/data/artifacts/checkpoints' # Ltrol's Modal Workspace

    from src.evaluation.xai import run_xai_diagnostics
    run_xai_diagnostics(oos_parquet_path="/mnt/data/checkpoints/oos_predictions_3.parquet")

@app.local_entrypoint(name="train")
def train_main():
    print("🚀 Launching Quant GPU training job on Modal (A10G)...")
    run_training.remote()
    print("✅ Training complete.")

@app.local_entrypoint(name="xai")
def xai_main():
    print(f"🚀 Launching XAI Diagnostics on Modal...")
    run_xai.remote()
    print("✅ XAI execution complete.")