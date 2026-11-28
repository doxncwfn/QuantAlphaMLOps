"""Modal serverless training job for v1.3 cross-sectional Russell 1000 ranking model."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("russell1000-v13-baseline")

    # Volume storing the pre-processed feature store and training artifacts
    volume = modal.Volume.from_name("russell1000-data", create_if_missing=True)

    # Remote GPU environment image
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.2.2",
            "numpy==1.26.4",
            "pandas==2.2.2",
            "polars==0.20.31",
            "scipy==1.12.0",
            "pyarrow==15.0.2",
            "duckdb==1.1.3",
            "wandb>=0.17.0",
        )
        .add_local_dir("src", remote_path="/root/src")
    )

    def _execute_training(
        target_folds: list[int] | None = None,
        max_epochs: int = 30,
        patience: int = 10,
        batch_size: int = 8,
        grad_accum_steps: int = 8,
        use_wandb: bool = True,
        wandb_project: str = "russell1000-v13-baseline",
    ) -> dict:
        """Internal helper to execute walk-forward folds on remote GPU."""
        import sys

        import torch

        sys.path.insert(0, "/root")
        from src.training.trainer import run_full_walk_forward_pipeline

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device name: {torch.cuda.get_device_name(0)}")

        res = run_full_walk_forward_pipeline(
            table_name="silver.features_daily",
            cache_dir="/data",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            n_splits=5,
            max_epochs=max_epochs,
            patience=patience,
            target_folds=target_folds,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )

        out_file = Path("/data/baseline_1a_results.json")
        with open(out_file, "w") as f:
            clean_res = json.loads(json.dumps(res, default=str))
            json.dump(clean_res, f, indent=2)

        volume.commit()
        return res

    @app.function(
        image=image,
        gpu="A10",
        volumes={"/data": volume},
        secrets=[modal.Secret.from_dotenv()],
        timeout=14400,
    )
    def train_remote_a10(
        target_folds: list[int] | None = None,
        max_epochs: int = 30,
        patience: int = 10,
        batch_size: int = 8,
        grad_accum_steps: int = 8,
        use_wandb: bool = True,
        wandb_project: str = "russell1000-v13-baseline",
    ) -> dict:
        return _execute_training(
            target_folds=target_folds,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )

    @app.function(
        image=image,
        gpu="A10G",
        volumes={"/data": volume},
        secrets=[modal.Secret.from_dotenv()],
        timeout=14400,
    )
    def train_remote_a10g(
        target_folds: list[int] | None = None,
        max_epochs: int = 30,
        patience: int = 10,
        batch_size: int = 8,
        grad_accum_steps: int = 8,
        use_wandb: bool = True,
        wandb_project: str = "russell1000-v13-baseline",
    ) -> dict:
        return _execute_training(
            target_folds=target_folds,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )

    @app.function(
        image=image,
        gpu="L4",
        volumes={"/data": volume},
        secrets=[modal.Secret.from_dotenv()],
        timeout=14400,
    )
    def train_remote_l4(
        target_folds: list[int] | None = None,
        max_epochs: int = 30,
        patience: int = 10,
        batch_size: int = 8,
        grad_accum_steps: int = 8,
        use_wandb: bool = True,
        wandb_project: str = "russell1000-v13-baseline",
    ) -> dict:
        return _execute_training(
            target_folds=target_folds,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )

    @app.function(
        image=image,
        gpu="L40S",
        volumes={"/data": volume},
        secrets=[modal.Secret.from_dotenv()],
        timeout=14400,
    )
    def train_remote_l40s(
        target_folds: list[int] | None = None,
        max_epochs: int = 30,
        patience: int = 10,
        batch_size: int = 8,
        grad_accum_steps: int = 8,
        use_wandb: bool = True,
        wandb_project: str = "russell1000-v13-baseline",
    ) -> dict:
        return _execute_training(
            target_folds=target_folds,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )

    # Legacy alias pointing to A10
    train_remote = train_remote_a10

    @app.local_entrypoint()
    def main(
        fold: int = 1,
        all_folds: bool = False,
        max_epochs: int = 30,
        gpu: str = "a10",
        use_wandb: bool = True,
        wandb_project: str = "russell1000-v13-baseline",
        upload_data: bool = False,
    ):
        """CLI entrypoint for Modal training."""
        local_cache = Path("data/cache/silver_features_daily_store.npz")
        if not local_cache.exists():
            raise FileNotFoundError(f"Local cache not found at {local_cache}")

        # Check / upload data to volume if requested
        if upload_data:
            print(
                f"Ensuring feature cache {local_cache} is uploaded to Modal Volume 'russell1000-data'..."
            )
            with volume.batch_upload(force=True) as batch:
                batch.put_file(str(local_cache), "/silver_features_daily_store.npz")
            print("Modal volume ready.")

        target = None if all_folds else [fold]
        gpu_choice = gpu.lower().strip()
        print(
            f"Launching Modal job on {gpu_choice.upper()} (folds: {'all' if all_folds else fold})..."
        )

        runners = {
            "a10": train_remote_a10,
            "a10g": train_remote_a10g,
            "l4": train_remote_l4,
            "l40s": train_remote_l40s,
        }

        order = [gpu_choice] + [
            g for g in ["a10", "l4", "l40s", "a10g"] if g != gpu_choice
        ]
        res = None
        last_err = None

        for candidate in order:
            if candidate not in runners:
                continue
            runner = runners[candidate]
            try:
                print(
                    f"Attempting execution on Modal GPU target: {candidate.upper()}..."
                )
                res = runner.remote(
                    target_folds=target,
                    max_epochs=max_epochs,
                    use_wandb=use_wandb,
                    wandb_project=wandb_project,
                )
                print(f"Remote training complete on {candidate.upper()}!")
                break
            except Exception as e:  # noqa: BLE001
                print(f"Execution failed on {candidate.upper()}: {e}")
                last_err = e
                print("Falling back to next available GPU candidate...")

        if res is None:
            raise RuntimeError(f"All GPU candidates failed. Last error: {last_err}")

        print(json.dumps(res, indent=2, default=str))

else:

    def main():
        print(
            "Modal client not installed locally. Run locally with python -m src.training.trainer."
        )
