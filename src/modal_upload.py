import modal
from pathlib import Path

app = modal.App("quant-data-uploader")
vol = modal.Volume.from_name("quant-data-vol", create_if_missing=True)

@app.local_entrypoint()
def main():
    base_dir = Path(__file__).parent.parent
    db_dir = base_dir / "db"
    factors_dir = base_dir / "factors"
    
    print("Starting batch upload to Modal Volume 'quant-data-vol'...")
    with vol.batch_upload(force=True) as batch:
        # Upload DB CSVs
        if db_dir.exists():
            csv_files = list(db_dir.glob("*.csv"))
            print(f"Uploading {len(csv_files)} files from local {db_dir}...")
            for filepath in csv_files:
                batch.put_file(filepath, f"/data/{filepath.name}")
        else:
            print(f"Warning: Local directory {db_dir} not found.")
            
        # Upload Factors CSVs
        if factors_dir.exists():
            csv_files = list(factors_dir.glob("*.csv"))
            print(f"Uploading {len(csv_files)} files from local {factors_dir}...")
            for filepath in csv_files:
                batch.put_file(filepath, f"/factors/{filepath.name}")
        else:
            print(f"Warning: Local directory {factors_dir} not found.")

        # Upload local artifacts (checkpoints/OOS Prediction)
        # # Upload Artifacts (all files recursively)
        # artifacts_dir = base_dir / "artifacts"
        # if artifacts_dir.exists():
        #     artifact_files = list(artifacts_dir.rglob("*"))
        #     files_to_upload = [f for f in artifact_files if f.is_file()]
        #     print(f"Uploading {len(files_to_upload)} files from local {artifacts_dir} recursively...")
        #     for filepath in files_to_upload:
        #         rel_path = filepath.relative_to(artifacts_dir)
        #         # Store under /artifacts/ path in volume, including subdirs
        #         batch.put_file(filepath, f"/artifacts/{rel_path.as_posix()}")
        # else:
        #     print(f"Warning: Local directory {artifacts_dir} not found.")
 

    print(f"Data upload complete! Files are safely stored in '{vol.name}'.")