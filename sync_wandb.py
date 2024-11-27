import os
import wandb
import glob
import shutil
from pathlib import Path
from datetime import datetime

def download_and_sync_wandb():
    # Path to the offline wandb directory
    offline_dir = 'wandb_offline'
    
    # Create a backup directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f'wandb_backup_{timestamp}'
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f"Created backup directory: {backup_dir}")
    
    # Check if the offline directory exists
    if not os.path.exists(offline_dir):
        print(f"No offline wandb directory found at {offline_dir}")
        return None
    
    # Find all run directories
    run_dirs = glob.glob(os.path.join(offline_dir, "**", "run-*.wandb"), recursive=True)
    
    if not run_dirs:
        print("No offline runs found to sync")
        return None
    
    print(f"Found {len(run_dirs)} offline runs")
    
    # First, backup all runs
    for run_file in run_dirs:
        try:
            run_dir = str(Path(run_file).parent)
            run_name = os.path.basename(run_dir)
            backup_path = os.path.join(backup_dir, run_name)
            
            print(f"\nBacking up run: {run_name}")
            shutil.copytree(run_dir, backup_path, dirs_exist_ok=True)
            print(f"Successfully backed up to: {backup_path}")
            
        except Exception as e:
            print(f"Error backing up run {run_file}: {str(e)}")
            continue
    
    print(f"\nAll runs have been backed up to: {backup_dir}")
    print("BACKUP_DIR=" + os.path.abspath(backup_dir))  # Special marker for the bash script
    
    # Initialize wandb in online mode
    os.environ.pop('WANDB_MODE', None)  # Remove offline mode if set
    
    for run_file in run_dirs:
        try:
            print(f"\nProcessing run: {run_file}")
            
            # Get the run directory
            run_dir = str(Path(run_file).parent)
            
            # Initialize a new run
            run = wandb.init(project="SwinIR", sync_tensorboard=True, dir=run_dir)
            
            # Sync the run
            print(f"Syncing run {run.id}...")
            wandb.finish()
            
        except Exception as e:
            print(f"Error syncing run {run_file}: {str(e)}")
            continue

    return backup_dir

if __name__ == "__main__":
    download_and_sync_wandb() 