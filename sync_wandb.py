import os
import glob
import shutil
from datetime import datetime

def backup_offline_wandb():
    # Path to the offline wandb directory
    offline_dir = 'wandb_offline/wandb'
    
    # Create a backup directory with timestamp in a known location
    backup_base = os.path.expanduser('wandb_backups')  # Store backups in user's home directory
    os.makedirs(backup_base, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = os.path.join(backup_base, f'wandb_backup_{timestamp}')
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f"Created backup directory: {backup_dir}")
    
    # Check if the offline directory exists
    if not os.path.exists(offline_dir):
        print(f"No wandb directory found at {offline_dir}")
        return None
    
    # Find all offline run directories
    run_dirs = glob.glob(os.path.join(offline_dir, "offline-run*"))

    if not run_dirs:
        print("No offline runs found to backup")
        return None
    
    print(f"Found {len(run_dirs)} offline runs")
    
    # Backup all runs
    for run_dir in run_dirs:
        try:
            run_name = os.path.basename(run_dir)
            backup_path = os.path.join(backup_dir, run_name)
            
            print(f"\nBacking up run: {run_name}")
            shutil.copytree(run_dir, backup_path, dirs_exist_ok=True)
            print(f"Successfully backed up to: {backup_path}")
            
        except Exception as e:
            print(f"Error backing up run {run_dir}: {str(e)}")
            continue
    
    print(f"\nAll runs have been backed up to: {backup_dir}")
    print("BACKUP_DIR=" + os.path.abspath(backup_dir))  # Marker for the shell script
    return backup_dir

if __name__ == "__main__":
    backup_offline_wandb() 