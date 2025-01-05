#!/bin/bash

# Set proxy settings if needed
export HTTP_PROXY="http://127.0.0.1:7897"
export HTTPS_PROXY="http://127.0.0.1:7897"

# Remote server details - MODIFY THESE
REMOTE_HOST="172.16.6.4"
REMOTE_USER="mayanze"
REMOTE_PATH="/home/mayanze/PycharmProjects/SwinTF"

# Create local directory for wandb data
LOCAL_WANDB_DIR="wandb_sync_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOCAL_WANDB_DIR"

# First, run the backup script on remote server
echo "Running backup script on remote server..."
backup_output=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_PATH} && python3 sync_wandb.py")

# Extract the backup directory path from the output
backup_dir=$(echo "$backup_output" | grep "BACKUP_DIR=" | cut -d'=' -f2)

if [ -z "$backup_dir" ]; then
    echo "Failed to get backup directory path from remote server"
    rm -rf "$LOCAL_WANDB_DIR"
    exit 1
fi

echo "Backup directory on remote server: $backup_dir"

# Download the backup directory from remote server
echo "Downloading wandb data from remote server..."
scp -r "${REMOTE_USER}@${REMOTE_HOST}:${backup_dir}/*" "$LOCAL_WANDB_DIR/"

if [ $? -ne 0 ]; then
    echo "Failed to download wandb data from remote server"
    rm -rf "$LOCAL_WANDB_DIR"
    exit 1
fi

# Sync each run with wandb
echo "Starting wandb sync..."
cd "$LOCAL_WANDB_DIR"
for run_dir in offline-run*; do
    if [ -d "$run_dir" ]; then
        echo "Syncing $run_dir..."
        wandb sync "$run_dir"
    fi
done
cd ..

# Cleanup
echo "Cleaning up..."
rm -rf "$LOCAL_WANDB_DIR"
ssh ${REMOTE_USER}@${REMOTE_HOST} "rm -rf ${backup_dir}"

# Unset proxy settings
unset HTTP_PROXY
unset HTTPS_PROXY

echo "Sync completed successfully!"
echo "Exiting terminal..."
exit 0
