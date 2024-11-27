#!/bin/bash

# Set proxy settings
export HTTP_PROXY="http://127.0.0.1:7897"
export HTTPS_PROXY="http://127.0.0.1:7897"

# Remote server details
REMOTE_HOST="172.16.6.4"
REMOTE_USER="mayanze"
REMOTE_PATH="/home/mayanze/PycharmProjects/SwinTF"

# SSH to remote server and run the Python script
echo "Connecting to remote server and running sync_wandb.py..."
backup_dir_output=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_PATH} && python3 sync_wandb.py")

# Extract the backup directory path from the output
backup_dir=$(echo "$backup_dir_output" | grep "BACKUP_DIR=" | cut -d'=' -f2)

if [ -z "$backup_dir" ]; then
    echo "Failed to get backup directory path from remote server"
    exit 1
fi

echo "Backup directory on remote server: $backup_dir"

# Copy the backup directory from remote server
echo "Copying backup directory from remote server..."
scp -r "${REMOTE_USER}@${REMOTE_HOST}:${backup_dir}" ./

# Delete the remote backup directory after successful download
echo "Removing remote backup directory..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "rm -rf ${backup_dir}"

# Get the local backup directory name
local_backup_dir=$(basename "$backup_dir")

# Change to the backup directory
cd "$local_backup_dir" || exit 1

# Run wandb sync command
echo "Running wandb sync..."
wandb sync *

# Go back and clean up
cd ..
rm -rf "$local_backup_dir"

# Unset proxy settings
unset HTTP_PROXY
unset HTTPS_PROXY

echo "Sync completed and directories removed successfully (both local and remote)!"
echo "Exiting terminal..."
exit 0
