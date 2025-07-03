#!/bin/bash

# Check if a camera number is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <cam_number>"
    exit 1
fi

CAM_NUM=$1
HOST="cam${CAM_NUM}.local"
USER="cam${CAM_NUM}"

echo "Connecting to ${USER}@${HOST}..."
ssh "${USER}@${HOST}"