#!/bin/bash
set -e

# ============================
# CONFIG
# ============================
GPU="device=0"
IMAGE="yolo-inference:1.0.1-rtx3090"
SCRIPT="/code/predict.sh"
CONTAINER_NAME="yolo_inference_container"

# ============================
# RUN DOCKER
# ============================
docker run --rm -d --gpus "$GPU" \
  --name "${CONTAINER_NAME}" \
  -v "$RESULT_DIR":/result \
  "$IMAGE" \
  /bin/bash "$SCRIPT"


