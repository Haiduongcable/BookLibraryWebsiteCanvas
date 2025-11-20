#!/bin/bash
set -e

# ============================
# CONFIG
# ============================
GPU="device=0"
IMAGE="yolo-inference:1.0.1-rtx3090"
SCRIPT="/code/predict.sh"
CONTAINER_NAME="yolo_inference_container"
RESULT_DIR="/bigdisk/duongnh59/temp/hungdeptrai/BookLibraryWebsiteCanvas/result"
DATA_DIR="/bigdisk/duongnh59/temp/hungdeptrai/BookLibraryWebsiteCanvas/data"
docker run --rm -d \
  -p 9777:9777 \
  --gpus "$GPU" \
  --name "$CONTAINER_NAME" \
  --network host \
  -v "$DATA_DIR":/data \
  -v "$RESULT_DIR":/result \
  "$IMAGE" \
  /bin/bash /code/start_jupyter.sh
