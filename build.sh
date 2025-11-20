#!/usr/bin/env bash

set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "[build] docker command not found" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
#IMAGE_NAME=${IMAGE_NAME:-vllm-inference:1.0.2-a100}
IMAGE_NAME=${IMAGE_NAME:-yolo-inference:1.0.1-rtx3090}
DOCKERFILE=${DOCKERFILE:-Dockerfile}

echo "[build] Building image ${IMAGE_NAME} using ${DOCKERFILE}"
docker build -f "$DOCKERFILE" -t "$IMAGE_NAME" "$SCRIPT_DIR" "$@"
