#!/usr/bin/env bash
set -euo pipefail

# Flexible wrapper around inference.py.
# Override defaults with environment variables (e.g. WEIGHTS, FRAMES_ROOT).
#Docker version
# EXTRACT_FRAMES=${EXTRACT_FRAMES:-false}
# EXTRACT_SCRIPT=${EXTRACT_SCRIPT:-/code/extract_frame.sh}

# WEIGHTS=${WEIGHTS:-/code/checkpoint/v8s_640_bs_16_30eps_v3/weights/best.pt}
# FRAMES_ROOT=${FRAMES_ROOT:-/data/extracted_frames}
# OUT_DIR=${OUT_DIR:-/result}

#Local version
EXTRACT_FRAMES=${EXTRACT_FRAMES:-true}
EXTRACT_SCRIPT=${EXTRACT_SCRIPT:-code/extract_frame.sh}

WEIGHTS=${WEIGHTS:-code/checkpoint/v8s_640_bs_16_30eps_v3/weights/best.pt}
FRAMES_ROOT=${FRAMES_ROOT:-data/extracted_frames}
OUT_DIR=${OUT_DIR:-result}
IMG_SIZE=${IMG_SIZE:-640}
CONF=${CONF:-0.3}
IOU=${IOU:-0.45}
DEVICE=${DEVICE:-0}
SAVE_VIS=${SAVE_VIS:-false}

echo "[predict] Configuration"
echo "  WEIGHTS     = ${WEIGHTS}"
echo "  FRAMES_ROOT = ${FRAMES_ROOT}"
echo "  OUT_DIR     = ${OUT_DIR}"
echo "  IMG_SIZE    = ${IMG_SIZE}"
echo "  CONF/IOU    = ${CONF}/${IOU}"
echo "  DEVICE      = ${DEVICE:-<unset>}"
echo "  SAVE_VIS    = ${SAVE_VIS}"
echo "  EXTRACT     = ${EXTRACT_FRAMES}"

if [[ "${EXTRACT_FRAMES}" == "true" ]]; then
  if [[ ! -x "${EXTRACT_SCRIPT}" ]]; then
    echo "[predict] Extract script not found or not executable: ${EXTRACT_SCRIPT}" >&2
    exit 1
  fi
  echo "[predict] Extracting frames via ${EXTRACT_SCRIPT}"
  bash "${EXTRACT_SCRIPT}"
fi

# CMD=(python /code/predict.py
#     --weights "${WEIGHTS}"
#     --frames_root "${FRAMES_ROOT}"
#     --out_dir "${OUT_DIR}"
#     --imgsz "${IMG_SIZE}"
#     --conf "${CONF}"
#     --iou "${IOU}"
# )

CMD=(python code/predict.py
    --weights "${WEIGHTS}"
    --frames_root "${FRAMES_ROOT}"
    --out_dir "${OUT_DIR}"
    --imgsz "${IMG_SIZE}"
    --conf "${CONF}"
    --iou "${IOU}"
)

if [[ -n "${DEVICE}" ]]; then
  CMD+=(--device "${DEVICE}")
fi

if [[ "${SAVE_VIS}" == "true" ]]; then
  CMD+=(--save_vis)
fi

echo "[predict] Running: ${CMD[*]}"
"${CMD[@]}"
