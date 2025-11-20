#!/usr/bin/env bash
set -euo pipefail

# Flexible wrapper around predict_yoloworld.py.
# Override defaults via env vars (e.g. WEIGHTS, FRAMES_ROOT).

EXTRACT_FRAMES=${EXTRACT_FRAMES:-true}
EXTRACT_SCRIPT=${EXTRACT_SCRIPT:-/code/extract_frame.sh}

WEIGHTS=${WEIGHTS:-/code/checkpoint/yoloworld_no_bg_v8s_20eps_bs32_exp2/weights/best.pt}
FRAMES_ROOT=${FRAMES_ROOT:-/data/extracted_frames}
OUT_DIR=${OUT_DIR:-/result}
IMG_SIZE=${IMG_SIZE:-640}
CONF=${CONF:-0.001}
IOU=${IOU:-0.7}
FILTER_BOX=${FILTER_BOX:-0.015}
DEVICE=${DEVICE:-0}
SAVE_VIS=${SAVE_VIS:-false}
USE_TRACKING=${USE_TRACKING:-false}
TRACK_ALPHA=${TRACK_ALPHA:-0.6}
TRACK_MAX_AGE=${TRACK_MAX_AGE:-5}
TRACK_CONF_DECAY=${TRACK_CONF_DECAY:-0.90}

echo "[predict_yoloworld] Configuration"
echo "  WEIGHTS      = ${WEIGHTS}"
echo "  FRAMES_ROOT  = ${FRAMES_ROOT}"
echo "  OUT_DIR      = ${OUT_DIR}"
echo "  IMG_SIZE     = ${IMG_SIZE}"
echo "  CONF/IOU     = ${CONF}/${IOU}"
echo "  FILTER_BOX   = ${FILTER_BOX}"
echo "  DEVICE       = ${DEVICE:-<unset>}"
echo "  SAVE_VIS     = ${SAVE_VIS}"
echo "  USE_TRACKING = ${USE_TRACKING}"
echo "  TRACK_ALPHA  = ${TRACK_ALPHA}"
echo "  TRACK_MAX_AGE= ${TRACK_MAX_AGE}"
echo "  TRACK_DECAY  = ${TRACK_CONF_DECAY}"
echo "  EXTRACT      = ${EXTRACT_FRAMES}"

if [[ "${EXTRACT_FRAMES}" == "true" ]]; then
  if [[ ! -x "${EXTRACT_SCRIPT}" ]]; then
    echo "[predict_yoloworld] Extract script not found or not executable: ${EXTRACT_SCRIPT}" >&2
    exit 1
  fi
  echo "[predict_yoloworld] Extracting frames via ${EXTRACT_SCRIPT}"
  bash "${EXTRACT_SCRIPT}"
fi

CMD=(python /code/predict_yoloworld.py
    --weights "${WEIGHTS}"
    --frames_root "${FRAMES_ROOT}"
    --out_dir "${OUT_DIR}"
    --imgsz "${IMG_SIZE}"
    --conf "${CONF}"
    --iou "${IOU}"
    --filter-box "${FILTER_BOX}"
)

if [[ -n "${DEVICE}" ]]; then
  CMD+=(--device "${DEVICE}")
fi
if [[ "${SAVE_VIS}" == "true" ]]; then
  CMD+=(--save_vis)
fi
if [[ "${USE_TRACKING}" == "true" ]]; then
  CMD+=(--use_tracking --track_alpha "${TRACK_ALPHA}" --track_max_age "${TRACK_MAX_AGE}" --track_conf_decay "${TRACK_CONF_DECAY}")
fi

echo "[predict_yoloworld] Running: ${CMD[*]}"
"${CMD[@]}"
