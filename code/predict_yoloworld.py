#!/usr/bin/env python3
"""
YOLO-World inference trên các frame -> submission.json + visualize (tùy chọn).
Hỗ trợ:
  - Tự động map video_id -> class prompt (BlackBox/CardboardBox/...).
  - Lọc bbox sát mép ảnh.
  - Tracker EMA + dự đoán ngắn hạn (tùy chọn).

Format & luồng tổng thể dựa trên predict.py / predict_notebook.ipynb.
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLOWorld


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("predict_yoloworld")


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------
def list_video_dirs(frames_root: Path) -> List[Path]:
    return [d for d in sorted(frames_root.iterdir()) if d.is_dir()]


def list_frames(video_dir: Path) -> List[Path]:
    return sorted(video_dir.glob("*.jpg"))


def parse_frame_idx(img_path: Path) -> int:
    stem = img_path.stem
    if "_frame_" in stem:
        try:
            return int(stem.split("_frame_")[-1])
        except Exception:
            pass
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else -1


def draw_bbox(im: np.ndarray, bbox: Sequence[int], label: Optional[str] = None) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if label:
        cv2.putText(
            im, label, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    return im


# ----------------------------------------------------------------------
# Prompt resolver
# ----------------------------------------------------------------------
SPECIAL_MAP = {
    "BlackBox": ["black box"],
    "CardboardBox": ["cardboard box"],
    "LifeJacket": ["life saver"],
}


def camel_to_words(s: str) -> str:
    spaced = re.sub(r"(?<!^)([A-Z])", r" \1", s).strip()
    return spaced.lower()


def video_id_to_prompts(video_id: str) -> List[str]:
    base = video_id.split("_")[0]
    if base in SPECIAL_MAP:
        return SPECIAL_MAP[base]
    return [camel_to_words(base)]


# ----------------------------------------------------------------------
# Edge filter helpers
# ----------------------------------------------------------------------
def clip_xyxy_to_image(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    xyxy = xyxy.astype(float)
    xyxy[0] = np.clip(xyxy[0], 0, w - 1)
    xyxy[1] = np.clip(xyxy[1], 0, h - 1)
    xyxy[2] = np.clip(xyxy[2], 0, w - 1)
    xyxy[3] = np.clip(xyxy[3], 0, h - 1)
    return xyxy


def _edge_filter_mask(xyxy: np.ndarray, img_w: int, img_h: int, edge_ratio: float) -> np.ndarray:
    if edge_ratio <= 0.0 or xyxy.size == 0:
        return np.ones((xyxy.shape[0],), dtype=bool)
    mw = edge_ratio * img_w
    mh = edge_ratio * img_h
    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]
    valid = (x1 >= mw) & (y1 >= mh) & (x2 <= (img_w - mw)) & (y2 <= (img_h - mh))
    return valid


def pick_best_bbox_with_edge(res: Any, img_w: int, img_h: int, edge_ratio: float) -> Optional[Dict[str, Any]]:
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    confs = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)

    for i in range(xyxy.shape[0]):
        xyxy[i] = clip_xyxy_to_image(xyxy[i], img_w, img_h)

    valid_mask = _edge_filter_mask(xyxy, img_w, img_h, edge_ratio)
    if not np.any(valid_mask):
        return None

    confs_v = confs[valid_mask]
    xyxy_v = xyxy[valid_mask]
    cls_v = cls_ids[valid_mask]
    best_i = int(np.argmax(confs_v))
    return {"conf": float(confs_v[best_i]), "xyxy": xyxy_v[best_i], "cls": int(cls_v[best_i])}


def passes_edge_filter(xyxy: np.ndarray, img_w: int, img_h: int, edge_ratio: float) -> bool:
    if edge_ratio <= 0.0:
        return True
    xyxy = clip_xyxy_to_image(xyxy, img_w, img_h)
    mw = edge_ratio * img_w
    mh = edge_ratio * img_h
    x1, y1, x2, y2 = xyxy.tolist()
    return (x1 >= mw) and (y1 >= mh) and (x2 <= (img_w - mw)) and (y2 <= (img_h - mh))


# ----------------------------------------------------------------------
# Tiny single-object tracker (EMA + short-term predict)
# ----------------------------------------------------------------------
class TinySingleTracker:
    def __init__(self, alpha: float = 0.6, max_age: int = 5, conf_decay: float = 0.90) -> None:
        self.alpha = float(alpha)
        self.max_age = int(max_age)
        self.conf_decay = float(conf_decay)
        self.reset()

    def reset(self) -> None:
        self.has_state = False
        self.bbox = None
        self.prev_bbox = None
        self.velocity = None
        self.conf = None
        self.missed = 0

    def update(self, det_bbox: np.ndarray, det_conf: float):
        det_bbox = det_bbox.astype(float)
        if not self.has_state:
            self.bbox = det_bbox
            self.prev_bbox = det_bbox
            self.velocity = np.zeros(4, dtype=float)
            self.conf = det_conf
            self.missed = 0
            self.has_state = True
            return self.bbox, self.conf

        new_bbox = self.alpha * det_bbox + (1.0 - self.alpha) * self.bbox
        self.velocity = new_bbox - self.bbox
        self.prev_bbox = self.bbox
        self.bbox = new_bbox
        self.conf = 0.5 * det_conf + 0.5 * (self.conf if self.conf is not None else det_conf)
        self.missed = 0
        return self.bbox, self.conf

    def predict(self):
        if not self.has_state:
            return None, None
        if self.missed >= self.max_age:
            self.reset()
            return None, None
        self.prev_bbox = self.bbox
        self.bbox = self.bbox + (self.velocity if self.velocity is not None else 0.0)
        if self.conf is None:
            self.conf = 0.0
        else:
            self.conf *= self.conf_decay
        self.missed += 1
        return self.bbox, self.conf


# ----------------------------------------------------------------------
# Device handling & safe set_classes
# ----------------------------------------------------------------------
def _normalize_device(dev_arg: Optional[str]) -> Optional[str]:
    if dev_arg is None:
        return None
    if isinstance(dev_arg, str):
        s = dev_arg.strip().lower()
        if s in ["cpu", "mps", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]:
            return s
        if s.isdigit():
            return f"cuda:{s}" if torch.cuda.is_available() else "cpu"
        return s
    if isinstance(dev_arg, int):
        return f"cuda:{dev_arg}" if torch.cuda.is_available() else "cpu"
    return None


def _ensure_txt_feats_on(model: YOLOWorld, device_str: Optional[str]) -> None:
    try:
        if hasattr(model, "model"):
            target = torch.device(device_str) if device_str is not None else next(model.model.parameters()).device
            if hasattr(model.model, "txt_feats") and model.model.txt_feats is not None:
                if isinstance(model.model.txt_feats, (list, tuple)):
                    model.model.txt_feats = [t.to(target) for t in model.model.txt_feats]
                else:
                    model.model.txt_feats = model.model.txt_feats.to(target)
    except Exception:
        pass


def set_yw_classes_safe(model: YOLOWorld, prompts: List[str], device_str: Optional[str]) -> None:
    if device_str is not None:
        model.to(device_str)
    try:
        model.set_classes(prompts, cache_clip_model=True)
    except TypeError:
        model.set_classes(prompts)
    _ensure_txt_feats_on(model, device_str)


# ----------------------------------------------------------------------
# Config + Predictor
# ----------------------------------------------------------------------
@dataclass
class YoloWorldConfig:
    weights: str
    imgsz: int = 640
    conf: float = 0.001
    iou: float = 0.7
    filter_box: float = 0.0
    device: Optional[str] = None
    save_vis: bool = False
    use_tracking: bool = False
    track_alpha: float = 0.6
    track_max_age: int = 5
    track_conf_decay: float = 0.90


class YoloWorldPredictor:
    """Wrapper cho YOLO-World phục vụ cả batch inference và tracking tùy chọn."""

    def __init__(self, cfg: YoloWorldConfig, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.device = _normalize_device(cfg.device)

        if self.device and self.device.startswith("cuda:") and torch.cuda.is_available():
            try:
                torch.cuda.set_device(int(self.device.split(":")[1]))
            except Exception:
                pass

        self.model = YOLOWorld(cfg.weights)
        if self.device is not None:
            self.model.to(self.device)
        self.logger.info("Loaded YOLO-World model: %s", cfg.weights)

    def prepare_video(self, video_id: str) -> List[str]:
        prompts = video_id_to_prompts(video_id)
        set_yw_classes_safe(self.model, prompts, self.device)
        self.logger.info("[%s] classes=%s", video_id, prompts)
        return prompts

    def build_tracker(self) -> Optional[TinySingleTracker]:
        if not self.cfg.use_tracking:
            return None
        return TinySingleTracker(
            alpha=self.cfg.track_alpha,
            max_age=self.cfg.track_max_age,
            conf_decay=self.cfg.track_conf_decay,
        )

    def _predict_raw(self, img_bgr: np.ndarray):
        return self.model.predict(
            source=img_bgr,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            stream=False,
            verbose=False,
            save=False,
        )[0]

    def predict_frame(
        self,
        img_path: Path,
        tracker: Optional[TinySingleTracker],
        video_vis_dir: Optional[Path],
    ) -> Optional[Dict[str, int]]:
        img = cv2.imread(str(img_path))
        if img is None:
            self.logger.warning("Cannot read image: %s", img_path)
            return None
        H, W = img.shape[:2]

        res = self._predict_raw(img)
        best = pick_best_bbox_with_edge(res, W, H, edge_ratio=self.cfg.filter_box)

        used_bbox = None
        used_conf = None

        if tracker is None:
            if best is not None:
                cand = clip_xyxy_to_image(best["xyxy"], W, H)
                if passes_edge_filter(cand, W, H, self.cfg.filter_box):
                    used_bbox = cand
                    used_conf = best["conf"]
        else:
            if best is not None:
                cand = clip_xyxy_to_image(best["xyxy"], W, H)
                if passes_edge_filter(cand, W, H, self.cfg.filter_box):
                    used_bbox, used_conf = tracker.update(cand, best["conf"])
                else:
                    used_bbox, used_conf = tracker.predict()
            else:
                used_bbox, used_conf = tracker.predict()

            if used_bbox is not None:
                used_bbox = clip_xyxy_to_image(used_bbox, W, H)
                if not passes_edge_filter(used_bbox, W, H, self.cfg.filter_box):
                    used_bbox, used_conf = None, None

        if used_bbox is None:
            if self.cfg.save_vis and video_vis_dir:
                vis_im = img.copy()
                cv2.putText(
                    vis_im, "No det (edge-filtered)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
                )
                cv2.imwrite(str(video_vis_dir / img_path.name), vis_im)
            return None

        x1, y1, x2, y2 = used_bbox.tolist()
        if self.cfg.save_vis and video_vis_dir:
            label = f"{used_conf:.2f}" if used_conf is not None else None
            vis_im = draw_bbox(img.copy(), used_bbox, label)
            cv2.imwrite(str(video_vis_dir / img_path.name), vis_im)

        return {
            "frame": parse_frame_idx(img_path),
            "x1": int(round(x1)),
            "y1": int(round(y1)),
            "x2": int(round(x2)),
            "y2": int(round(y2)),
        }


# ----------------------------------------------------------------------
# Video runner
# ----------------------------------------------------------------------
def process_video(
    video_dir: Path,
    predictor: YoloWorldPredictor,
    vis_root: Optional[Path],
    logger: logging.Logger,
) -> Dict[str, Any]:
    video_id = video_dir.name
    frames = list_frames(video_dir)
    detections: List[Dict[str, int]] = []

    if len(frames) == 0:
        return {"video_id": video_id, "detections": []}

    video_vis_dir = vis_root / video_id if vis_root else None
    if video_vis_dir:
        video_vis_dir.mkdir(parents=True, exist_ok=True)

    predictor.prepare_video(video_id)
    tracker = predictor.build_tracker()

    for img_path in tqdm(frames, desc=f"  └ {video_id}", leave=False):
        try:
            det = predictor.predict_frame(img_path, tracker, video_vis_dir)
            if det:
                detections.append(det)
        except Exception as e:
            logger.warning("%s: %s", img_path.name, e)

    if not detections:
        return {"video_id": video_id, "detections": []}
    return {"video_id": video_id, "detections": [{"bboxes": detections}]}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="YOLO-World -> submission.json (+ optional visualize)")
    ap.add_argument("--weights", required=True, help="Path YOLO-World weights (best.pt)")
    ap.add_argument("--frames_root", default="data/public_test_frames", help="Root chứa thư mục video frames")
    ap.add_argument("--out_dir", default="out/submission_pred_yw", help="Thư mục output (JSON + visualize)")
    ap.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="IoU threshold NMS")
    ap.add_argument("--imgsz", type=int, default=640, help="Resize ảnh khi predict")
    ap.add_argument("--filter-box", dest="filter_box", type=float, default=0.0,
                    help="Tỷ lệ biên để loại box sát mép ảnh (0.0..0.49). Ví dụ 0.02 = 2%% kích thước ảnh.")
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:N | mps")
    ap.add_argument("--save_vis", action="store_true", help="Lưu visualize ảnh overlay bbox")
    ap.add_argument("--use_tracking", action="store_true", help="Bật tracking mượt (EMA + dự đoán ngắn hạn)")
    ap.add_argument("--track_alpha", type=float, default=0.6, help="EMA alpha (0..1)")
    ap.add_argument("--track_max_age", type=int, default=5, help="Miss tối đa N khung vẫn dự đoán bbox")
    ap.add_argument("--track_conf_decay", type=float, default=0.90, help="Giảm conf mỗi khung khi dự đoán")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.filter_box < 0.5):
        raise ValueError("--filter-box phải trong khoảng [0.0, 0.49]")

    logger = setup_logger()

    frames_root = Path(args.frames_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "submission.json"
    vis_root = out_dir / "visualize" if args.save_vis else None
    if vis_root:
        vis_root.mkdir(parents=True, exist_ok=True)

    cfg = YoloWorldConfig(
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        filter_box=args.filter_box,
        device=args.device,
        save_vis=args.save_vis,
        use_tracking=args.use_tracking,
        track_alpha=args.track_alpha,
        track_max_age=args.track_max_age,
        track_conf_decay=args.track_conf_decay,
    )
    predictor = YoloWorldPredictor(cfg, logger)

    videos = list_video_dirs(frames_root)
    logger.info("Found %d videos in %s", len(videos), frames_root)

    submission: List[Dict[str, Any]] = []
    for vdir in tqdm(videos, desc="Processing videos"):
        result = process_video(vdir, predictor, vis_root, logger)
        submission.append(result)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    logger.info("✅ Saved submission: %s", out_json)
    if args.save_vis:
        logger.info("🖼️ Visualizations saved in: %s", vis_root)


if __name__ == "__main__":
    main()
