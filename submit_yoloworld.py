#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
submit_yoloworld.py

- Duyệt các thư mục video (khung ảnh .jpg) trong --frames_root
- Với mỗi VIDEO_ID, đặt lớp YOLO-World theo quy tắc:
    BlackBox_0/BlackBox_1 -> "black box"
    CardboardBox_0        -> "cardboard box"
    (mặc định) CamelCase  -> "camel case" dạng thường (LifeJacket -> "life jacket")
- Mỗi frame: chạy infer, lấy bbox có confidence cao nhất SAU LỌC MÉP ẢNH
- Nếu --use_tracking: làm mượt EMA + dự đoán tiếp khi miss tối đa --track_max_age khung
  (kết quả tracker cũng bị lọc mép; nếu vi phạm, coi như miss)
- Xuất submission.json theo schema yêu cầu
- Tùy chọn lưu visualize

Usage ví dụ:
  No-tracking:
    python submit_yoloworld.py \
      --weights runs/finetune/yoloworld_custom/weights/best.pt \
      --frames_root data/public_test_frames \
      --out_dir out/submit_yw_no_track \
      --conf 0.001 --iou 0.7 --imgsz 640 --filter-box 0.02 --save_vis

  Tracking:
    python submit_yoloworld.py \
      --weights runs/finetune/yoloworld_custom/weights/best.pt \
      --frames_root data/public_test_frames \
      --out_dir out/submit_yw_track \
      --conf 0.001 --iou 0.7 --imgsz 640 \
      --use_tracking --track_alpha 0.6 --track_max_age 5 --track_conf_decay 0.9 \
      --filter-box 0.015 \
      --save_vis
"""

import argparse
import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLOWorld


# -----------------------------
# Logging
# -----------------------------
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("submit_yoloworld")


# -----------------------------
# IO helpers
# -----------------------------
def list_video_dirs(frames_root: Path):
    return [d for d in sorted(frames_root.iterdir()) if d.is_dir()]


def list_frames(video_dir: Path):
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


def draw_bbox(im, bbox, label: str = None):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if label:
        cv2.putText(
            im, label, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    return im


# -----------------------------
# Class prompt resolver
# -----------------------------
SPECIAL_MAP = {
    "BlackBox": ["black box"],
    "CardboardBox": ["cardboard box"],
    "LifeJacket": ["life saver"]
    # Có thể mở rộng thêm nếu cần
}


def camel_to_words(s: str) -> str:
    spaced = re.sub(r"(?<!^)([A-Z])", r" \1", s).strip()
    return spaced.lower()


def video_id_to_prompts(video_id: str):
    base = video_id.split("_")[0]
    if base in SPECIAL_MAP:
        return SPECIAL_MAP[base]
    return [camel_to_words(base)]


# -----------------------------
# Edge filter utils
# -----------------------------
def clip_xyxy_to_image(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    """Clip bbox [x1,y1,x2,y2] vào biên ảnh."""
    xyxy = xyxy.astype(float)
    xyxy[0] = np.clip(xyxy[0], 0, w - 1)
    xyxy[1] = np.clip(xyxy[1], 0, h - 1)
    xyxy[2] = np.clip(xyxy[2], 0, w - 1)
    xyxy[3] = np.clip(xyxy[3], 0, h - 1)
    return xyxy


def _edge_filter_mask(xyxy: np.ndarray, img_w: int, img_h: int, edge_ratio: float) -> np.ndarray:
    """
    Tạo mask True cho các box HỢP LỆ (không sát mép).
    edge_ratio: 0..0.49; ví dụ 0.02 -> yêu cầu cách mép ≥ 2% kích thước ảnh.
    """
    if edge_ratio <= 0.0 or xyxy.size == 0:
        return np.ones((xyxy.shape[0],), dtype=bool)

    mw = edge_ratio * img_w
    mh = edge_ratio * img_h
    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]
    # Hợp lệ nếu cách mép >= margin
    valid = (x1 >= mw) & (y1 >= mh) & (x2 <= (img_w - mw)) & (y2 <= (img_h - mh))
    return valid


def pick_best_bbox_with_edge(res, img_w: int, img_h: int, edge_ratio: float):
    """
    Lấy bbox có confidence cao nhất sau khi:
      - Clip vào biên ảnh
      - Lọc mép theo edge_ratio
    Trả về dict hoặc None:
      { 'conf': float, 'xyxy': np.ndarray(4,), 'cls': int }
    """
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    confs = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)

    # Clip toàn bộ box vào ảnh để tránh số âm / vượt biên
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
    """Kiểm tra 1 bbox có pass edge filter không (sau khi đã clip)."""
    if edge_ratio <= 0.0:
        return True
    xyxy = clip_xyxy_to_image(xyxy, img_w, img_h)
    mw = edge_ratio * img_w
    mh = edge_ratio * img_h
    x1, y1, x2, y2 = xyxy.tolist()
    return (x1 >= mw) and (y1 >= mh) and (x2 <= (img_w - mw)) and (y2 <= (img_h - mh))


# -----------------------------
# Tiny single-object tracker (EMA + linear prediction)
# -----------------------------
class TinySingleTracker:
    def __init__(self, alpha=0.6, max_age=5, conf_decay=0.90):
        self.alpha = float(alpha)
        self.max_age = int(max_age)
        self.conf_decay = float(conf_decay)
        self.reset()

    def reset(self):
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


# -----------------------------
# Device handling & safe set_classes
# -----------------------------
def _normalize_device(dev_arg):
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


def _ensure_txt_feats_on(model, device_str):
    """
    Nếu YOLO-World đã tạo text features (txt_feats), đảm bảo chúng nằm cùng device với model.
    """
    try:
        if hasattr(model, "model"):
            target = torch.device(device_str) if device_str is not None else next(model.model.parameters()).device
            if hasattr(model.model, "txt_feats") and model.model.txt_feats is not None:
                if isinstance(model.model.txt_feats, (list, tuple)):
                    model.model.txt_feats = [t.to(target) for t in model.model.txt_feats]
                else:
                    model.model.txt_feats = model.model.txt_feats.to(target)
    except Exception:
        # Không fail job nếu khác version Ultralytics
        pass


def set_yw_classes_safe(model, prompts, device_str):
    """
    Thiết lập classes cho YOLO-World sao cho text feats + model cùng device.
    """
    # B1: đảm bảo model đang ở đúng device trước khi set
    if device_str is not None:
        model.to(device_str)

    # B2: gọi set_classes (ưu tiên cache_clip_model nếu có)
    try:
        model.set_classes(prompts, cache_clip_model=True)
    except TypeError:
        model.set_classes(prompts)

    # B3: ép txt_feats về cùng device với model
    _ensure_txt_feats_on(model, device_str)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="YOLO-World -> submission.json (+ optional visualize), tracking or not")
    ap.add_argument("--weights", required=True, help="Path YOLO-World weights (e.g., best.pt)")
    ap.add_argument("--frames_root", default="data/public_test_frames", help="Root chứa các thư mục video")
    ap.add_argument("--out_dir", default="out/submission_pred_yw", help="Thư mục output (JSON + visualize)")
    ap.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (nên thấp cho recall)")
    ap.add_argument("--iou", type=float, default=0.7, help="IoU threshold NMS")
    ap.add_argument("--imgsz", type=int, default=640, help="Resize ảnh khi predict")
    ap.add_argument("--filter-box", dest="filter_box", type=float, default=0.0,
                    help="Tỷ lệ biên để loại box sát mép ảnh (0.0..0.49). Ví dụ 0.02 = 2%% kích thước ảnh.")
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:N | mps")
    ap.add_argument("--save_vis", action="store_true", help="Lưu visualize ảnh overlay bbox")
    # Tracking options
    ap.add_argument("--use_tracking", action="store_true", help="Bật tracking mượt (EMA + dự đoán ngắn hạn)")
    ap.add_argument("--track_alpha", type=float, default=0.6, help="EMA alpha (0..1), cao -> bám detection nhiều hơn")
    ap.add_argument("--track_max_age", type=int, default=5, help="Miss tối đa N khung vẫn dự đoán bbox")
    ap.add_argument("--track_conf_decay", type=float, default=0.90, help="Giảm conf mỗi khung khi dự đoán")
    args = ap.parse_args()

    if not (0.0 <= args.filter_box < 0.5):
        raise ValueError("--filter-box phải trong khoảng [0.0, 0.49]")

    logger = setup_logger()

    # Chọn device thống nhất
    dev = _normalize_device(args.device)
    if dev and dev.startswith("cuda:") and torch.cuda.is_available():
        # Đặt current device (giảm khả năng "index_select" mismatch)
        try:
            torch.cuda.set_device(int(dev.split(":")[1]))
        except Exception:
            pass

    # Load model và đưa về dev (một lần duy nhất)
    model = YOLOWorld(args.weights)
    logger.info(f"Loaded YOLO-World model: {args.weights}")
    if dev is not None:
        model.to(dev)

    frames_root = Path(args.frames_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "submission.json"
    vis_root = out_dir / "visualize"
    if args.save_vis:
        vis_root.mkdir(parents=True, exist_ok=True)

    videos = list_video_dirs(frames_root)
    logger.info(f"Found {len(videos)} videos in {frames_root}")

    submission = []

    for vdir in tqdm(videos, desc="Videos"):
        video_id = vdir.name
        prompts = video_id_to_prompts(video_id)

        # set_classes an toàn: đảm bảo txt_feats cùng device với model
        set_yw_classes_safe(model, prompts, dev)
        logger.info(f"[{video_id}] classes={prompts}")

        frames = list_frames(vdir)
        if len(frames) == 0:
            submission.append({"video_id": video_id, "detections": []})
            continue

        if args.save_vis:
            out_vis_dir = vis_root / video_id
            out_vis_dir.mkdir(parents=True, exist_ok=True)

        detections = []
        tracker = TinySingleTracker(
            alpha=args.track_alpha,
            max_age=args.track_max_age,
            conf_decay=args.track_conf_decay
        ) if args.use_tracking else None

        for img_path in tqdm(frames, desc=f"{video_id}", leave=False):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise FileNotFoundError(f"Cannot read image: {img_path}")
                H, W = img.shape[:2]

                # Quan trọng: KHÔNG truyền device=... vào predict; dùng device của model
                results = model.predict(
                    source=img,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    stream=False,
                    verbose=False,
                    save=False
                )
                res = results[0]

                # pick detection sau lọc mép
                best = pick_best_bbox_with_edge(res, W, H, edge_ratio=args.filter_box)

                used_bbox = None
                used_conf = None

                if tracker is None:
                    # No tracking: dùng trực tiếp detection nếu có
                    if best is not None:
                        used_bbox = clip_xyxy_to_image(best["xyxy"], W, H)
                        if passes_edge_filter(used_bbox, W, H, args.filter_box):
                            used_conf = best["conf"]
                        else:
                            used_bbox, used_conf = None, None
                else:
                    # Tracking: update nếu có detection hợp lệ; nếu không -> predict
                    if best is not None:
                        cand = clip_xyxy_to_image(best["xyxy"], W, H)
                        if passes_edge_filter(cand, W, H, args.filter_box):
                            used_bbox, used_conf = tracker.update(cand, best["conf"])
                        else:
                            used_bbox, used_conf = tracker.predict()
                    else:
                        used_bbox, used_conf = tracker.predict()

                    # Với bbox từ tracker (update/predict), clip + edge check
                    if used_bbox is not None:
                        used_bbox = clip_xyxy_to_image(used_bbox, W, H)
                        if not passes_edge_filter(used_bbox, W, H, args.filter_box):
                            used_bbox, used_conf = None, None

                # Append to submission (chỉ khi có bbox)
                if used_bbox is not None:
                    x1, y1, x2, y2 = used_bbox.tolist()
                    frame_idx = parse_frame_idx(img_path)
                    detections.append({
                        "frame": frame_idx,
                        "x1": int(round(x1)),
                        "y1": int(round(y1)),
                        "x2": int(round(x2)),
                        "y2": int(round(y2)),
                    })

                # Visualize
                if args.save_vis:
                    vis_im = img.copy()
                    if used_bbox is not None:
                        label = f"{used_conf:.2f}" if used_conf is not None else None
                        vis_im = draw_bbox(vis_im, used_bbox, label)
                    else:
                        cv2.putText(
                            vis_im, "No det (edge-filtered)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
                        )
                    cv2.imwrite(str((out_vis_dir / img_path.name)), vis_im)

            except Exception as e:
                logger.warning(f"{img_path}: {e}")

        # Add to submission
        if len(detections) == 0:
            submission.append({"video_id": video_id, "detections": []})
        else:
            submission.append({
                "video_id": video_id,
                "detections": [{"bboxes": detections}]
            })

    # Save JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Saved submission: {out_json}")
    if args.save_vis:
        logger.info(f"🖼️ Visualizations saved in: {vis_root}")


if __name__ == "__main__":
    main()
