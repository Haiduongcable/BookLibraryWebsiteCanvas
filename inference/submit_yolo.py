#!/usr/bin/env python
"""
YOLO inference trên các frame video -> tạo submission.json + visualize (tùy chọn)
Chỉ lấy bbox có confidence cao nhất mỗi frame.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def setup_logger() -> logging.Logger:
    """Cấu hình logger với format đẹp."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger("predict2sub_vis")


def list_video_dirs(frames_root: Path) -> List[Path]:
    """Liệt kê tất cả thư mục video (con của frames_root)."""
    return [d for d in sorted(frames_root.iterdir()) if d.is_dir()]


def list_frames(video_dir: Path) -> List[Path]:
    """Liệt kê tất cả file .jpg trong thư mục video."""
    return sorted(video_dir.glob("*.jpg"))


def parse_frame_idx(img_path: Path) -> int:
    """
    Trích xuất chỉ số frame từ tên file.
    Hỗ trợ: xxx_frame_123.jpg hoặc xxx123.jpg
    """
    stem = img_path.stem

    # Trường hợp: xxx_frame_123
    if "_frame_" in stem:
        try:
            return int(stem.split("_frame_")[-1])
        except ValueError:
            pass

    # Trường hợp: xxx123 (lấy dãy số cuối)
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else -1


def draw_bbox(im: np.ndarray, bbox: tuple, conf: Optional[float] = None) -> np.ndarray:
    """Vẽ 1 bbox (x1,y1,x2,y2) lên ảnh BGR."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if conf is not None:
        label = f"{conf:.2f}"
        cv2.putText(
            im, label, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    return im


def process_video(
    video_dir: Path,
    model: YOLO,
    args: argparse.Namespace,
    vis_dir: Optional[Path],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Xử lý một video: inference từng frame -> detections + visualize (nếu cần).
    """
    video_id = video_dir.name
    frames = list_frames(video_dir)
    detections: List[Dict[str, int]] = []

    if len(frames) == 0:
        return {"video_id": video_id, "detections": []}

    # Tạo thư mục visualize cho video (nếu cần)
    video_vis_dir = vis_dir / video_id if vis_dir else None
    if video_vis_dir:
        video_vis_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(frames, desc=f"  └ {video_id}", leave=False):
        try:
            # Inference
            results = model.predict(
                source=str(img_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                stream=False,
                verbose=False,
                save=False
            )[0]

            im = cv2.imread(str(img_path))
            if im is None:
                continue

            # Lưu ảnh gốc nếu không có detection (và cần visualize)
            should_save_original = args.save_vis and (not results.boxes or len(results.boxes) == 0)

            if should_save_original and video_vis_dir:
                cv2.imwrite(str(video_vis_dir / img_path.name), im)
                continue

            if results.boxes is None or len(results.boxes) == 0:
                continue

            # Lấy bbox có conf cao nhất
            confs = results.boxes.conf.cpu().numpy()
            xyxy = results.boxes.xyxy.cpu().numpy()
            best_idx = int(np.argmax(confs))
            x1, y1, x2, y2 = xyxy[best_idx]
            conf = float(confs[best_idx])
            frame_idx = parse_frame_idx(img_path)

            detections.append({
                "frame": frame_idx,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })

            # Visualize
            if args.save_vis and video_vis_dir:
                im_vis = draw_bbox(im.copy(), (x1, y1, x2, y2), conf)
                cv2.imwrite(str(video_vis_dir / img_path.name), im_vis)

        except Exception as e:
            logger.warning(f"{img_path.name}: {e}")

    # Định dạng submission
    if not detections:
        return {"video_id": video_id, "detections": []}
    else:
        return {"video_id": video_id, "detections": [{"bboxes": detections}]}


def main() -> None:
    # Argument parser
    parser = argparse.ArgumentParser(
        description="YOLO inference (serial) → submission.json + visualize"
    )
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file weights (.pt)")
    parser.add_argument("--frames_root", type=str, default="data/public_test_frames",
                        help="Thư mục chứa các thư mục frame video")
    parser.add_argument("--out_dir", type=str, default="out/submission_pred",
                        help="Thư mục lưu submission.json và visualize")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence")
    parser.add_argument("--iou", type=float, default=0.45, help="Ngưỡng IoU cho NMS")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu', 'cuda', '0', ...")
    parser.add_argument("--save_vis", action="store_true",
                        help="Lưu ảnh visualize có bbox")
    args = parser.parse_args()

    # Setup
    logger = setup_logger()
    frames_root = Path(args.frames_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "submission.json"
    vis_root = out_dir / "visualize" if args.save_vis else None
    if vis_root:
        vis_root.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # List videos
    videos = list_video_dirs(frames_root)
    logger.info(f"Found {len(videos)} videos in {frames_root}")

    # Process
    submission: List[Dict[str, Any]] = []
    for vdir in tqdm(videos, desc="Processing videos"):
        result = process_video(vdir, model, args, vis_root, logger)
        submission.append(result)

    # Save submission
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Submission saved: {out_json}")
    if args.save_vis:
        logger.info(f"✅ Visualizations saved in: {vis_root}")


if __name__ == "__main__":
    main()