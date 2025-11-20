#!/usr/bin/env python
"""
YOLO inference trên các frame video -> tạo submission.json + visualize (tùy chọn).
Chỉ lấy bbox có confidence cao nhất mỗi frame.

Streaming requirement: define a model class with predict_streaming(frame_rgb_np, frame_idx)
that returns [x1, y1, x2, y2] if an object is found, otherwise None.
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence, Tuple

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


@dataclass
class StreamingConfig:
    weights: str
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    device: Optional[str] = None
    save_vis: bool = False


class StreamingYOLO:
    """
    Wrapper cho YOLO với giao diện predict_streaming.
    Sử dụng lại cùng model cho streaming hoặc batch inference.
    """

    def __init__(self, cfg: StreamingConfig, logger: logging.Logger, vis_root: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.model = YOLO(cfg.weights)
        self.vis_root = vis_root if cfg.save_vis else None

    def _select_best_bbox(self, results: Any) -> Optional[Tuple[int, int, int, int]]:
        """Chọn bbox có confidence cao nhất trong kết quả YOLO."""
        if results.boxes is None or len(results.boxes) == 0:
            return None
        confs = results.boxes.conf.cpu().numpy()
        xyxy = results.boxes.xyxy.cpu().numpy()
        best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = xyxy[best_idx]
        return int(x1), int(y1), int(x2), int(y2)

    def predict_streaming(self, frame_rgb_np: np.ndarray, frame_idx: int) -> Optional[Sequence[int]]:
        """
        Giả lập môi trường streaming: nhận frame RGB (numpy) và trả về bbox tốt nhất hoặc None.
        - frame_rgb_np: ảnh RGB dạng numpy array.
        - frame_idx: chỉ số frame (dùng khi log / visualize).
        """
        if frame_rgb_np is None or frame_rgb_np.size == 0:
            self.logger.warning("Empty frame received at idx %s", frame_idx)
            return None

        frame_bgr = frame_rgb_np[:, :, ::-1]  # YOLO nhận BGR (OpenCV style)
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.cfg.device,
            stream=False,
            verbose=False,
            save=False,
        )[0]

        best_bbox = self._select_best_bbox(results)
        if best_bbox is None:
            return None
        return list(best_bbox)

    def predict_image_file(
        self,
        img_path: Path,
        frame_idx: int,
        video_vis_dir: Optional[Path] = None,
    ) -> Optional[Dict[str, int]]:
        """Inference trên 1 file ảnh, kèm visualize nếu cần (không đi qua predict_streaming)."""
        results = self.model.predict(
            source=str(img_path),
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.cfg.device,
            stream=False,
            verbose=False,
            save=False,
        )[0]

        im_bgr = cv2.imread(str(img_path))
        if im_bgr is None:
            self.logger.warning("Cannot read image: %s", img_path)
            return None

        should_save_original = self.cfg.save_vis and (not results.boxes or len(results.boxes) == 0)
        if should_save_original and video_vis_dir:
            cv2.imwrite(str(video_vis_dir / img_path.name), im_bgr)
            return None

        if results.boxes is None or len(results.boxes) == 0:
            return None

        best_bbox = self._select_best_bbox(results)
        if best_bbox is None:
            return None

        x1, y1, x2, y2 = best_bbox
        if self.cfg.save_vis and video_vis_dir:
            im_vis = draw_bbox(im_bgr.copy(), (x1, y1, x2, y2))
            cv2.imwrite(str(video_vis_dir / img_path.name), im_vis)

        return {"frame": frame_idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def process_video(
    video_dir: Path,
    streamer: StreamingYOLO,
    vis_dir: Optional[Path],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Xử lý một video: inference từng frame -> detections + visualize (nếu cần).
    """
    video_id = video_dir.name
    frames = list_frames(video_dir)
    detections: List[Dict[str, int]] = []

    if len(frames) == 0:
        return {"video_id": video_id, "detections": []}

    video_vis_dir = vis_dir / video_id if vis_dir else None
    if video_vis_dir:
        video_vis_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(frames, desc=f"  └ {video_id}", leave=False):
        frame_idx = parse_frame_idx(img_path)
        try:
            det = streamer.predict_image_file(img_path, frame_idx, video_vis_dir)
            if det:
                detections.append(det)
        except Exception as e:
            logger.warning("%s: %s", img_path.name, e)

    if not detections:
        return {"video_id": video_id, "detections": []}
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

    cfg = StreamingConfig(
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_vis=args.save_vis,
    )

    # Load model
    logger.info("Loading model: %s", args.weights)
    streamer = StreamingYOLO(cfg, logger, vis_root)

    # List videos
    videos = list_video_dirs(frames_root)
    logger.info(f"Found {len(videos)} videos in {frames_root}")

    # Process
    submission: List[Dict[str, Any]] = []
    for vdir in tqdm(videos, desc="Processing videos"):
        result = process_video(vdir, streamer, vis_root, logger)
        submission.append(result)

    # Save submission
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Submission saved: {out_json}")
    if args.save_vis:
        logger.info(f"✅ Visualizations saved in: {vis_root}")


if __name__ == "__main__":
    main()
