#!/usr/bin/env python
"""
YOLO inference trên các frame video -> tạo submission.json + visualize (tùy chọn)
Chỉ lấy bbox có confidence cao nhất mỗi frame.
Hỗ trợ:
- Infer full-frame
- Infer 4 tile + WBF (weighted box fusion) với --wbf-split
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
from ensemble_boxes import weighted_boxes_fusion  # NEW


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


def infer_full_frame(
    img_path: Path,
    im: np.ndarray,
    model: YOLO,
    args: argparse.Namespace
):
    """Infer trực tiếp trên full frame (logic cũ)."""
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
    return results


def infer_wbf_split(
    img_path: Path,
    im: np.ndarray,
    model: YOLO,
    args: argparse.Namespace,
    logger: logging.Logger
):
    """
    Chia ảnh thành 4 tile (2x2), infer batch 4 tile,
    sau đó dùng Weighted Boxes Fusion để merge bbox về global-image.
    """
    h, w = im.shape[:2]
    if h <= 0 or w <= 0:
        logger.warning(f"{img_path.name}: invalid image size h={h}, w={w}")
        return None

    # Tọa độ tile: (x1, y1, x2, y2)
    mid_x = w // 2
    mid_y = h // 2
    tiles_coords = [
        (0,      0,      mid_x, mid_y),  # top-left
        (mid_x,  0,      w,     mid_y),  # top-right
        (0,      mid_y,  mid_x, h),      # bottom-left
        (mid_x,  mid_y,  w,     h),      # bottom-right
    ]

    # Cắt tile
    tile_imgs = []
    for (x1, y1, x2, y2) in tiles_coords:
        tile = im[y1:y2, x1:x2]
        tile_imgs.append(tile)

    # Infer batch 4 tile cùng lúc
    results_list = model.predict(
        source=tile_imgs,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        stream=False,
        verbose=False,
        save=False
    )

    # Chuẩn bị data cho WBF
    boxes_list: List[List[List[float]]] = []
    scores_list: List[List[float]] = []
    labels_list: List[List[int]] = []

    for res, (tx1, ty1, tx2, ty2) in zip(results_list, tiles_coords):
        if res.boxes is None or len(res.boxes) == 0:
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])
            continue

        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        # nếu không quan tâm class thì cứ cho tất cả = 0
        if res.boxes.cls is not None:
            clses = res.boxes.cls.cpu().numpy().astype(int)
        else:
            clses = np.zeros_like(confs, dtype=int)

        tile_boxes_norm = []
        tile_scores = []
        tile_labels = []

        tile_w = tx2 - tx1
        tile_h = ty2 - ty1
        if tile_w <= 0 or tile_h <= 0:
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])
            continue

        for (x1, y1, x2, y2), c, lab in zip(xyxy, confs, clses):
            # box local tile -> box global pixel
            gx1 = x1 + tx1
            gy1 = y1 + ty1
            gx2 = x2 + tx1
            gy2 = y2 + ty1

            # normalize về [0,1] theo full-image (WBF yêu cầu)
            nx1 = gx1 / w
            ny1 = gy1 / h
            nx2 = gx2 / w
            ny2 = gy2 / h

            tile_boxes_norm.append([nx1, ny1, nx2, ny2])
            tile_scores.append(float(c))
            tile_labels.append(int(lab))

        boxes_list.append(tile_boxes_norm)
        scores_list.append(tile_scores)
        labels_list.append(tile_labels)

    # Nếu tất cả đều rỗng -> không có detection
    if all(len(b) == 0 for b in boxes_list):
        return None

    # Weighted Boxes Fusion
    # iou_thr: dùng lại args.iou
    # skip_box_thr: dùng lại args.conf
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=None,
        iou_thr=args.iou,
        skip_box_thr=args.conf
    )

    if boxes is None or len(boxes) == 0:
        return None

    # Chọn box có score cao nhất
    best_idx = int(np.argmax(scores))
    best_box = boxes[best_idx]  # [x1,y1,x2,y2] normalized
    best_score = float(scores[best_idx])

    # Convert về pixel theo full-image
    x1 = best_box[0] * w
    y1 = best_box[1] * h
    x2 = best_box[2] * w
    y2 = best_box[3] * h

    class DummyBoxes:
        """Wrapper đơn giản để giả lập results.boxes như output YOLO."""
        def __init__(self, xyxy, conf):
            self.xyxy = xyxy
            self.conf = conf

    class DummyResult:
        def __init__(self, boxes):
            self.boxes = boxes

    # Tạo 'results' tương thích với logic cũ
    xyxy_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    conf_np = np.array([best_score], dtype=np.float32)
    boxes = DummyBoxes(xyxy=xyxy_np, conf=conf_np)
    results = DummyResult(boxes=boxes)
    return results


def process_video(
    video_dir: Path,
    model: YOLO,
    args: argparse.Namespace,
    vis_dir: Optional[Path],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Xử lý một video: inference từng frame -> detections + visualize (nếu cần).
    Hỗ trợ:
      - Full frame
      - 4 tile + WBF (--wbf-split)
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
            im = cv2.imread(str(img_path))
            if im is None:
                logger.warning(f"Cannot read image: {img_path}")
                continue

            # Chọn mode infer
            if args.wbf_split:
                results = infer_wbf_split(img_path, im, model, args, logger)
            else:
                results = infer_full_frame(img_path, im, model, args)

            # Không có detection
            if results is None or results.boxes is None:
                if args.save_vis and video_vis_dir:
                    # Lưu ảnh gốc nếu muốn visualize mà không có bbox
                    cv2.imwrite(str(video_vis_dir / img_path.name), im)
                continue

            # results.boxes giống như YOLO output
            if getattr(results.boxes, "xyxy", None) is None or len(results.boxes.xyxy) == 0:
                if args.save_vis and video_vis_dir:
                    cv2.imwrite(str(video_vis_dir / img_path.name), im)
                continue

            # Với full-frame: có thể có nhiều bbox -> lấy bbox conf cao nhất
            # Với WBF: đã tạo sẵn chỉ 1 bbox, nhưng code dưới vẫn hoạt động bình thường.
            confs = results.boxes.conf
            xyxy = results.boxes.xyxy

            # Đảm bảo là numpy
            if not isinstance(confs, np.ndarray):
                confs = confs.cpu().numpy()
            if not isinstance(xyxy, np.ndarray):
                xyxy = xyxy.cpu().numpy()

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
    parser.add_argument("--iou", type=float, default=0.45, help="Ngưỡng IoU cho NMS/WBF")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu', 'cuda', '0', ...")
    parser.add_argument("--save_vis", action="store_true",
                        help="Lưu ảnh visualize có bbox")
    parser.add_argument("--wbf-split", action="store_true",
                        help="Chia ảnh thành 4 tile, infer batch 4 tile + Weighted Boxes Fusion")  # NEW
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
    if args.wbf_split:
        logger.info("WBF mode: 4-tile split + weighted_boxes_fusion enabled.")

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
