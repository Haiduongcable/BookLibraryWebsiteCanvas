#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLOWorld
import yaml
from tqdm import tqdm


# -----------------------------
# Utils
# -----------------------------
def load_model_and_classes(model_path, data_yaml=None, custom_classes=None):
    """
    Load YOLO-World model và set classes

    Args:
        model_path: Path tới weights đã train
        data_yaml:  Path tới data.yaml (để lấy classes gốc khi train)
        custom_classes: List tên lớp custom (zero-shot / unseen classes)

    Returns:
        (model, class_names)
    """
    print(f"[INFO] Load model: {model_path}")
    model = YOLOWorld(model_path)

    if custom_classes and len(custom_classes) > 0:
        class_names = list(custom_classes)
        print(f"[INFO] Dùng custom classes (inference): {class_names}")
    elif data_yaml:
        with open(data_yaml, "r") as f:
            data_config = yaml.safe_load(f)
        names = data_config.get("names")
        if isinstance(names, dict):
            class_names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        else:
            class_names = list(names)
        print(f"[INFO] Dùng trained classes từ data.yaml: {class_names}")
    else:
        raise ValueError("Cần truyền --data hoặc --classes")

    model.set_classes(class_names)
    return model, class_names


def _edge_filter_mask(xyxy, img_w, img_h, edge_margin_ratio):
    """
    Tạo mask True cho các box HỢP LỆ (không sát mép).
    edge_margin_ratio: tỷ lệ 0..0.49. 0 -> không lọc.
    """
    if edge_margin_ratio <= 0.0:
        return np.ones((xyxy.shape[0],), dtype=bool)

    mw = edge_margin_ratio * img_w
    mh = edge_margin_ratio * img_h
    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]

    # Hợp lệ nếu cách mép >= margin
    valid = (x1 >= mw) & (y1 >= mh) & (x2 <= (img_w - mw)) & (y2 <= (img_h - mh))
    return valid


def pick_best_bbox(result, img_shape, edge_margin_ratio=0.0):
    """
    Từ 1 result của ultralytics, chọn bbox có confidence cao nhất sau khi lọc mép.
    img_shape: (H, W, C)
    Returns:
        best (dict) hoặc None
        {
          'conf': float,
          'xyxy': np.ndarray shape (4,),
          'cls': int
        }
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    confs = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)

    H, W = img_shape[0], img_shape[1]
    valid_mask = _edge_filter_mask(xyxy, W, H, edge_margin_ratio)
    if not np.any(valid_mask):
        return None

    confs_v = confs[valid_mask]
    xyxy_v = xyxy[valid_mask]
    cls_v = cls_ids[valid_mask]

    best_i = int(np.argmax(confs_v))
    return {
        "conf": float(confs_v[best_i]),
        "xyxy": xyxy_v[best_i],
        "cls": int(cls_v[best_i]),
    }


def draw_best_on_image(img_bgr, best, class_names):
    """
    Vẽ bbox tốt nhất lên ảnh (BGR).
    """
    x1, y1, x2, y2 = map(int, best["xyxy"].tolist())
    cls_id = best["cls"]
    conf = best["conf"]
    name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"id={cls_id}"
    label = f"{name} {conf:.2f}"

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    yt = max(0, y1 - 6)
    cv2.putText(
        img_bgr, label, (x1, yt),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    return img_bgr


# -----------------------------
# Predictors
# -----------------------------
def predict_image(model, image_path, class_names, conf_threshold=0.001, iou_threshold=0.7,
                  output_dir=None, visualize=True, imgsz=640, edge_margin_ratio=0.0):
    """
    Chạy inference cho 1 ảnh. Chỉ vẽ và lưu detection có confidence cao nhất.
    """
    im = cv2.imread(str(image_path))
    if im is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    results = model.predict(
        im,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        save=False,      # không để YOLO tự lưu
        show=False,
        verbose=False,
    )
    res = results[0]
    best = pick_best_bbox(res, im.shape, edge_margin_ratio=edge_margin_ratio)

    out_path = None
    if visualize and output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        vis = im.copy()
        if best is not None:
            vis = draw_best_on_image(vis, best, class_names)
        else:
            cv2.putText(
                vis, "No detections", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )
        out_path = Path(output_dir) / Path(image_path).name
        cv2.imwrite(str(out_path), vis)

    if best is not None:
        name = class_names[best["cls"]] if 0 <= best["cls"] < len(class_names) else f"id={best['cls']}"
        print("✅ Highest-confidence detection:")
        print(f"  Class: {name} (id={best['cls']})")
        print(f"  Confidence: {best['conf']:.4f}")
        print(f"  BBox [x1, y1, x2, y2]: {best['xyxy'].astype(int).tolist()}")
    else:
        print("⚠️ No detections found (sau lọc mép).")

    return res, best, out_path


def predict_directory(model, input_dir, class_names, conf_threshold=0.001, iou_threshold=0.7,
                      output_dir=None, imgsz=640, edge_margin_ratio=0.0):
    """
    Chạy inference cho tất cả ảnh trong thư mục. Vẽ/lưu bbox tốt nhất mỗi ảnh.
    """
    input_dir = Path(input_dir)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in exts]

    print(f"[INFO] Tìm thấy {len(image_files)} ảnh trong {input_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True) if output_dir else None

    results = []
    for img_path in tqdm(image_files, desc="Processing images"):
        res, best, out_path = predict_image(
            model, img_path, class_names,
            conf_threshold=conf_threshold, iou_threshold=iou_threshold,
            output_dir=output_dir, visualize=True,
            imgsz=imgsz, edge_margin_ratio=edge_margin_ratio
        )
        results.append((img_path, res, best, out_path))
    return results


def predict_video(model, video_path, class_names, conf_threshold=0.001, iou_threshold=0.7,
                  output_dir=None, save_video=True, imgsz=640, edge_margin_ratio=0.0):
    """
    Chạy inference cho video. Mỗi frame chỉ vẽ detection có confidence cao nhất.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")

    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {w}x{h} @ {fps}fps | {total} frames")

    out = None
    output_path = None
    if save_video and output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / f"pred_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    pbar = tqdm(total=total if total > 0 else None, desc="Processing video")
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            save=False,
            show=False,
            verbose=False,
        )
        res = results[0]
        best = pick_best_bbox(res, frame.shape, edge_margin_ratio=edge_margin_ratio)

        vis = frame
        if best is not None:
            vis = draw_best_on_image(frame.copy(), best, class_names)
            if frame_idx % max(1, fps // 2) == 0:
                name = class_names[best["cls"]] if 0 <= best["cls"] < len(class_names) else f"id={best['cls']}"
                print(f"[Frame {frame_idx}] {name} conf={best['conf']:.3f} xyxy={np.round(best['xyxy']).astype(int).tolist()}")
        else:
            if frame_idx % max(1, fps // 2) == 0:
                print(f"[Frame {frame_idx}] No detections (sau lọc mép).")

        if out is not None:
            out.write(vis)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if out is not None:
        out.release()
        print(f"[INFO] Saved video to: {output_path}")

    print(f"[INFO] Processed {frame_idx} frames")
    return output_path


# -----------------------------
# Runner
# -----------------------------
def run_inference(
    model_path: str,
    input_path: str,
    data_yaml: str = None,
    custom_classes=None,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    output_dir: str = "outputs/predictions",
    save_results: bool = True,
    imgsz: int = 640,
    edge_margin_ratio: float = 0.0,
):
    """
    Run inference với YOLO-World theo logic:
    - Chỉ chọn detection có confidence cao nhất cho mỗi ảnh/frame
    - Tự vẽ & lưu (không dùng .predict(save=True))
    - Lọc bỏ các box sát mép nếu edge_margin_ratio > 0
    """
    if not (0.0 <= edge_margin_ratio < 0.5):
        raise ValueError("--filter-box phải trong khoảng [0.0, 0.49]")

    model, class_names = load_model_and_classes(model_path, data_yaml, custom_classes)

    print("\n" + "=" * 60)
    print("Inference Configuration")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Classes: {class_names}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Image size (imgsz): {imgsz}")
    print(f"Edge filter ratio (--filter-box): {edge_margin_ratio}")
    print("=" * 60 + "\n")

    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            print(f"[INFO] Processing video: {input_path}")
            video_out = predict_video(
                model, input_path, class_names,
                conf_threshold=conf_threshold, iou_threshold=iou_threshold,
                output_dir=output_dir if save_results else None,
                save_video=save_results,
                imgsz=imgsz,
                edge_margin_ratio=edge_margin_ratio,
            )
            return {"type": "video", "output": video_out}
        else:
            print(f"[INFO] Processing image: {input_path}")
            res, best, out_path = predict_image(
                model, input_path, class_names,
                conf_threshold=conf_threshold, iou_threshold=iou_threshold,
                output_dir=output_dir if save_results else None,
                visualize=save_results,
                imgsz=imgsz,
                edge_margin_ratio=edge_margin_ratio,
            )
            return {"type": "image", "best": best, "output": out_path}
    elif input_path.is_dir():
        print(f"[INFO] Processing directory: {input_path}")
        results = predict_directory(
            model, input_path, class_names,
            conf_threshold=conf_threshold, iou_threshold=iou_threshold,
            output_dir=output_dir if save_results else None,
            imgsz=imgsz,
            edge_margin_ratio=edge_margin_ratio,
        )
        return {"type": "dir", "results": results}
    else:
        raise ValueError(f"Invalid input path: {input_path}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO-World inference (pick highest-confidence bbox)")

    # Model args
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO-World weights")
    parser.add_argument("--input", type=str, required=True, help="Path tới ảnh / thư mục ảnh / video")

    # Class args (chọn 1 trong 2)
    parser.add_argument("--data", type=str, help="Path tới data.yaml để lấy classes đã train")
    parser.add_argument("--classes", type=str, nargs="+", help="Danh sách custom class names để zero-shot")

    # Inference args
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (mặc định giống sample)")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold cho NMS")

    # ⭐️ Thêm mới
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào cho predict (ví dụ 640, 960, 1280)")
    parser.add_argument(
        "--filter-box",
        dest="filter_box",
        type=float,
        default=0.0,
        help="Tỷ lệ biên để loại box sát mép ảnh (0.0..0.49). Ví dụ 0.02 = 2%% kích thước ảnh."
    )

    parser.add_argument("--output", type=str, default="outputs/predictions", help="Thư mục lưu kết quả")
    parser.add_argument("--no_save", action="store_true", help="Không lưu kết quả visualize/video")

    args = parser.parse_args()

    if not args.data and not args.classes:
        parser.error("Cần truyền 1 trong 2: --data hoặc --classes")

    run_inference(
        model_path=args.model,
        input_path=args.input,
        data_yaml=args.data,
        custom_classes=args.classes,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_dir=args.output,
        save_results=not args.no_save,
        imgsz=args.imgsz,
        edge_margin_ratio=args.filter_box,
    )

