#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# extract_test_frames.py

import cv2
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("extract_test")

def iter_test_videos(samples_dir: Path):
    # Expect: samples/<ObjectID>/drone_video.mp4
    for obj_dir in sorted(samples_dir.iterdir()):
        if obj_dir.is_dir():
            vpath = obj_dir / "drone_video.mp4"
            if vpath.exists():
                yield obj_dir.name, vpath

def extract_one_video(video_id: str, vpath: Path, out_root: Path, frame_stride: int, max_frames: int, jpeg_quality: int):
    out_dir = out_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return video_id, 0, "Cannot open video"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_stride == 0:
            name = f"{video_id}_frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_dir / name), frame, imwrite_params)
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break

        idx += 1

    cap.release()
    return video_id, saved, None

def main():
    ap = argparse.ArgumentParser(description="Extract frames for public test (no annotations).")
    ap.add_argument("--root", default="data/public_test", help="root chứa 'samples/'")
    ap.add_argument("--out", default="data/public_test_frames", help="thư mục output frames")
    ap.add_argument("--frame_stride", type=int, default=5, help="lấy 1 frame mỗi N frame")
    ap.add_argument("--max_frames_per_video", type=int, default=0, help="0 = không giới hạn")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    logger = setup_logger()
    samples_dir = Path(args.root) / "samples"
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    videos = list(iter_test_videos(samples_dir))
    logger.info(f"Found {len(videos)} videos under {samples_dir}")

    total_saved = 0
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                extract_one_video,
                vid, vpath, out_root, args.frame_stride, args.max_frames_per_video, args.jpeg_quality
            )
            for vid, vpath in videos
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Extracting"):
            vid, saved, err = fut.result()
            if err:
                errors.append((vid, err))
            total_saved += saved

    logger.info(f"✅ Done. Frames saved: {total_saved}")
    if errors:
        logger.warning(f"{len(errors)} videos failed:")
        for vid, err in errors[:10]:
            logger.warning(f"  - {vid}: {err}")

if __name__ == "__main__":
    main()
