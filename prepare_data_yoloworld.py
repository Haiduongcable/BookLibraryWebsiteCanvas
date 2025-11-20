#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import shutil
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm
import argparse


def extract_class_name(video_id):
    """Extract class name from video_id (e.g., 'Backpack_0' -> 'Backpack')"""
    parts = video_id.rsplit('_', 1)
    return parts[0] if len(parts) > 1 else video_id


def extract_frames_from_video(video_path, output_dir, frame_indices=None):
    """
    Extract specific frames from video using SEQUENTIAL reading for speed.
    (Not used in main loop anymore, but kept for reference/utility.)
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    extracted_frames = {}

    if frame_indices is None:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted_frames[frame_idx] = frame_path
            frame_idx += 1
    else:
        frame_indices_set = set(frame_indices)
        max_frame_idx = max(frame_indices)
        frame_idx = 0
        while frame_idx <= max_frame_idx:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in frame_indices_set:
                frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted_frames[frame_idx] = frame_path
            frame_idx += 1

    cap.release()
    return extracted_frames, total_frames


def get_background_frames(total_frames, annotated_frames, bg_ratio=0.3):
    """Select background frames (frames without annotations)"""
    all_frames = set(range(total_frames))
    annotated_set = set(annotated_frames)
    background_frames = list(all_frames - annotated_set)

    num_bg_samples = int(len(annotated_frames) * bg_ratio)
    num_bg_samples = min(num_bg_samples, len(background_frames))

    selected_bg = random.sample(background_frames, num_bg_samples) if num_bg_samples > 0 else []
    return selected_bg


def convert_to_yolo_format(annotations, img_width, img_height, class_id: int):
    """Convert bboxes to YOLO format lines: class_id x_center y_center width height (normalized)"""
    yolo_annotations = []
    for ann in annotations:
        x1, y1, x2, y2 = ann['x1'], ann['y1'], ann['x2'], ann['y2']

        x_center = (x1 + x2) / 2.0 / img_width
        y_center = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        # Clip to [0,1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_annotations


def prepare_yolo_dataset(
    data_dir,
    annotations_path,
    output_dir,
    bg_ratio=0.3,
    train_split=0.8,
    jpeg_quality=95,
    resize_width=None,
    multi_class=False,
):
    """
    Prepare dataset in YOLO format with OPTIMIZED SEQUENTIAL frame extraction.

    Args:
        data_dir: Directory containing video folders
        annotations_path: Path to annotations JSON
        output_dir: Output directory for prepared dataset
        bg_ratio: Ratio of background samples to annotated samples
        train_split: Train/val split ratio
        jpeg_quality: JPEG compression quality (1-100)
        resize_width: Optional width to resize frames (keep aspect ratio)
        multi_class: If True, use multiple classes by video base name. If False, single class 'object'.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Create output dirs
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(annotations_path, 'r') as f:
        all_annotations = json.load(f)

    # Group annotations by video_id and collect class names (if multi_class)
    video_annotations = defaultdict(list)
    class_names_set = set()

    for item in all_annotations:
        video_id = item['video_id']
        video_annotations[video_id] = item.get('annotations', [])
        if multi_class:
            class_names_set.add(extract_class_name(video_id))

    # Build class mapping
    if multi_class:
        class_names = sorted(list(class_names_set))
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
    else:
        class_names = ["object"]
        class_to_id = {"object": 0}

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"multi_class: {multi_class}")
    print(f"Background ratio: {bg_ratio}")
    print(f"JPEG quality: {jpeg_quality}")
    if resize_width:
        print(f"Resize width: {resize_width}px (keep aspect ratio)")
    print(f"Performance: Sequential frame extraction")

    # Split videos into train/val
    video_ids = list(video_annotations.keys())
    random.shuffle(video_ids)
    split_idx = int(len(video_ids) * train_split)
    train_videos = set(video_ids[:split_idx])
    val_videos = set(video_ids[split_idx:])

    print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")

    all_samples = {'train': [], 'val': []}

    for video_id in tqdm(video_ids, desc="Processing videos"):
        split = 'train' if video_id in train_videos else 'val'
        video_path = data_dir / video_id / 'drone_video.mp4'

        if not video_path.exists():
            print(f"[WARN] Video not found: {video_path}")
            continue

        # Determine class_id for this video
        if multi_class:
            class_name = extract_class_name(video_id)
            class_id = class_to_id[class_name]
        else:
            class_id = 0  # single class "object"

        # Group annotations by frame
        frame_annotations = defaultdict(list)
        for ann in video_annotations[video_id]:
            for bbox in ann.get('bboxes', []):
                frame_idx = bbox['frame']
                frame_annotations[frame_idx].append(bbox)

        annotated_frame_indices = list(frame_annotations.keys())

        # Total frames
        cap_probe = cv2.VideoCapture(str(video_path))
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_probe.release()

        # Background frames
        bg_frames = get_background_frames(total_frames, annotated_frame_indices, bg_ratio)

        # Frames to extract
        all_frames_to_extract = set(annotated_frame_indices + bg_frames)
        if not all_frames_to_extract:
            # nothing to extract for this video (no annots and bg_ratio=0)
            continue

        print(f"  {video_id}: Extracting {len(annotated_frame_indices)} annotated + {len(bg_frames)} background frames...")

        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        max_frame_needed = max(all_frames_to_extract)

        # We'll store dimensions after first successful read (any frame)
        img_width = None
        img_height = None

        while frame_idx <= max_frame_needed:
            ret, frame = cap.read()
            if not ret:
                break

            # Initialize dims on first available frame
            if img_width is None or img_height is None:
                img_height, img_width = frame.shape[:2]

            if frame_idx in all_frames_to_extract:
                is_annotated = frame_idx in frame_annotations

                # Resize if required
                save_frame = frame
                scale_x = scale_y = 1.0
                if resize_width and img_width and img_width > resize_width:
                    scale_x = resize_width / img_width
                    scale_y = scale_x
                    new_height = int(img_height * scale_y)
                    save_frame = cv2.resize(frame, (resize_width, new_height))

                used_width = save_frame.shape[1]
                used_height = save_frame.shape[0]

                if is_annotated:
                    # Adjust bboxes if resized
                    adjusted_anns = []
                    for ann in frame_annotations[frame_idx]:
                        adj = ann.copy()
                        if scale_x != 1.0:
                            adj['x1'] = int(ann['x1'] * scale_x)
                            adj['y1'] = int(ann['y1'] * scale_y)
                            adj['x2'] = int(ann['x2'] * scale_x)
                            adj['y2'] = int(ann['y2'] * scale_y)
                        adjusted_anns.append(adj)

                    yolo_lines = convert_to_yolo_format(adjusted_anns, used_width, used_height, class_id)

                    img_name = f"{video_id}_frame_{frame_idx:06d}.jpg"
                    label_name = f"{video_id}_frame_{frame_idx:06d}.txt"

                    img_path = output_dir / split / 'images' / img_name
                    cv2.imwrite(str(img_path), save_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

                    with open(output_dir / split / 'labels' / label_name, 'w') as f:
                        f.write('\n'.join(yolo_lines))

                    all_samples[split].append(img_name)
                else:
                    # Background: empty label file
                    img_name = f"{video_id}_bg_frame_{frame_idx:06d}.jpg"
                    label_name = f"{video_id}_bg_frame_{frame_idx:06d}.txt"

                    img_path = output_dir / split / 'images' / img_name
                    cv2.imwrite(str(img_path), save_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

                    with open(output_dir / split / 'labels' / label_name, 'w') as f:
                        pass

                    all_samples[split].append(img_name)

            frame_idx += 1

        cap.release()

    # Write data.yaml
    yaml_content = (
        f"# YOLO-World Dataset Configuration\n"
        f"path: {output_dir.absolute()}\n"
        f"train: train/images\n"
        f"val: val/images\n\n"
        f"# Classes\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n\n"
        f"# Class mapping\n"
        f"class_to_id: {class_to_id}\n"
    )

    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print("\n" + "=" * 50)
    print("Dataset preparation completed!")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Train samples: {len(all_samples['train'])}")
    print(f"Val samples: {len(all_samples['val'])}")
    print(f"Classes: {class_names}")
    print(f"Data config saved to: {output_dir / 'data.yaml'}")

    return class_names, class_to_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare YOLO-World dataset from drone videos')
    parser.add_argument('--data_dir', type=str, default='data/train/samples',
                        help='Directory containing video folders')
    parser.add_argument('--annotations', type=str, default='data/train/annotations/annotations.json',
                        help='Path to annotations JSON file')
    parser.add_argument('--output_dir', type=str, default='dataset_yolo',
                        help='Output directory for prepared dataset')
    parser.add_argument('--bg_ratio', type=float, default=0.3,
                        help='Ratio of background samples to annotated samples (default: 0.3)')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--jpeg_quality', type=int, default=90,
                        help='JPEG compression quality 1-100 (default: 95)')
    parser.add_argument('--resize_width', type=int, default=None,
                        help='Resize frames to this width (keeps aspect ratio)')
    parser.add_argument('--multi_class', action='store_true',
                        help='Use multiple classes by video name. If omitted, use single class "object".')
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)

    # Run
    prepare_yolo_dataset(
        args.data_dir,
        args.annotations,
        args.output_dir,
        bg_ratio=args.bg_ratio,
        train_split=args.train_split,
        jpeg_quality=args.jpeg_quality,
        resize_width=args.resize_width,
        multi_class=args.multi_class,
    )

