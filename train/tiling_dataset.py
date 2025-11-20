#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tile YOLO-format dataset into 4 quadrants (2x2 grid) and update labels accordingly.

Input layout (example):
  data/yoloworld_v3/
    train/images
    train/labels
    val/images
    val/labels
  data.yaml  (optional; used to copy nc/names and emit new data_tiles.yaml)

Output layout:
  <out_root>/
    train/images
    train/labels
    val/images
    val/labels
    data_tiles.yaml

Usage:
  python tile_yoloworld_quadrants.py \
      --in-root data/yoloworld_v3 \
      --out-root data/yoloworld_v3_tiles \
      --data-yaml data/yoloworld_v3/data.yaml \
      --min-coverage 0.2 \
      --min-area 16

Notes:
- Class IDs are preserved 1:1.
- Boxes are clipped to each tile. A box is kept in a tile if
  (area_of_intersection / original_box_area) >= --min-coverage (default 0.2)
  and the clipped area in pixels is >= --min-area (default 16 px^2).
- Supports images of any size; split positions are computed with floor/ceil.
- Accepts .txt YOLO labels with lines: `cls cx cy w h` (normalized).
"""

import argparse
import math
from pathlib import Path
import shutil
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(p: Path) -> List[Path]:
    return sorted([x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS])


def read_yolo_labels(lbl_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not lbl_path.exists():
        return []
    out = []
    with lbl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                out.append((cls, cx, cy, w, h))
            except Exception:
                # ignore malformed lines
                continue
    return out


def write_yolo_labels(lbl_path: Path, rows: List[Tuple[int, float, float, float, float]]):
    if not rows:
        # If no labels, we still create an empty file to be explicit
        lbl_path.write_text("", encoding="utf-8")
        return
    with lbl_path.open("w", encoding="utf-8") as f:
        for cls, cx, cy, w, h in rows:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def xywhn_to_xyxy_abs(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    x = cx * W
    y = cy * H
    bw = w * W
    bh = h * H
    x1 = x - bw / 2.0
    y1 = y - bh / 2.0
    x2 = x + bw / 2.0
    y2 = y + bh / 2.0
    return x1, y1, x2, y2


def xyxy_abs_to_xywhn(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float, float, float, float]:
    # clip to [0, W/H]
    x1 = max(0.0, min(float(W), x1))
    y1 = max(0.0, min(float(H), y1))
    x2 = max(0.0, min(float(W), x2))
    y2 = max(0.0, min(float(H), y2))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if W <= 0 or H <= 0 or bw <= 0 or bh <= 0:
        return 0.0, 0.0, 0.0, 0.0
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w = bw / W
    h = bh / H
    return cx, cy, w, h


def bbox_intersection(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0, 0.0
    return x1, y1, x2, y2


def area(b: Tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def compute_quadrants(W: int, H: int):
    # Split positions: left width wL, right width wR, similarly for height
    wL = W // 2
    wR = W - wL
    hT = H // 2
    hB = H - hT
    # tiles: (x, y, w, h), row-major (top-left, top-right, bottom-left, bottom-right)
    tiles = [
        (0, 0, wL, hT),            # r0 c0
        (wL, 0, wR, hT),           # r0 c1
        (0, hT, wL, hB),           # r1 c0
        (wL, hT, wR, hB),          # r1 c1
    ]
    return tiles


def process_image(img_path: Path, lbl_path: Path, out_img_dir: Path, out_lbl_dir: Path,
                  min_coverage: float = 0.2, min_area: float = 16.0):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    H, W = img.shape[:2]

    labels = read_yolo_labels(lbl_path)

    tiles = compute_quadrants(W, H)

    base = img_path.stem
    ext = img_path.suffix.lower()

    for idx, (tx, ty, tw, th) in enumerate(tiles):
        # Crop and save tile image
        tile_img = img[ty:ty+th, tx:tx+tw]
        row = 0 if idx < 2 else 1
        col = 0 if idx % 2 == 0 else 1
        out_img_name = f"{base}_r{row}_c{col}{ext}"
        out_img_path = out_img_dir / out_img_name
        cv2.imwrite(str(out_img_path), tile_img)

        # Prepare labels for this tile
        tile_labels: List[Tuple[int, float, float, float, float]] = []
        tile_rect = (float(tx), float(ty), float(tx + tw), float(ty + th))
        for (cls, cx, cy, w, h) in labels:
            # original bbox in absolute coords
            bx1, by1, bx2, by2 = xywhn_to_xyxy_abs(cx, cy, w, h, W, H)
            orig = (bx1, by1, bx2, by2)
            inter = bbox_intersection(orig, tile_rect)
            inter_area = area(inter)
            orig_area = area(orig)
            if inter_area <= 0 or orig_area <= 0:
                continue
            coverage = inter_area / orig_area
            if coverage < min_coverage or inter_area < min_area:
                continue
            # shift to tile coordinates
            ix1, iy1, ix2, iy2 = inter
            # Convert intersection in global coords -> local tile coords
            lx1 = ix1 - tx
            ly1 = iy1 - ty
            lx2 = ix2 - tx
            ly2 = iy2 - ty
            # to normalized xywh w.r.t tile size
            cxn, cyn, wn, hn = xyxy_abs_to_xywhn(lx1, ly1, lx2, ly2, tw, th)
            if wn <= 0 or hn <= 0:
                continue
            tile_labels.append((cls, cxn, cyn, wn, hn))

        out_lbl_name = f"{base}_r{row}_c{col}.txt"
        out_lbl_path = out_lbl_dir / out_lbl_name
        write_yolo_labels(out_lbl_path, tile_labels)


def ensure_clean_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_or_emit_yaml(in_root: Path, out_root: Path, data_yaml_path: Path = None):
    """Create a new data_tiles.yaml pointing to tiled dataset. Try to copy nc/names from existing data.yaml if provided."""
    names = None
    nc = None
    if data_yaml_path is not None and data_yaml_path.exists() and yaml is not None:
        try:
            with data_yaml_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                names = cfg.get("names", None)
                nc = cfg.get("nc", None)
        except Exception:
            pass
    # fallback: attempt to infer nc from first labels file if missing
    if nc is None:
        # naive: scan any label file for max class id
        lbl_dirs = [out_root/"train"/"labels", out_root/"val"/"labels"]
        max_cls = -1
        for d in lbl_dirs:
            for txt in d.glob("*.txt"):
                rows = read_yolo_labels(txt)
                for (cls, *_rest) in rows:
                    max_cls = max(max_cls, cls)
        if max_cls >= 0:
            nc = max_cls + 1
    data = {
        "path": str(out_root.resolve()),
        "train": str((out_root/"train"/"images").resolve()),
        "val": str((out_root/"val"/"images").resolve()),
    }
    if nc is not None:
        data["nc"] = int(nc)
    if names is not None:
        data["names"] = names

    if yaml is None:
        # If pyyaml missing, write minimal txt
        (out_root/"data_tiles.yaml").write_text(str(data), encoding="utf-8")
    else:
        with (out_root/"data_tiles.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def process_split(split: str, in_root: Path, out_root: Path, min_coverage: float, min_area: float):
    in_img_dir = in_root / split / "images"
    in_lbl_dir = in_root / split / "labels"
    out_img_dir = out_root / split / "images"
    out_lbl_dir = out_root / split / "labels"
    ensure_clean_dir(out_img_dir)
    ensure_clean_dir(out_lbl_dir)

    images = list_images(in_img_dir)
    if not images:
        print(f"[WARN] No images found in {in_img_dir}")
        return

    for img_path in tqdm(images, desc=f"Tiling {split}"):
        lbl_path = in_lbl_dir / (img_path.stem + ".txt")
        try:
            process_image(img_path, lbl_path, out_img_dir, out_lbl_dir, min_coverage, min_area)
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Tile YOLO dataset into 2x2 quadrants with label updates")
    ap.add_argument("--in-root", type=Path, required=True, help="Input dataset root (contains train/ and val/)")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root for tiled dataset")
    ap.add_argument("--data-yaml", type=Path, default=None, help="Path to original data.yaml (optional)")
    ap.add_argument("--min-coverage", type=float, default=0.2, help="Min intersection/original coverage to keep a box in a tile")
    ap.add_argument("--min-area", type=float, default=16.0, help="Min intersection area (px^2) to keep a box in a tile")
    args = ap.parse_args()

    # Create out structure
    for split in ("train", "val"):
        ensure_clean_dir(args.out_root / split / "images")
        ensure_clean_dir(args.out_root / split / "labels")

    # Process splits
    for split in ("train", "val"):
        process_split(split, args.in_root, args.out_root, args.min_coverage, args.min_area)

    # Emit data_tiles.yaml
    copy_or_emit_yaml(args.in_root, args.out_root, args.data_yaml)
    print(f"\n✅ Done. Tiled dataset written to: {args.out_root}")
    print(f"📄 data_tiles.yaml generated at: {args.out_root / 'data_tiles.yaml'}")


if __name__ == "__main__":
    main()
