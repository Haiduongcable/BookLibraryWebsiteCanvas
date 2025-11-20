#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finetune YOLO-World with single-class toggle and rich augmentation knobs.

Usage example:

python finetune_yoloworld_singlecls_augment.py \
  --data data/yoloworld_v3/data.yaml \
  --model_size m \
  --epochs 100 --batch_size 16 --img_size 640 --device 0 \
  --project runs/finetune --name yoloworld_custom \
  --single-cls \
  --flipud 0.0 --fliplr 0.5 \
  --degrees 0.0 --translate 0.1 --scale 0.5 --shear 0.0 --perspective 0.0 \
  --hsv-h 0.015 --hsv-s 0.7 --hsv-v 0.4 \
  --mosaic 1.0 --close-mosaic 10 --mixup 0.2 --copy-paste 0.0 --erasing 0.4

Notes:
- `--single-cls` collapses all labels into one class during training (binary object vs. background). Useful for single-object presence tasks or when class labels are unreliable.
- Augmentation arguments map 1:1 to Ultralytics training keys.
"""

import os
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLOWorld


def setup_training_config(data_yaml, epochs, batch_size, img_size, model_size='m'):
    """Setup training configuration and print a nice summary."""
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    class_names = data_config.get('names', [])

    print(f"\n{'='*60}")
    print("Training Configuration")
    print(f"{'='*60}")
    print(f"Model: YOLO-World-{model_size.upper()}")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"{'='*60}\n")

    return class_names


def finetune_yoloworld(
    data_yaml: str,
    model_size: str = 'm',
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = '0',
    project: str = 'runs/finetune',
    name: str = 'yoloworld_custom',
    pretrained: bool = True,
    resume: bool = False,
    cache: bool = False,
    workers: int = 8,
    patience: int = 20,
    save_period: int = 10,
    # NEW: classification collapse
    single_cls: bool = False,
    # NEW: augmentation knobs
    flipud: float = 0.0,
    fliplr: float = 0.5,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    mosaic: float = 1.0,
    close_mosaic: int = 10,
    mixup: float = 0.0,
    copy_paste: float = 0.0,
    erasing: float = 0.0,
    dropout: float = 0.0,
):
    """
    Finetune YOLO-World model on custom dataset with optional single-class mode
    and rich augmentation controls.
    """

    # Setup configuration
    class_names = setup_training_config(data_yaml, epochs, batch_size, img_size, model_size)

    # Initialize YOLO-World model
    model_name = f'yolov8{model_size}-worldv2.pt'
    print(f"Loading YOLO-World model: {model_name}")

    try:
        model = YOLOWorld(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to download model...")
        model = YOLOWorld(model_name)

    # Compose training args
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': True,
        'pretrained': pretrained,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'mosaic': mosaic,
        'close_mosaic': close_mosaic,
        'mixup': mixup,
        'copy_paste': copy_paste,
        'erasing': erasing,
        'flipud': flipud,
        'fliplr': fliplr,
        'degrees': degrees,
        'translate': translate,
        'scale': scale,
        'shear': shear,
        'perspective': perspective,
        'hsv_h': hsv_h,
        'hsv_s': hsv_s,
        'hsv_v': hsv_v,
        'amp': True,          # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': dropout,
        'val': True,
        'save': True,
        'save_period': save_period,
        'cache': cache,
        'rect': False,
        'resume': resume,
        'workers': workers,
        'patience': patience,
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'plots': True,
        'show': False,
        'single_cls': single_cls,
    }

    # Pretty print important toggles
    print(f"\n{'='*60}")
    print("Augmentation Settings")
    print(f"{'='*60}")
    for k in ['single_cls','flipud','fliplr','degrees','translate','scale','shear','perspective',
              'hsv_h','hsv_s','hsv_v','mosaic','close_mosaic','mixup','copy_paste','erasing','dropout']:
        print(f"{k:>14}: {train_args[k]}")
    print(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    # Start training
    results = model.train(**train_args)

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Best weights saved to: {Path(project) / name / 'weights' / 'best.pt'}")
    print(f"Last weights saved to: {Path(project) / name / 'weights' / 'last.pt'}")
    print(f"Results saved to: {Path(project) / name}")

    # Validate the best model
    print(f"\n{'='*60}")
    print("Validating best model...")
    print(f"{'='*60}\n")

    best_model_path = Path(project) / name / 'weights' / 'best.pt'
    best_model = YOLOWorld(str(best_model_path))
    # For evaluation/inference with YOLO-World, set the evaluation vocabulary
    best_model.set_classes(class_names)
    metrics = best_model.val(data=data_yaml, split='val')

    print(f"\n{'='*60}")
    print("Validation Results")
    print(f"{'='*60}")
    try:
        print(f"mAP50    : {metrics.box.map50:.4f}")
        print(f"mAP50-95 : {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall   : {metrics.box.mr:.4f}")
    except Exception:
        print(metrics)
    print(f"{'='*60}\n")

    return best_model, results, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune YOLO-World on custom dataset (with single-cls + augmentation).')

    # Data arguments
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--model_size', type=str, default='m', choices=['s', 'm', 'l'], help='Model size (s/m/l)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--device', type=str, default='0', help='CUDA device (e.g., "0" or "0,1" or "cpu")')

    # Output arguments
    parser.add_argument('--project', type=str, default='runs/finetune', help='Project directory')
    parser.add_argument('--name', type=str, default='yoloworld_custom', help='Experiment name')

    # Advanced arguments
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--cache', action='store_true', help='Cache images for faster training')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (epochs)')
    parser.add_argument('--save_period', type=int, default=10, help='Save checkpoint every N epochs')

    # NEW: classification collapse
    parser.add_argument('--single-cls', action='store_true', help='Treat dataset as single class during training')

    # NEW: augmentation knobs
    parser.add_argument('--flipud', type=float, default=0.0, help='Probability of vertical flip')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Probability of horizontal flip')
    parser.add_argument('--degrees', type=float, default=0.0, help='Rotation degrees (±)')
    parser.add_argument('--translate', type=float, default=0.1, help='Translation fraction')
    parser.add_argument('--scale', type=float, default=0.5, help='Scale gain')
    parser.add_argument('--shear', type=float, default=0.0, help='Shear degrees (±)')
    parser.add_argument('--perspective', type=float, default=0.0, help='Perspective fraction (0-0.001)')
    parser.add_argument('--hsv-h', dest='hsv_h', type=float, default=0.015, help='HSV-H gain')
    parser.add_argument('--hsv-s', dest='hsv_s', type=float, default=0.7, help='HSV-S gain')
    parser.add_argument('--hsv-v', dest='hsv_v', type=float, default=0.4, help='HSV-V gain')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic augmentation probability')
    parser.add_argument('--close-mosaic', dest='close_mosaic', type=int, default=10, help='Disable mosaic in last N epochs')
    parser.add_argument('--mixup', type=float, default=0.0, help='MixUp augmentation probability')
    parser.add_argument('--copy-paste', dest='copy_paste', type=float, default=0.0, help='Copy-Paste augmentation probability')
    parser.add_argument('--erasing', type=float, default=0.0, help='Random erasing probability')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout for regularization (if supported)')

    args = parser.parse_args()

    finetune_yoloworld(
        data_yaml=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        resume=args.resume,
        cache=args.cache,
        workers=args.workers,
        patience=args.patience,
        save_period=args.save_period,
        single_cls=args.single_cls,
        flipud=args.flipud,
        fliplr=args.fliplr,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        dropout=args.dropout,
    )
