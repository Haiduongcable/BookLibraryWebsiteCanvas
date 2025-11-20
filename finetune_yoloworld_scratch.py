#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train YOLO-World FROM SCRATCH (WorldTrainerFromScratch) with flexible data + augmentation.

Example:

python finetune_yoloworld_from_scratch.py \
  --model-size s \
  --yolo-train data/yoloworld_v3/data.yaml \
  --yolo-val data/yoloworld_v3/data.yaml \
  --grounding "flickr30k/images:flickr30k/final_flickr_separateGT_train.json" \
  --grounding "GQA/images:GQA/final_mixed_train_no_coco.json" \
  --epochs 50 --batch-size 64 --img-size 640 --device 0 \
  --project runs/world_from_scratch --name exp_scratch_v1 \
  --single-cls \
  --mosaic 1.0 --close-mosaic 10 --mixup 0.2 \
  --copy-paste 0.2 --erasing 0.4 \
  --flipud 0.0 --fliplr 0.5 \
  --degrees 0.0 --translate 0.1 --scale 0.5 --shear 0.0 --perspective 0.0 \
  --hsv-h 0.015 --hsv-s 0.7 --hsv-v 0.4

Notes:
- This script trains YOLO-World using ONLY the yaml architecture file:
    yolov8s-worldv2.yaml / yolov8m-worldv2.yaml / yolov8l-worldv2.yaml
  => No pretrained detection weights => real "from scratch".
- `--single-cls` applies Ultralytics single_cls flag: treat all classes as 1 object class.
- `--grounding` entries integrate grounding data like the official WorldTrainerFromScratch example.
"""

import argparse
from pathlib import Path
import sys
import textwrap

from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch


def parse_grounding_arg(s: str):
    """
    Parse a grounding spec of the form:
        img_path:json_file
    into dict(img_path=..., json_file=...).
    """
    parts = s.split(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --grounding format: '{s}'. Expected 'img_path:json_file'"
        )
    img_path, json_file = parts
    return dict(img_path=img_path, json_file=json_file)


def build_data_dict(yolo_train, yolo_val, grounding_list):
    """
    Build the `data` dict expected by WorldTrainerFromScratch.

    train:
      - yolo_data: list of dataset yaml files
      - grounding_data: list of {img_path, json_file}
    val:
      - yolo_data: list of dataset yaml files
    """
    data = dict(
        train=dict(
            yolo_data=yolo_train,
            grounding_data=grounding_list,
        ),
        val=dict(
            yolo_data=yolo_val,
        ),
    )
    return data


def print_config_summary(
    model_cfg,
    data_dict,
    epochs,
    batch_size,
    img_size,
    single_cls,
    trainer_kwargs,
):
    print("\n" + "=" * 70)
    print("YOLO-World FROM-SCRATCH Training Configuration")
    print("=" * 70)
    print(f"Backbone / Model config : {model_cfg}")
    print(f"Epochs                  : {epochs}")
    print(f"Batch size              : {batch_size}")
    print(f"Image size              : {img_size}")
    print(f"Single-class mode       : {single_cls}")
    print("-" * 70)
    print("Train YOLO datasets:")
    for y in data_dict["train"]["yolo_data"]:
        print(f"  - {y}")
    print("Train grounding datasets:")
    if data_dict["train"]["grounding_data"]:
        for g in data_dict["train"]["grounding_data"]:
            print(f"  - img_path={g['img_path']} | json_file={g['json_file']}")
    else:
        print("  (none)")
    print("Val YOLO datasets:")
    for y in data_dict["val"]["yolo_data"]:
        print(f"  - {y}")
    print("-" * 70)
    print("Key training args:")
    important_keys = [
        "project",
        "name",
        "device",
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "mosaic",
        "close_mosaic",
        "mixup",
        "copy_paste",
        "erasing",
        "flipud",
        "fliplr",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "single_cls",
        "freeze",
    ]
    for k in important_keys:
        if k in trainer_kwargs:
            print(f"{k:>18}: {trainer_kwargs[k]}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO-World from scratch using WorldTrainerFromScratch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Grounding format:
              --grounding "img_dir:annotations.json"
              (can be used multiple times)

            Examples:
              --grounding "flickr30k/images:flickr30k/final_flickr_separateGT_train.json"
              --grounding "GQA/images:GQA/final_mixed_train_no_coco.json"
            """
        ),
    )

    # Model config (from-scratch)
    parser.add_argument(
        "--model-size",
        type=str,
        default="s",
        choices=["s", "m", "l"],
        help="YOLO-World model size: yolov8{s,m,l}-worldv2.yaml",
    )

    # YOLO-format datasets
    parser.add_argument(
        "--yolo-train",
        action="append",
        required=True,
        help="YOLO data.yaml for training (can be used multiple times).",
    )
    parser.add_argument(
        "--yolo-val",
        action="append",
        required=True,
        help="YOLO data.yaml for validation (can be used multiple times).",
    )

    # Grounding datasets
    parser.add_argument(
        "--grounding",
        action="append",
        default=[],
        help="Grounding dataset spec 'img_path:json_file' (can be used multiple times).",
    )

    # Core training hyperparams
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Training image size")
    parser.add_argument("--device", type=str, default="0", help='Device, e.g. "0", "0,1", "cpu"')

    # Output/management
    parser.add_argument("--project", type=str, default="runs/world_from_scratch", help="Project dir")
    parser.add_argument("--name", type=str, default="exp", help="Run/experiment name")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every N epochs (-1 = only best/last)")
    parser.add_argument("--freeze", type=int, default=0, help="Number of backbone layers to freeze (for stability)")

    # Single-class mode
    parser.add_argument(
        "--single-cls",
        action="store_true",
        help="Treat all classes as a single class during training.",
    )

    # Augmentations
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip prob")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip prob")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation degrees (+/-)")
    parser.add_argument("--translate", type=float, default=0.1, help="Translate fraction")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale gain")
    parser.add_argument("--shear", type=float, default=0.0, help="Shear degrees (+/-)")
    parser.add_argument("--perspective", type=float, default=0.0, help="Perspective fraction (0-0.001)")
    parser.add_argument("--hsv-h", dest="hsv_h", type=float, default=0.015, help="HSV-H gain")
    parser.add_argument("--hsv-s", dest="hsv_s", type=float, default=0.7, help="HSV-S gain")
    parser.add_argument("--hsv-v", dest="hsv_v", type=float, default=0.4, help="HSV-V gain")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic prob")
    parser.add_argument(
        "--close-mosaic",
        dest="close_mosaic",
        type=int,
        default=10,
        help="Disable mosaic in last N epochs",
    )
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp prob")
    parser.add_argument("--copy-paste", dest="copy_paste", type=float, default=0.0, help="Copy-paste prob")
    parser.add_argument("--erasing", type=float, default=0.0, help="Random erasing prob")

    # Misc
    parser.add_argument("--cache", action="store_true", help="Cache images")
    parser.add_argument("--rect", action="store_true", help="Rectangular training")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic training")
    parser.add_argument("--no-val", action="store_true", help="Disable validation during training")

    args = parser.parse_args()

    # Build data dict
    grounding_data = [parse_grounding_arg(g) for g in args.grounding]
    data = build_data_dict(args.yolo_train, args.yolo_val, grounding_data)

    # Model config yaml (from scratch)
    model_cfg = f"yolov8{args.model_size}-worldv2.yaml"

    # Create model from yaml (NO pretrained .pt -> from scratch)
    try:
        model = YOLOWorld(model_cfg)
    except Exception as e:
        print(f"[ERROR] Failed to load model config '{model_cfg}': {e}", file=sys.stderr)
        sys.exit(1)

    # Trainer args (passed directly into model.train)
    trainer_kwargs = {
        "data": data,
        "trainer": WorldTrainerFromScratch,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": args.img_size,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "workers": args.workers,
        "patience": args.patience,
        "save_period": args.save_period,
        "cache": args.cache,
        "rect": args.rect,
        "verbose": args.verbose,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "single_cls": args.single_cls,
        "freeze": args.freeze,
        # Opt / sched
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        # Loss
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        # Augmentations
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "erasing": args.erasing,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        # Misc
        "amp": True,
        "fraction": 1.0,
        "profile": False,
        "overlap_mask": True,
        "mask_ratio": 4,
        "plots": True,
        "save": True,
        "val": not args.no_val,
        "show": False,
    }

    # Summary
    print_config_summary(
        model_cfg=model_cfg,
        data_dict=data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        single_cls=args.single_cls,
        trainer_kwargs=trainer_kwargs,
    )

    print("=" * 70)
    print("Starting YOLO-World FROM-SCRATCH training with WorldTrainerFromScratch...")
    print("=" * 70 + "\n")

    results = model.train(**trainer_kwargs)

    print("\n" + "=" * 70)
    print("Training finished.")
    print("=" * 70)
    run_dir = Path(args.project) / args.name
    print(f"Run directory : {run_dir}")
    print(f"Weights       : {run_dir / 'weights'}")
    print("=" * 70 + "\n")

    # Optional: run final val on best.pt using same trainer/data (if val enabled)
    if not args.no_val:
        try:
            best_model_path = run_dir / "weights" / "best.pt"
            if best_model_path.exists():
                print("Running final validation on best.pt ...\n")
                best_model = YOLOWorld(str(best_model_path))
                # For YOLO-World eval, you usually configure evaluation vocabulary via set_classes if needed.
                metrics = best_model.val(data=args.yolo_val[0])
                print(metrics)
            else:
                print("best.pt not found, skipping final val summary.")
        except Exception as e:
            print(f"Validation error (non-fatal): {e}")


if __name__ == "__main__":
    main()
