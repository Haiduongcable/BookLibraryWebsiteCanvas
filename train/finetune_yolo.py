#!/usr/bin/env python3
# train_yolo.py

import argparse
import logging
from ultralytics import YOLO
from pathlib import Path
import time
import os

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("train")

def main():
    parser = argparse.ArgumentParser()
    # Data / model / runtime
    parser.add_argument("--data", default="aero_eyes/data/yolo_dataset/data.yaml")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="aero_eyes_yolo")
    parser.add_argument("--project", default="runs", help="Thư mục gốc để lưu kết quả (Ultralytics sẽ tạo subdir theo task)")
    parser.add_argument("--device", default=None, help="e.g., '0', '0,1', or 'cpu'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    # Augmentation (YOLOv8)
    parser.add_argument("--degrees", type=float, default=0.0, help="rotation max degrees")
    parser.add_argument("--translate", type=float, default=0.1, help="image translation fraction")
    parser.add_argument("--scale", type=float, default=0.5, help="image scale gain")
    parser.add_argument("--shear", type=float, default=0.0, help="shear degrees")
    parser.add_argument("--perspective", type=float, default=0.0, help="perspective fraction")
    parser.add_argument("--flipud", type=float, default=0.0, help="vertical flip probability")
    parser.add_argument("--fliplr", type=float, default=0.5, help="horizontal flip probability")
    parser.add_argument("--mosaic", type=float, default=1.0, help="mosaic probability (0 tắt)")
    parser.add_argument("--mixup", type=float, default=0.0, help="mixup probability")
    parser.add_argument("--copy_paste", type=float, default=0.0, help="copy-paste probability")
    parser.add_argument("--hsv_h", type=float, default=0.015, help="HSV-H gain")
    parser.add_argument("--hsv_s", type=float, default=0.7, help="HSV-S gain")
    parser.add_argument("--hsv_v", type=float, default=0.4, help="HSV-V gain")
    parser.add_argument("--erasing", type=float, default=0.0, help="random erasing probability")
    parser.add_argument("--close_mosaic", type=int, default=10, help="Tắt mosaic ở N epoch cuối")

    # (optional) Optim & warmup (để tiện bạn tinh chỉnh nếu cần)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--warmup_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_momentum", type=float, default=0.8)
    parser.add_argument("--warmup_bias_lr", type=float, default=0.1)

    args = parser.parse_args()

    logger = setup_logger()
    logger.info(f"Start training with: data={args.data}, weights={args.weights}")

    # Hiển thị cấu hình chính
    logger.info(
        f"project={args.project} | name={args.name} | epochs={args.epochs} | imgsz={args.imgsz} | "
        f"batch={args.batch} | device={args.device}"
    )
    logger.info(
        "Augment: "
        f"deg={args.degrees} trans={args.translate} scale={args.scale} shear={args.shear} "
        f"persp={args.perspective} flipud={args.flipud} fliplr={args.fliplr} "
        f"mosaic={args.mosaic} mixup={args.mixup} copy_paste={args.copy_paste} "
        f"hsv=({args.hsv_h},{args.hsv_s},{args.hsv_v}) erasing={args.erasing} "
        f"close_mosaic={args.close_mosaic}"
    )

    # Set seed (để reproducible tốt hơn)
    if args.seed is not None:
        os.environ["YOLO_RANDOM_SEED"] = str(args.seed)

    model = YOLO(args.weights)

    t0 = time.time()
    results = model.train(
        # dataset / runtime
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
        device=args.device,
        exist_ok=True,
        pretrained=True,
        workers=args.workers,
        seed=args.seed,

        # optimizer & sched
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,

        # augmentation
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        erasing=args.erasing,
        close_mosaic=args.close_mosaic,
        single_cls=True
    )

    logger.info(f"Training done in {time.time()-t0:.1f}s.")
    run_dir = Path(args.project) / "detect" / args.name
    logger.info(f"Run dir: {run_dir.resolve()}")

if __name__ == "__main__":
    main()
