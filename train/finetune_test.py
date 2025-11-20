from ultralytics import YOLOWorld, YOLO
# train model yolov8s 1280: public LB 0.9285 private LB 0.9226
# init yolov8s model and load a pretrained model from Ultralytics https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt
model1 = YOLOWorld("yolov8s-worldv2.pt")

import random, numpy as np, torch
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# model1 = YOLO("yolov8s.pt")
results1 = model1.train(
    data="data/yoloworld_v1/data.yaml",
    epochs=10,
    imgsz=640,
    batch=32,
    workers=8,
    name="test",
    lr0=0.01,
    flipud=0.5,
    fliplr=0.5,
    close_mosaic=5,
    device='0',
    single_cls=False,
    mosaic=0.5
)