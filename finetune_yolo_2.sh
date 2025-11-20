python finetune_yolo.py \
    --data data/yoloworld_no_bg_tiles/data_tiles.yaml \
    --weights yolo12s.pt \
    --epochs 10 \
    --batch 16 \
    --imgsz 640 \
    --project runs/yolo_tiling \
    --name v12s_640_bs_16_10eps_no_bg_tiling \
    --device 1

python finetune_yolo.py \
    --data data/yoloworld_no_bg_tiles/data_tiles.yaml \
    --weights yolo12s.pt \
    --epochs 20 \
    --batch 16 \
    --imgsz 640 \
    --project runs/yolo_tiling \
    --name v12s_640_bs_16_20eps_no_bg_tiling \
    --device 1

python finetune_yolo.py \
    --data data/yoloworld_no_bg_tiles/data_tiles.yaml \
    --weights yolo12s.pt \
    --epochs 30 \
    --batch 16 \
    --imgsz 640 \
    --project runs/yolo_tiling \
    --name v12s_640_bs_16_30eps_no_bg_tiling \
    --device 1


python finetune_yolo.py \
    --data data/yoloworld_no_bg_tiles/data_tiles.yaml \
    --weights yolo12s.pt \
    --epochs 40 \
    --batch 16 \
    --imgsz 640 \
    --project runs/yolo_tiling \
    --name v12s_640_bs_16_40eps_no_bg_tiling \
    --device 1

# python finetune_yolo.py \
#     --data data/yoloworld_no_bg/data.yaml \
#     --weights yolov8s.pt \
#     --epochs 30 \
#     --batch 16 \
#     --imgsz 640 \
#     --mosaic 0.0 \
#     --project runs/yolo \
#     --name v8s_640_bs_16_50eps_no_bg_exp3 \
#     --device 1

# python finetune_yolo.py \
#     --data data/yoloworld_no_bg/data.yaml \
#     --weights yolov8s.pt \
#     --epochs 30 \
#     --batch 16 \
#     --imgsz 640 \
#     --mosaic 1.0 \
#     --mixup 0.7 \
#     --project runs/yolo \
#     --name v8s_640_bs_16_30eps_no_bg_exp4 \
#     --device 1

# python finetune_yolo.py \
#     --data data/yoloworld_no_bg/data.yaml \
#     --weights yolov8s.pt \
#     --epochs 30 \
#     --batch 16 \
#     --imgsz 640 \
#     --mosaic 1.0 \
#     --mixup 0.5 \
#     --project runs/yolo \
#     --name v8s_640_bs_16_30eps_no_bg_exp5 \
#     --device 1

# python finetune_yolo.py \
#     --data data/yoloworld_no_bg/data.yaml \
#     --weights yolov8s.pt \
#     --epochs 30 \
#     --batch 16 \
#     --imgsz 640 \
#     --mosaic 1.0 \
#     --mixup 0.3 \
#     --project runs/yolo \
#     --name v8s_640_bs_16_30eps_no_bg_exp6 \
#     --device 1


# python finetune_yolo.py \
#     --data data/yoloworld_no_bg/data.yaml \
#     --weights yolov8s.pt \
#     --epochs 30 \
#     --batch 16 \
#     --imgsz 960 \
#     --mosaic 1.0 \
#     --mixup 0.7 \
#     --project runs/yolo \
#     --name v8s_960_bs_16_30eps_no_bg_exp4 \
#     --device 1

# python finetune_yolo.py \
#     --data data/yoloworld_no_bg/data.yaml \
#     --weights yolov8s.pt \
#     --epochs 30 \
#     --batch 16 \
#     --imgsz 960 \
#     --mosaic 1.0 \
#     --mixup 0.5 \
#     --project runs/yolo \
#     --name v8s_960_bs_16_30eps_no_bg_exp5 \
#     --device 1

# python finetune_yolo.py \
#     --data data/yoloworld_no_bg/data.yaml \
#     --weights yolov8s.pt \
#     --epochs 30 \
#     --batch 16 \
#     --imgsz 960 \
#     --mosaic 1.0 \
#     --mixup 0.3 \
#     --project runs/yolo \
#     --name v8s_960_bs_16_30eps_no_bg_exp6 \
#     --device 1
