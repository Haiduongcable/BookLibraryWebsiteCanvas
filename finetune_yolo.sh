python finetune_yolo.py \
    --data data/yoloworld_no_bg_tiles/data_tiles.yaml \
    --weights yolov8s.pt \
    --epochs 10 \
    --batch 16 \
    --imgsz 640 \
    --project runs/yolo_tiling \
    --name v8s_640_bs_16_10eps_no_bg_tiling

python finetune_yolo.py \
    --data data/yoloworld_no_bg_tiles/data_tiles.yaml \
    --weights yolov8s.pt \
    --epochs 20 \
    --batch 16 \
    --imgsz 640 \
    --project runs/yolo_tiling \
    --name v8s_640_bs_16_20eps_no_bg_tiling

python finetune_yolo.py \
    --data data/yoloworld_no_bg_tiles/data_tiles.yaml \
    --weights yolov8s.pt \
    --epochs 30 \
    --batch 16 \
    --imgsz 640 \
    --project runs/yolo_tiling \
    --name v8s_640_bs_16_30eps_no_bg_tiling


python finetune_yolo.py \
    --data data/yoloworld_no_bg_tiles/data_tiles.yaml \
    --weights yolov8s.pt \
    --epochs 40 \
    --batch 16 \
    --imgsz 640 \
    --project runs/yolo_tiling \
    --name v8s_640_bs_16_40eps_no_bg_tiling

# python finetune_yolo.py \
#     --data data/yoloworld_no_bg/data.yaml \
#     --weights yolo12s.pt \
#     --epochs 50 \
#     --batch 16 \
#     --imgsz 640 \
#     --project runs/yolo \
#     --name v12s_640_bs_16_50eps_no_bg


# python finetune_yolo.py \
#     --data data/yoloworld_no_bg/data.yaml \
#     --weights yolo12s.pt \
#     --epochs 30 \
#     --batch 16 \
#     --imgsz 960 \
#     --project runs/yolo \
#     --name v12s_960_bs_16_30eps_no_bg


# python finetune_yolo.py \
#     --data data/yoloworld_no_bg/data.yaml \
#     --weights yolo12s.pt \
#     --epochs 40 \
#     --batch 16 \
#     --imgsz 960 \
#     --project runs/yolo \
#     --name v12s_960_bs_16_40eps_no_bg

# python finetune_yolo.py \
#     --data data/yoloworld_no_bg/data.yaml \
#     --weights yolo12s.pt \
#     --epochs 50 \
#     --batch 16 \
#     --imgsz 960 \
#     --project runs/yolo \
#     --name v12s_960_bs_16_50eps_no_bg

