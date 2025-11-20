python submit_yoloworld.py \
  --weights runs/yoloworld/yoloworld_v8s_20eps_bs64_exp2/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir output/yoloworld_v8s_20eps_bs64_exp2 \
  --conf 0.001 --iou 0.7 --imgsz 640 --filter-box 0.015 \
  --device 0

python submit_yoloworld.py \
  --weights runs/yoloworld/yoloworld_v8s_20eps_bs64_exp3/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir output/yoloworld_v8s_20eps_bs64_exp3 \
  --conf 0.001 --iou 0.7 --imgsz 640 --filter-box 0.015 \
  --device 0


python submit_yoloworld.py \
  --weights runs/yoloworld/yoloworld_v8s_20eps_bs64_exp4/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir output/yoloworld_v8s_20eps_bs64_exp4 \
  --conf 0.001 --iou 0.7 --imgsz 640 --filter-box 0.015 \
  --device 0

python submit_yoloworld.py \
  --weights runs/yoloworld/yoloworld_v8s_30eps_bs32_exp1/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir output/yoloworld_v8s_30eps_bs32_exp1 \
  --conf 0.001 --iou 0.7 --imgsz 640 --filter-box 0.015 \
  --device 0


python submit_yoloworld.py \
  --weights runs/yoloworld/yoloworld_v8s_30eps_bs32_exp2_v3/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir output/yoloworld_v8s_30eps_bs32_exp2_v3 \
  --conf 0.001 --iou 0.7 --imgsz 640 --filter-box 0.015 \
  --device 0

python submit_yoloworld.py \
  --weights runs/yoloworld/yoloworld_v8s_30eps_bs32exp1_v3/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir output/yoloworld_v8s_30eps_bs32exp1_v3 \
  --conf 0.001 --iou 0.7 --imgsz 640 --filter-box 0.015 \
  --device 0



python submit_yoloworld.py \
  --weights runs/yoloworld/yoloworld_v8s_40eps_bs32_exp2_v3/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir output/yoloworld_v8s_40eps_bs32_exp2_v3 \
  --conf 0.001 --iou 0.7 --imgsz 640 --filter-box 0.015 \
  --device 0