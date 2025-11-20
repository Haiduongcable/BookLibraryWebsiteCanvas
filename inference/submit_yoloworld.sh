python submit_yoloworld.py \
  --weights runs/yoloworld/yoloworld_no_bg_v8s_20eps_bs32exp1/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out/yoloworld_no_bg_v8s_20eps_bs32exp1 \
  --conf 0.001 --iou 0.7 --imgsz 640 --filter-box 0.015 \
  --device 1

python submit_yoloworld.py \
  --weights runs/yoloworld/yoloworld_no_bg_v8s_20eps_bs32_exp2/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out/yoloworld_no_bg_v8s_20eps_bs32_exp2 \
  --conf 0.001 --iou 0.7 --imgsz 640 --filter-box 0.015 \
  --device 1


python submit_yoloworld.py \
  --weights runs/yoloworld/yoloworld_no_bg_v8s_20eps_bs32exp3/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out/yoloworld_no_bg_v8s_20eps_bs32exp3 \
  --conf 0.001 --iou 0.7 --imgsz 640 --filter-box 0.015 \
  --device 1

python submit_yoloworld.py \
  --weights runs/yoloworld/yoloworld_no_bg_v8s_20eps_bs32_exp4/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out/yoloworld_no_bg_v8s_20eps_bs32_exp4 \
  --conf 0.001 --iou 0.7 --imgsz 640 --filter-box 0.015 \
  --device 1


python submit_yoloworld.py \
  --weights runs/yoloworld/yoloworld_no_bg_v8s_20eps_bs32exp5/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out/yoloworld_no_bg_v8s_20eps_bs32exp5 \
  --conf 0.001 --iou 0.7 --imgsz 640 --filter-box 0.015 \
  --device 1

