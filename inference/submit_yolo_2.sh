python submit_yolo.py \
  --weights runs/yolo/v8s_640_bs_16_30eps_no_bg_exp1/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out2/yolo/v8s_640_bs_16_30eps_no_bg_exp1 \
  --conf 0.3 \
  --device 1
  
python submit_yolo.py \
  --weights runs/yolo/v8s_640_bs_16_30eps_no_bg_exp2/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out2/yolo/v8s_640_bs_16_30eps_no_bg_exp2 \
  --conf 0.3 \
  --device 1
  
python submit_yolo.py \
  --weights runs/yolo/v8s_640_bs_16_30eps_no_bg_exp3/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out2/yolo/v8s_640_bs_16_30eps_no_bg_exp3 \
  --conf 0.3 \
  --device 1

python submit_yolo.py \
  --weights runs/yolo/v8s_640_bs_16_30eps_no_bg_exp4/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out2/yolo/v8s_640_bs_16_30eps_no_bg_exp4 \
  --conf 0.3 \
  --device 1

python submit_yolo.py \
  --weights runs/yolo/v8s_640_bs_16_30eps_no_bg_exp5/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out2/yolo/v8s_640_bs_16_30eps_no_bg_exp5 \
  --conf 0.3 \
  --device 1

python submit_yolo.py \
  --weights runs/yolo/v8s_640_bs_16_30eps_no_bg_exp6/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out2/yolo/v8s_640_bs_16_30eps_no_bg_exp6 \
  --conf 0.3 \
  --device 1

python submit_yolo.py \
  --weights runs/yolo/v8s_960_bs_16_30eps_no_bg_exp4/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out2/yolo/v8s_960_bs_16_30eps_no_bg_exp4 \
  --conf 0.3 \
  --imgsz 960 \
  --device 1

python submit_yolo.py \
  --weights runs/yolo/v8s_960_bs_16_30eps_no_bg_exp5/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out2/yolo/v8s_960_bs_16_30eps_no_bg_exp5 \
  --conf 0.3 \
  --imgsz 960 \
  --device 1

python submit_yolo.py \
  --weights runs/yolo/v8s_960_bs_16_30eps_no_bg_exp6/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out2/yolo/v8s_960_bs_16_30eps_no_bg_exp6 \
  --conf 0.3 \
  --imgsz 960 \
  --device 1
  
