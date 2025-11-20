python predict_yolo_tiles.py \
  --weights runs/yolo_tiling/v8s_640_bs_16_20eps_no_bg_tiling/weights/best.pt \
  --frames_root data/public_test_frames \
  --out_dir out/sub_wbf_split \
  --device 0 \
  --wbf-split \
  --save_vis