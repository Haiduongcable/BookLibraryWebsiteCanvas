python predict_yoloworld.py \
    --model yolov8s-worldv2.pt \
    --input /root/zac/aero-eyes/data/public_test_frames/CardboardBox_0/CardboardBox_0_frame_002255.jpg \
    --classes "object" \
    --conf 0.001 \
    --output vis_test