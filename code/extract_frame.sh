#Docker 
# python /code/extract_test_frames.py \
#   --root /data \
#   --out  /data/extracted_frames \
#   --frame_stride 1 \
#   --max_frames_per_video 10000 \
#   --workers 8

#local
python code/extract_frame.py \
  --root data \
  --out  data/extracted_frames \
  --frame_stride 1 \
  --max_frames_per_video 10000 \
  --workers 12
