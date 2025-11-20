from ultralytics.models.fastsam import FastSAMPredictor
from pathlib import Path
import cv2

# 1) Tạo predictor với config
overrides = dict(
    conf=0.5,
    task="segment",
    mode="predict",
    model="FastSAM-s.pt",  # đổi path nếu cần
    imgsz=1024,
    save=False,
)
predictor = FastSAMPredictor(overrides=overrides)

img_path = "/root/zac/aero-eyes/data/public_test_frames/LifeJacket_1/LifeJacket_1_frame_001907.jpg"
texts = ["life jacket"]  # hoặc ["xe ô tô", "biển báo"]

# 2) Segment everything
everything_results = predictor(img_path)

# 3) Dùng text prompt để filter mask
prompt_results = predictor.prompt(
    everything_results,
    texts=texts,        # có thể là string hoặc list string
)

# prompt_results thường là list Results, lấy phần tử đầu
res = prompt_results[0]

# 4) Vẽ lên ảnh và lưu
vis = res.plot()  # BGR np.ndarray
out_path = "output_prompt_vis.jpg"
cv2.imwrite(out_path, vis)
print(f"Saved visualization to {out_path}")
