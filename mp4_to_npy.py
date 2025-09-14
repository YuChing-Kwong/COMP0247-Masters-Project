import cv2
import numpy as np
from pathlib import Path


# Paths
video_path = r"./Recordings/Testing/participant_001_20250905_143615.mp4"
output_dir = r"./Recordings/Testing_npy"
output_name = "participant_001_20250905_143615.npy"

Path(output_dir).mkdir(parents=True, exist_ok=True)

# Parameters
target_size = (72, 72)   # Model input size
target_fps  = 30         # Target FPS
normalize   = True       # Whether to normalize to 0-1

# ==== Read and convert ====
cap = cv2.VideoCapture(video_path)
orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0
interval = 1 if not orig_fps else max(1, round(orig_fps / target_fps))

frames, idx = [], 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx % interval == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        if normalize:
            frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    idx += 1
cap.release()

frames = np.array(frames, dtype=np.float32)  # shape: (T, 72, 72, 3)

# Save
save_path = Path(output_dir) / output_name
np.save(save_path, frames)

print(f"Save successful: {save_path}, shape={frames.shape}")
