import torch
import numpy as np
from pathlib import Path
from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys, model_config


# Paths
npy_path = r"./Recordings/Testing_npy/participant_001_20250905_143615.npy"
checkpoint_path = r"./FactorizePhys/final_model_release/iBVP_FactorizePhys_FSAM.pth"
output_path = r"./Recordings/Testing_npy/output_rppg_20250905_143615.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"


# 1. Load video frames npy
frames = np.load(npy_path)   # shape: (T, H, W, C)
T, H, W, C = frames.shape

# Convert to model input (B, C, T, H, W)
frames = np.transpose(frames, (3, 0, 1, 2))   # (C, T, H, W)
frames = np.expand_dims(frames, 0)            # (1, C, T, H, W)
frames = torch.from_numpy(frames).float().to(device)


# 2. Initialize model + load pre-trained weights
md_config = model_config.copy()
frames_len = frames.shape[2]

model = FactorizePhys(frames=frames_len, md_config=md_config, in_channels=3, device=device, debug=False)
model.to(device)

# Load pre-trained weights
state_dict = torch.load(checkpoint_path, map_location=device)

# If the weights are saved with DataParallel, remove the "module." prefix
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k[7:]] = v  # Remove "module."
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict=False)
print(f"Loaded pre-trained weights from {checkpoint_path}")

model.eval()


with torch.no_grad():
    rppg, voxel_embeddings = model(frames)[:2]

rppg = rppg.squeeze().cpu().numpy()   # shape (length-1,)
print("rPPG shape:", rppg.shape)


# 4. Save results

np.save(output_path, rppg)
print("rPPG saved to:", output_path)