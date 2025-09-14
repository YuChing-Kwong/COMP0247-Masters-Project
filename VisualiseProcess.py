import numpy as np
import matplotlib.pyplot as plt

# Load frames
frames = np.load(r"./Recordings/Testing_npy/participant_001_20250823_183623.npy")  # shape (T,H,W,C)

# Select a pixel position to observe
pixel = (36, 36)  # Center point
r = frames[:, pixel[0], pixel[1], 0]  # R channel over time
g = frames[:, pixel[0], pixel[1], 1]
b = frames[:, pixel[0], pixel[1], 2]

plt.figure(figsize=(12,4))
plt.plot(r, label="R channel")
plt.plot(g, label="G channel")
plt.plot(b, label="B channel")
plt.title("Raw pixel intensity over time")
plt.xlabel("Frame index")
plt.ylabel("Pixel value")
plt.legend()
plt.show()

# Compute diff
r_diff = np.diff(r)
plt.figure(figsize=(12,4))
plt.plot(r_diff, label="diff(R)")
plt.title("Temporal difference of R channel (what model sees)")
plt.xlabel("Frame index")
plt.ylabel("Diff value")
plt.legend()
plt.show()
