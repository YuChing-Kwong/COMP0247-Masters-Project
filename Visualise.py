import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import welch


# Path
frames_path = r"./Recordings/Testing_npy/participant_001_20250823_183623.npy"
rppg_path = r"./Recordings/Testing_npy/output_rppg.npy"
fps = 30  # Frame rate of the video
window_sec = 8  # Heart rate estimation window (seconds)


# Load data
frames = np.load(frames_path)  # (T,H,W,C)
rppg = np.load(rppg_path)      # (T-1,)
T, H, W, C = frames.shape
rppg_full = np.concatenate([rppg, [rppg[-1]]])  # Align frame numbers


# Heart rate estimation function
def estimate_hr(signal, fs=30):
    f, pxx = welch(signal, fs=fs, nperseg=min(len(signal), 256))
    mask = (f >= 0.7) & (f <= 3.0)  # 42–180 bpm
    if not np.any(mask):
        return np.nan
    f_hr = f[mask][np.argmax(pxx[mask])]
    return f_hr * 60  # bpm


# Initialize plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 10))

# Video frame display
im = ax1.imshow(frames[0])
ax1.axis('off')
title1 = ax1.set_title(f"Frame 1/{T}")

# rPPG waveform display
line, = ax2.plot([], [], color='red')
ax2.set_xlim(0, T)
ax2.set_ylim(rppg_full.min(), rppg_full.max())
ax2.set_title("rPPG over time")
ax2.set_xlabel("Frame index")
ax2.set_ylabel("rPPG amplitude")

# Dynamic display of HR numbers
text_hr = ax2.text(0.02, 0.9, "", transform=ax2.transAxes, color="blue", fontsize=12)

# HR curve display
hr_times = []
hr_values = []
line_hr, = ax3.plot([], [], color='blue')
ax3.set_xlim(0, T / fps)
ax3.set_ylim(40, 180)  # bpm range
ax3.set_title("Estimated Heart Rate over time")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("HR (bpm)")

# Update function
def update(frame_idx):
    # Update video frame
    im.set_array(frames[frame_idx])
    title1.set_text(f"Frame {frame_idx+1}/{T}")

    # Update rPPG waveform
    line.set_data(np.arange(frame_idx+1), rppg_full[:frame_idx+1])

    # Refresh HR every 1 second
    if frame_idx > window_sec * fps and frame_idx % fps == 0:
        start = frame_idx - window_sec * fps
        signal_window = rppg_full[start:frame_idx]
        hr = estimate_hr(signal_window, fs=fps)

        if not np.isnan(hr):
            sec = frame_idx / fps
            hr_times.append(sec)
            hr_values.append(hr)

            # Update displayed HR
            text_hr.set_text(f"HR: {hr:.1f} bpm")

            # Update HR curve
            line_hr.set_data(hr_times, hr_values)
            ax3.set_xlim(0, max(10, sec))  # Dynamically expand x-axis

    return [im, line, text_hr, line_hr]


# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=T,
    interval=int(1000/fps),
    blit=False
)

plt.tight_layout()
plt.show()
