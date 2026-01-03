import mne
import numpy as np
import os
from scipy.signal import savgol_filter

# Load raw data
file_path = "data/raw/chb01_03.edf"
raw = mne.io.read_raw_edf(file_path, preload=True)

# Filter again for safety
raw.filter(0.5, 40.0)
raw.notch_filter(50.0)

# Downsample from 256Hz to 128Hz
raw.resample(128)

# Pick seizure-relevant channels
channels = ["T7-P7", "F7-T7", "FP1-F7"]
raw.pick_channels(channels)

# Convert to numpy
data = raw.get_data()  # shape: [channels, samples]
sfreq = raw.info["sfreq"]

window_size = int(10 * sfreq)  # 10 seconds
step = window_size  # non-overlapping windows

processed_dir = "data/processed"
os.makedirs(processed_dir, exist_ok=True)

X_list, dX_list = [], []

# Sliding window segmentation
for start in range(0, data.shape[1] - window_size, step):
    window = data[:, start:start + window_size]
    
    # Compute derivative using Savitzky–Golay
    dwindow = savgol_filter(window, 7, 3, deriv=1)
    
    X_list.append(window.T)
    dX_list.append(dwindow.T)

X = np.array(X_list)   # shape: (windows, samples, channels)
dX = np.array(dX_list)

np.save(f"{processed_dir}/X.npy", X)
np.save(f"{processed_dir}/dX.npy", dX)

print("Saved processed files to data/processed/:")
print("X.npy =", X.shape)
print("dX.npy =", dX.shape)
