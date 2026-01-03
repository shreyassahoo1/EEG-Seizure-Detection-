import os
import numpy as np
import mne
from scipy.signal import savgol_filter

RAW_DIR = "data/raw"
SAVE_DIR = "data/processed"

os.makedirs(SAVE_DIR, exist_ok=True)

FILES = ["chb01_01.edf", "chb01_02.edf"]

WINDOW_SEC = 10
SFREQ_NEW = 128

X_all = []
dX_all = []

for file in FILES:
    file_path = os.path.join(RAW_DIR, file)
    print(f"\nLoading {file_path} ...")

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw = raw.pick_channels(raw.ch_names[:3])  # keep 3 channels
    raw.load_data()

    # Filtering
    raw.filter(0.5, 40.0, fir_design="firwin", verbose=False)
    raw.notch_filter(50.0, verbose=False)

    # Resample
    raw.resample(SFREQ_NEW, verbose=False)
    data = raw.get_data().T  # shape: (samples, channels)

    window_samples = WINDOW_SEC * SFREQ_NEW
    num_windows = data.shape[0] // window_samples

    print(f" → Windows: {num_windows}")

    for w in range(num_windows):
        start = w * window_samples
        end = start + window_samples
        window = data[start:end]

        # Derivative per channel
        dwindow = np.zeros_like(window)
        for ch in range(window.shape[1]):
            dwindow[:, ch] = savgol_filter(window[:, ch], 7, 3, deriv=1)

        X_all.append(window)
        dX_all.append(dwindow)

# Convert to arrays
X_all = np.array(X_all)
dX_all = np.array(dX_all)

np.save(os.path.join(SAVE_DIR, "X_baseline.npy"), X_all)
np.save(os.path.join(SAVE_DIR, "dX_baseline.npy"), dX_all)

print("\n🎉 Baseline preprocessing complete!")
print(f"Saved: {X_all.shape}, {dX_all.shape}")
