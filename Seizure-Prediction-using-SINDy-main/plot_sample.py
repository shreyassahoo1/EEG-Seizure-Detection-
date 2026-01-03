import mne
import matplotlib.pyplot as plt

# Load a seizure EEG file from CHB01
file_path = "data/raw/chb01_03.edf"

print("Loading EEG file...")
raw = mne.io.read_raw_edf(file_path, preload=True)
print(raw)

# Show basic info
print("\nChannels:", raw.info["ch_names"])
print("Sampling rate:", raw.info["sfreq"])

# Pick first channel
channel = raw.ch_names[0]

# Plot the first 20 seconds
raw.plot(start=0, duration=20, title=f"EEG Signal - {channel}")

plt.show()
