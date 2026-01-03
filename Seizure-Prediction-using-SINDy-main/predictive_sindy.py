import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
BASELINE_X = "data/processed/X_baseline.npy"
SEIZURE_X = "data/processed/X.npy"    # From seizure file (03)
WINDOW_SEC = 10
SFREQ = 128
SEIZURE_IDX = 297     # from chb01_03

CONSEC_ALERT_WINDOWS = 3
SMOOTH_K = 5

# ---------------------------------------------
# 1) LOAD DATA
# ---------------------------------------------
print("Loading baseline and seizure EEG windows...")
X_base = np.load(BASELINE_X)   # baseline windows
X_test = np.load(SEIZURE_X)   # seizure windows (03)

print(f"Baseline: {X_base.shape}, Seizure: {X_test.shape}")
n_windows = X_test.shape[0]
n_channels = X_test.shape[2]

# Flatten baseline for training
X_train = X_base.reshape(-1, n_channels)

print(f"Training SINDy on {len(X_train)} baseline samples...")


# ---------------------------------------------
# 2) TRAIN SINDY BASELINE MODEL
# ---------------------------------------------
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.006),
    feature_library=ps.PolynomialLibrary(degree=3)  # IMPORTANT IMPROVEMENT
)

model.fit(X_train, t=1/SFREQ)
print("\n📌 Baseline Brain Dynamics Learned:")
model.print()


# ---------------------------------------------
# 3) TEST: PREDICTION ERROR ACROSS SEIZURE FILE
# ---------------------------------------------
errors = []
t = np.arange(0, WINDOW_SEC, 1/SFREQ)

print("\nComputing prediction errors...")
for i, w in enumerate(X_test):
    sim = model.simulate(w[0], t)  # integrate ODE
    mse = np.mean((sim[:len(w),0] - w[:len(sim),0])**2)
    errors.append(mse)

    if i % 50 == 0:
        print(f" Window {i}/{n_windows} done")

errors = np.array(errors)
print("Error computation complete.")


# ---------------------------------------------
# 4) THRESHOLD
# ---------------------------------------------
threshold = np.percentile(errors[:50], 95)  # first 50 clearly normal

print(f"\n📌 Alert Threshold = {threshold:.3e}")


# ---------------------------------------------
# 5) ALERT DETECTION + LEAD TIME
# ---------------------------------------------
alert_idx = None
for i in range(n_windows - CONSEC_ALERT_WINDOWS):
    if np.all(errors[i:i+CONSEC_ALERT_WINDOWS] > threshold):
        alert_idx = i
        break

if alert_idx:
    lead_time = (SEIZURE_IDX - alert_idx)*WINDOW_SEC/60
    print(f"\n🚨 ALERT at window {alert_idx}")
    print(f"⏱ Lead Time ≈ {lead_time:.2f} minutes!")
else:
    print("\n⚠ No early alert detected!")


# ---------------------------------------------
# 6) VISUALIZE FITS
# ---------------------------------------------
def plot_fit(idx, title):
    w = X_test[idx]
    sim = model.simulate(w[0], t)

    plt.figure(figsize=(10,4))
    plt.plot(t, w[:,1], label="Actual (x1)")  # Oscillatory channel
    plt.plot(t, sim[:,1], label="Predicted (x1)")
    plt.title(f"{title} (window {idx})")
    plt.legend()
    plt.grid(True)
    plt.show()

print("\nPlotting normal, alert, and seizure fits...")
plot_fit(30, "Normal")
if alert_idx: plot_fit(alert_idx, "Alert")
plot_fit(SEIZURE_IDX, "Seizure")


# ---------------------------------------------
# 7) PLOT INSTABILITY SCORE
# ---------------------------------------------
def smooth(a, k):
    pad = k//2
    a = np.pad(a,(pad,pad),mode="edge")
    return np.convolve(a,np.ones(k)/k,mode="valid")

errors_s = smooth(errors, SMOOTH_K)

plt.figure(figsize=(12,5))
plt.plot(errors_s, label="Instability Score", linewidth=2)
plt.axhline(threshold, linestyle="--", color="green", label="Threshold")
plt.axvline(SEIZURE_IDX, color="red", linestyle="--", label="Seizure Onset")
if alert_idx: plt.axvline(alert_idx, color="purple", linestyle="--", label="Alert")
plt.xlabel("Window Index (10s each)")
plt.ylabel("Error (MSE)")
plt.title("Dynamic Instability vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

