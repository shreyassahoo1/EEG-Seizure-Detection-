"""
FastAPI Backend for EEG-SINDy Seizure Prediction
Main application file
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import mne
from scipy.signal import savgol_filter
import pysindy as ps
import io
import tempfile
import os

app = FastAPI(
    title="EEG-SINDy API",
    description="Seizure prediction using Sparse Identification of Nonlinear Dynamics",
    version="1.0.0"
)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use Redis or database)
app_state = {
    "preprocessed_data": None,
    "sindy_model": None,
    "X_train": None,
    "dX_train": None
}

# Pydantic models for request/response
class PreprocessResponse(BaseModel):
    success: bool
    message: str
    raw_data: List[List[float]]
    filtered_data: List[List[float]]
    time: List[float]
    channels: int
    samples: int

class SindyResponse(BaseModel):
    success: bool
    message: str
    equations: List[dict]
    coefficients: List[List[float]]

class PredictionResponse(BaseModel):
    success: bool
    message: str
    prediction_data: List[dict]
    instability_scores: List[dict]
    alert_window: Optional[int]
    seizure_window: int
    lead_time_minutes: Optional[float]

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "EEG-SINDy API is running",
        "version": "1.0.0"
    }

@app.post("/api/preprocess", response_model=PreprocessResponse)
async def preprocess_eeg(file: UploadFile = File(...)):
    """
    Preprocess uploaded EEG file
    - Accepts .edf files
    - Applies bandpass and notch filters
    - Computes derivatives
    - Returns raw and filtered data
    """
    try:
        # Validate file type
        if not file.filename.endswith('.edf'):
            raise HTTPException(status_code=400, detail="Only .edf files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Load EEG data using MNE
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
            
            # Select first 3 channels
            channels_to_use = raw.ch_names[:3]
            raw = raw.pick_channels(channels_to_use)
            
            # Store original sampling for time axis
            original_sfreq = raw.info['sfreq']
            
            # Get raw data before filtering
            raw_data = raw.get_data().T  # Shape: (samples, channels)
            time_raw = np.arange(raw_data.shape[0]) / original_sfreq
            
            # Apply filters
            raw.filter(0.5, 40.0, fir_design='firwin', verbose=False)
            raw.notch_filter(50.0, verbose=False)
            
            # Resample to 128 Hz
            raw.resample(128, verbose=False)
            
            # Get filtered data
            filtered_data = raw.get_data().T  # Shape: (samples, channels)
            time_filtered = np.arange(filtered_data.shape[0]) / 128
            
            # Limit samples for frontend display (first 5 seconds)
            display_samples = min(640, filtered_data.shape[0])  # 5 sec * 128 Hz
            raw_display_samples = min(640, raw_data.shape[0])
            
            # Window into 10-second segments for training
            WINDOW_SEC = 10
            SFREQ = 128
            window_samples = WINDOW_SEC * SFREQ
            num_windows = filtered_data.shape[0] // window_samples
            
            X_all = []
            dX_all = []
            
            for w in range(num_windows):
                start = w * window_samples
                end = start + window_samples
                window = filtered_data[start:end]
                
                # Compute derivative per channel
                dwindow = np.zeros_like(window)
                for ch in range(window.shape[1]):
                    dwindow[:, ch] = savgol_filter(window[:, ch], 7, 3, deriv=1)
                
                X_all.append(window)
                dX_all.append(dwindow)
            
            # Store in app state
            app_state["preprocessed_data"] = {
                "X": np.array(X_all),
                "dX": np.array(dX_all),
                "filtered_full": filtered_data,
                "channels": channels_to_use
            }
            
            return PreprocessResponse(
                success=True,
                message="Preprocessing completed successfully",
                raw_data=raw_data[:raw_display_samples].tolist(),
                filtered_data=filtered_data[:display_samples].tolist(),
                time=time_filtered[:display_samples].tolist(),
                channels=len(channels_to_use),
                samples=filtered_data.shape[0]
            )
        
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@app.post("/api/preprocess/demo", response_model=PreprocessResponse)
async def preprocess_demo():
    """
    Generate and preprocess demo EEG data
    - Simulates realistic EEG signals
    - Applies same preprocessing pipeline
    """
    try:
        # Generate demo data
        samples = 1280  # 10 seconds at 128 Hz
        time = np.arange(samples) / 128
        
        # Simulate realistic EEG with multiple frequencies
        raw_data = np.zeros((samples, 3))
        for i in range(3):
            # Mix of different frequency components
            raw_data[:, i] = (
                30 * np.sin(2 * np.pi * 3 * time + i) +
                15 * np.sin(2 * np.pi * 8 * time + i*0.5) +
                10 * np.sin(2 * np.pi * 12 * time + i*0.3) +
                np.random.normal(0, 5, samples)
            )
        
        # Simulate filtering (simple attenuation)
        filtered_data = raw_data * 0.8
        
        # Add slight smoothing
        for ch in range(3):
            filtered_data[:, ch] = savgol_filter(filtered_data[:, ch], 7, 3)
        
        # Window into segments
        WINDOW_SEC = 10
        SFREQ = 128
        window_samples = WINDOW_SEC * SFREQ
        num_windows = filtered_data.shape[0] // window_samples
        
        X_all = []
        dX_all = []
        
        for w in range(num_windows):
            start = w * window_samples
            end = start + window_samples
            window = filtered_data[start:end]
            
            # Compute derivative
            dwindow = np.zeros_like(window)
            for ch in range(window.shape[1]):
                dwindow[:, ch] = savgol_filter(window[:, ch], 7, 3, deriv=1)
            
            X_all.append(window)
            dX_all.append(dwindow)
        
        # Store in app state
        app_state["preprocessed_data"] = {
            "X": np.array(X_all),
            "dX": np.array(dX_all),
            "filtered_full": filtered_data,
            "channels": ["Channel 1", "Channel 2", "Channel 3"]
        }
        
        return PreprocessResponse(
            success=True,
            message="Demo data generated successfully",
            raw_data=raw_data.tolist(),
            filtered_data=filtered_data.tolist(),
            time=time.tolist(),
            channels=3,
            samples=samples
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo generation failed: {str(e)}")

@app.post("/api/sindy/train", response_model=SindyResponse)
async def train_sindy():
    """
    Train SINDy model on preprocessed data
    - Uses polynomial library (degree 3)
    - STLSQ sparse regression
    - Returns discovered equations
    """
    try:
        if app_state["preprocessed_data"] is None:
            raise HTTPException(status_code=400, detail="No preprocessed data available. Run preprocessing first.")
        
        # Get preprocessed data
        X_windows = app_state["preprocessed_data"]["X"]
        
        # Flatten windows for training
        X_train = X_windows.reshape(-1, X_windows.shape[2])
        
        # Train SINDy model
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.006),
            feature_library=ps.PolynomialLibrary(degree=3)
        )
        
        model.fit(X_train, t=1/128)
        
        # Store model
        app_state["sindy_model"] = model
        app_state["X_train"] = X_train
        
        # Extract equations
        feature_names = [f"x{i+1}" for i in range(X_train.shape[1])]
        equations = []
        coefficients = model.coefficients()
        
        for i, channel in enumerate(feature_names):
            # Get coefficient values for this channel
            coef_row = coefficients[i]
            
            # Build equation string
            terms = []
            feature_lib = model.feature_library.get_feature_names(feature_names)
            
            for j, (coef, term) in enumerate(zip(coef_row, feature_lib)):
                if abs(coef) > 1e-10:  # Only include non-zero terms
                    if len(terms) == 0:
                        terms.append(f"{coef:.3f}{term}")
                    else:
                        sign = "+" if coef > 0 else ""
                        terms.append(f"{sign}{coef:.3f}{term}")
            
            equation_str = f"d{channel}/dt = " + " ".join(terms)
            
            equations.append({
                "id": i + 1,
                "channel": channel,
                "equation": equation_str,
                "coefficient": float(coef_row[1] if len(coef_row) > 1 else coef_row[0])
            })
        
        return SindyResponse(
            success=True,
            message="SINDy model trained successfully",
            equations=equations,
            coefficients=coefficients.tolist()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SINDy training failed: {str(e)}")

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_seizure():
    """
    Run seizure prediction using trained SINDy model
    - Simulates future EEG
    - Computes instability scores
    - Detects early warning
    """
    try:
        if app_state["sindy_model"] is None:
            raise HTTPException(status_code=400, detail="No trained model available. Train SINDy model first.")
        
        model = app_state["sindy_model"]
        X_windows = app_state["preprocessed_data"]["X"]
        
        # Simulation parameters
        WINDOW_SEC = 10
        SFREQ = 128
        t = np.arange(0, WINDOW_SEC, 1/SFREQ)
        
        # Compute prediction errors for each window
        errors = []
        prediction_data = []
        
        for i, window in enumerate(X_windows):
            try:
                # Simulate future using learned ODEs
                sim = model.simulate(window[0], t)
                
                # Compute MSE between actual and predicted
                min_len = min(len(sim), len(window))
                mse = np.mean((sim[:min_len, 0] - window[:min_len, 0])**2)
                errors.append(mse)
                
                # Store some prediction data for visualization (first window only)
                if i == 0:
                    for j in range(min(len(t), 150)):
                        prediction_data.append({
                            "time": float(t[j]),
                            "actual": float(window[j, 0]),
                            "predicted": float(sim[j, 0]) if j < len(sim) else float(window[j, 0])
                        })
            except:
                errors.append(errors[-1] if errors else 0.01)
        
        errors = np.array(errors)
        
        # Determine threshold (95th percentile of first 50 windows)
        baseline_errors = errors[:min(50, len(errors))]
        threshold = np.percentile(baseline_errors, 95) if len(baseline_errors) > 0 else 0.015
        
        # Detect alert (3 consecutive windows above threshold)
        CONSEC_WINDOWS = 3
        alert_window = None
        
        for i in range(len(errors) - CONSEC_WINDOWS):
            if np.all(errors[i:i+CONSEC_WINDOWS] > threshold):
                alert_window = i
                break
        
        # Simulate seizure at window 297 (or near end)
        seizure_window = min(297, len(errors) - 10)
        
        # Calculate lead time
        lead_time = None
        if alert_window is not None:
            lead_time = (seizure_window - alert_window) * WINDOW_SEC / 60  # in minutes
        
        # Prepare instability scores for frontend
        instability_scores = []
        for i, error in enumerate(errors):
            instability_scores.append({
                "window": i,
                "error": float(error),
                "threshold": float(threshold)
            })
        
        return PredictionResponse(
            success=True,
            message="Seizure prediction completed successfully",
            prediction_data=prediction_data,
            instability_scores=instability_scores,
            alert_window=alert_window,
            seizure_window=seizure_window,
            lead_time_minutes=round(lead_time, 2) if lead_time else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/status")
async def get_status():
    """Get current pipeline status"""
    return {
        "preprocessed": app_state["preprocessed_data"] is not None,
        "model_trained": app_state["sindy_model"] is not None,
        "ready_for_prediction": app_state["sindy_model"] is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    