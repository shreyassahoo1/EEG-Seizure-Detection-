# 🧠 EEG Seizure Detection Using SINDy & Machine Learning

---

## 📌 Overview
This project implements a **seizure detection system** using Electroencephalogram (EEG) data and modern data-driven modeling techniques.  
It applies **Sparse Identification of Nonlinear Dynamics (SINDy)** along with signal processing and classification methods to detect epileptic seizure events from EEG recordings.

---

## 🎯 Motivation
EEG-based seizure detection is important for:
- Real-time patient monitoring
- Automated healthcare systems
- Reducing manual review workload
- Enhancing diagnostic accuracy

This system demonstrates how machine learning and dynamical system modeling can be applied to real-world biomedical signal analysis.

---

## 📂 Project Structure
EEG-Seizure-Detection/
│
├── data/ # EEG dataset files (if included or referenced)
├── preprocessing.py # EEG data cleaning and filtering
├── feature_extraction.py # Feature extraction module
├── sindy_model.py # SINDy model implementation
├── classifier.py # Classification & detection logic
├── visualization.py # Plots and result visualization
│
├── README.md
├── requirements.txt
└── .gitignore

---

## 🧠 Key Modules

### `preprocessing.py`
- Loads raw EEG signals
- Filters noise and artifacts
- Normalizes time series

### `feature_extraction.py`
- Extracts meaningful EEG features
- Frequency domain and time domain characteristics

### `sindy_model.py`
- Builds a SINDy model for underlying EEG dynamics
- Identifies sparse governing equations

### `classifier.py`
- Uses feature and dynamic behavior for seizure classification
- Metrics evaluation

### `visualization.py`
- Plots EEG signals and detection results
- Helps in analysis and interpretation

---

## ⚖️ Disclaimer & Ownership

This project is an original academic work developed by Shreyas Sahoo along with contributing team members as part of coursework and learning in EEG Signal Processing.

All source code, system design, algorithms, and architectural decisions are jointly authored by the project owner and the team members involved in the development of this SINDy-based EEG seizure detection system.

This project is intended strictly for educational and academic use.
Unauthorized commercial use, redistribution without proper credit, or plagiarism of this work—either in full or in part—is not permitted.

The authors assume no responsibility for any misuse of this project or its outcomes.
If this project is referenced, reused, or built upon, clear and proper attribution to the authors and team members is mandatory.
