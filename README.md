# SymptoSense / Project-360

* AI-based symptom checker with two-round questionnaire (initial screening + pinpointing).
* Uses MongoDB backend for persistence + clinician feedback learning.
* Aims to triage ~10–15 common diseases with structured symptom categories (hallmark/core/associated/general).

---

# ECG Abnormality Detection

* Classical ML models (LightGBM, RandomForest, XGBoost) trained on PTB-XL / extracted features.
* Optimized with Optuna, evaluated with accuracy + macro F1.
* Includes interpretability with SHAP and a Streamlit inference UI.

---

# Potassium Extraction from ECG (VitalDB)

* Uses ECG_II and lab_data.csv from VitalDB to estimate serum potassium levels (K-value).
* Regression for estimated K (MAE, RMSE) + classification for severity ranges (hypo/hyper/normal).
* Pipeline: preprocessing → feature extraction → resampling → regression + severity classifier → GUI with report.

---

# Respiratory Features / Pulse2Breath (PPG)

* Toolkit to derive respiratory metrics from PPG signals.
* Modes: simulation, file playback, or serial input from USB-to-UART device.
* Extracts RR (time + frequency domain), breath count, synchronized views (PPG waveform, respiratory proxy, PSD).
* Target: real-time monitoring, hospital-like live plots.
