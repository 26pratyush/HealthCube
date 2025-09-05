# 🩺 HealthCube Internship Projects (July–Sept 2025)

This repository contains the work completed during my HealthCube internship, where I was assigned to develop and prototype AI-driven solutions for multimodal health diagnostics.
The internship was focused on signal processing, machine learning, and clinical decision support, spanning four major projects.

Each project directory includes detailed instructions, modular code, and demo videos showing workflows and UI functionality.

### 📂 Projects

**1. Project-360 / SymptoSense**
* **Goal**: Build an AI symptom-checker tool with a two-round questionnaire system.
* **Rounds**:
    * Round 1: Initial checklist of symptoms (screening stage) → narrows down likely conditions.
    * Round 2: Adaptive, one-by-one follow-up questions → pinpoints the disease with confidence.
* **Backend**: MongoDB stores cases, symptom–disease mappings, and clinician feedback (doctor corrections feed into the system to improve over time).
* **Scope**: 34 common diseases and condition (Dengue, Typhoid, Respiratory Infections, Hyperuricemia (High Uric Acid) etc.) with structured symptoms (hallmark, core, associated, general).
* **Deliverables**:
    * Streamlit interface for testing.
    * Disease database with question sets.
    * Scripts for DB setup and seeding.

**2. ECG Abnormality Detection / Binary Classifier**
* **Goal**: Detect whether ECG signals indicate normal vs abnormal cardiac activity.
* **Dataset**: PTB-XL (2000 records after cleaning).
* **Pipeline**: Preprocessing → Feature extraction → Feature selection → ML model training.
* **Models**: Random Forest, LightGBM, XGBoost with Optuna hyperparameter tuning.
* **Evaluation**: Accuracy, Macro F1, Confusion Matrix.
* **Interpretability**: SHAP for feature importance & per-patient explanations.
* **Deliverables**:
    * Training + inference scripts.
    * Streamlit UI with CSV upload → prediction + explanation visualization.

**3. Potassium Extraction from ECG / K+ Predictor**
* **Goal**: Predict serum potassium concentration (K level) and classify severity.
* **Data**: VitalDB .vital ECG signals + lab_data.csv.
* **Pipeline**: Signal processing → Feature extraction → Resampling → Regression + Severity Classification.
* **Outputs**:
    * **Regression**: Estimated potassium level.
    * **Classification**: Severity categories (normal, mild hypo, moderate hypo, severe hypo, hyperkalemia).
* **Evaluation Metrics**:
    * MAE, RMSE for regression.
    * Accuracy, Macro F1, Confusion Matrix for severity.
* **Deliverables**:
    * End-to-end pipeline scripts.
    * GUI to select patient, run prediction, and download report.

**4.Respiratory Features from PPG / Pulse2Breath**
* **Goal**: Extract respiratory metrics from PPG and provide real-time monitoring.
* **Methods**:
    * Smooth PPG → Low-pass filter to respiratory band.
    * Derive Respiration Rate (RR) by FFT peak & time-domain counts.
    * Display synchronized views: PPG waveform, Respiration proxy, Power Spectral Density.
* **Modes**:
    * Normal (plotting pre-recorded file as a whole).
    * File playback (real-time plotting and feature extraction of a pre-recorded file).
    * Serial (live USB-to-UART device input plotting).
* **Deliverables**:
    * Tkinter UI with live plotting.
    * Feature estimation and insights (RR, breath count, etc.).

---

### 📊 Common Workflow Across Projects

* Data acquisition & preprocessing
* Feature extraction (ECG/PPG/metadata)
* Model development & evaluation
* Visualization & interpretability
* Interactive GUI for usability and testing

---

### 🔮 Future Work / Recommendations for Next Intern

* **SymptoSense**:
    * Expand disease coverage beyond the current 34
    * Integrate NLP for richer free-text symptom extraction.
    * Add real patient data validation in collaboration with clinicians.
* **ECG Abnormality Detection**:
    * Move toward deep learning (CNNs on raw ECG).
    * Extend classification to multiple cardiac disorders (AFib, MI, etc.).
    * Automated pipeline -> From ECG raw signal to abnormality classification
* **Potassium Extraction**:
    * Explore multimodal features (combine ECG + PPG + labs).
    * Improve regression accuracy using ensemble models.
    * Validate with larger real-world datasets.
* **Pulse2Breath**:
    * Optimize live plotting for hospital-grade stability (straight from the PPG device)
    * Add more respiratory metrics (tidal volume proxies, apnea detection).
    * Connect with HealthCube’s other vitals for integrated dashboards.

---

### 📁 Repository Contents

Each project folder includes:
* **Code**: Python scripts and IPYNB notebooks.
* **Docs**: Setup + usage instructions.
* **Usage**: Video Demonstrations.
* **Reports/READMEs**: Evaluation summaries and key findings.

---

### 🚀 Usage

* Clone the repository and maintain folder structure.
* Navigate into each project folder for specific instructions.
* Scripts are modular and numbered (e.g., `step1_*.py`, `step2_*.py`) for clarity, such that they can be run in the right order.

---

### 🙌 Acknowledgements

* HealthCube Team for guidance and providing problem statements.
* Open-source datasets (PTB-XL, VitalDB) for enabling experiments.

👉 This repository not only documents completed internship work but also serves as a foundation for future interns to build on HealthCube’s mission of making diagnostics faster, accessible, and multimodal.
