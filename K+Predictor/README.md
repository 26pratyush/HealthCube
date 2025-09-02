# K⁺ Severity Inference Explorer

## Overview

This project is an AI-based potassium ($K^{+}$) severity prediction tool built on ECG-derived features. It provides both quantitative estimates of potassium levels (mmol/L) and a clinical severity classification (e.g., severe\_hypo, mild\_hypo, normal, hyper). The system is wrapped in a Streamlit UI for interactive exploration.

## What Has Been Done

* **Feature extraction:** ECG and HRV features were computed from raw biosignals.
* **Model training:** Machine learning models were trained to predict both serum potassium levels and severity class.
* **Inference dashboard:** Built using Streamlit with adjustable sample size, random seed, severity comparison, feature snapshots, and report download.

## Accuracy Constraints

* **Estimation error:** Predictions may deviate by $0.1–0.3 \text{ mmol/L}$ compared to lab values.
* **Class overlap:** Borderline cases (e.g., K = 3.4 vs 3.5) reduce severity classification accuracy.
* **Data dependence:** Performance is limited by dataset size, signal quality, and ECG noise.
* **Fragmented workflow:** Current system requires multiple separate scripts for different stages.

## Future Work

* **Unified pipeline:** Convert into a single-step pipeline where raw ECG → preprocessing → features → inference is automated.
* **Improved accuracy:** Use multimodal signals (PPG, LDF, vitals) and stronger ML models.
* **Explainability:** Integrate SHAP explanations directly into the UI.
* **Scalability:** Containerize and deploy for integration into hospital or research systems.

## Project Structure

* `1_cleaned_dataset.csv` – Initial cleaned dataset after preprocessing.
* `2_assessment.py` – Script for assessment and exploratory analysis of features.
* `3_resolution.py` – Data resolution and additional preprocessing logic.
* `4_resampling_&_modelling.py` – Core resampling and model training script.
* `6_test.py` – Script to test and validate trained models.
* `7_app.py` – Streamlit UI application for inference and exploration.
* `deployment_5.py` – Deployment script to wrap the model for end-to-end use.
* `feature_schema.json` – Schema and feature definitions required by the model.
* `vital_clinical_model.pkl` – Trained ML model saved for inference.
* `vital_enhanced_features.csv` – Feature dataset used for model training.
* `vital_enhanced_targets.csv` – Target labels used for training.
* `vital_enhanced_meta.json` – Metadata describing features and samples.
* `vital_prepared_dataset.csv` – Final prepared dataset used in the pipeline.

## How to Run

1.  **Install requirements:** `pip install -r requirements.txt`
2.  **Start the app:** `streamlit run 7_app.py`
3.  Adjust sample size and seed to explore predictions interactively.

## Disclaimer

This is a research prototype and not a medical diagnostic tool. Predictions should not be used for clinical decision-making without further validation.
