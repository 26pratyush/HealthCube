# K⁺ Severity Inference Explorer

## Overview

This project is an AI-based potassium ($K^{+}$) severity prediction tool built on ECG-derived features. It provides both quantitative estimates of potassium levels (mmol/L) and a clinical severity classification (e.g., severe\_hypo, mild\_hypo, normal, hyper). The system is wrapped in a Streamlit UI for interactive exploration.

## Methodology

**Data and Feature Extraction**

* The project is built on the VitalDB dataset, a large collection of multi-signal and multi-device physiological recordings. From ECG waveforms (primarily lead II), features were extracted to capture both heart rate variability (HRV), morphological information and patient metadata. Time-domain HRV measures included MeanNN, SDNN, RMSSD, and SDSD, while morphological descriptors such as ECG rate and beat-to-beat variability were also computed. These features were consolidated into structured datasets linking each patient record with corresponding serum potassium (K⁺) values and categorical severity labels (normal, mild hypo/hyper, severe hypo/hyper). The pipeline generated several artifacts: cleaned datasets, prepared feature-target matrices, and enhanced versions with resampling , ensuring reproducibility and balance across classes.

**Modeling and Inference**

* Using the extracted feature datasets, machine learning models were developed with scikit-learn. The modeling phase explored supervised classifiers to predict both continuous potassium levels and categorical severity. To mitigate class imbalance, resampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) were applied. The best-performing model was serialized as vital_clinical_model.pkl for inference. An interactive Streamlit dashboard was then implemented, allowing users to explore patient-level predictions, compare estimated vs. true potassium, review severity matches, and download structured HTML reports.

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

1.  **Clone the Repository** 
2.  Ensure file structure remains intact
3.  **Install requirements:** `pip install -r requirements.txt`
4.  **Start the app:** `streamlit run 7_app.py`
5.  Adjust sample size and seed to explore predictions interactively.

## Disclaimer

This is a research prototype and not a medical diagnostic tool. Predictions should not be used for clinical decision-making without further validation.
