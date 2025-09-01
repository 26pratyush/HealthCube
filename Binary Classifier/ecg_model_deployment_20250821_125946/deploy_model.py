
# ECG Model Deployment Script
# Generated on 2025-08-21 12:59:54.355193

import joblib
import json
import pandas as pd

# Load model components
model = joblib.load('ecg_model_deployment_20250821_125946\ecg_model.joblib')
scaler = joblib.load('ecg_model_deployment_20250821_125946\ecg_scaler.joblib') if 'ecg_model_deployment_20250821_125946\ecg_scaler.joblib' else None
predict_function = joblib.load('ecg_model_deployment_20250821_125946\ecg_predictor.joblib')

with open('ecg_model_deployment_20250821_125946\ecg_features.json', 'r') as f:
    metadata = json.load(f)

print("ECG Classifier loaded successfully!")
print(f"Model: {metadata['model_name']}")
print(f"Features: {metadata['training_info']['feature_count']}")
print(f"Performance: ROC-AUC = {metadata['performance_metrics']['ROC-AUC']:.3f}")

# Example usage:
# result = predict_function({'feature1': value1, 'feature2': value2, ...})
# print(result)
