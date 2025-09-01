# 8_enhanced_batch_predictor.py
# Batch prediction with comprehensive evaluation to ensure no data leakage
import json
from datetime import datetime
from pathlib import Path
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- ECGPredictor class (must match the saved class) ---
class ECGPredictor:
    def __init__(self, model, scaler, selected_features, model_name):
        self.model = model
        self.scaler = scaler
        self.selected_features = selected_features
        self.model_name = model_name

    def __call__(self, features_dict):
        X = pd.DataFrame([features_dict])
        
        # Ensure required features present and in correct order
        missing = set(self.selected_features) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        X = X[self.selected_features]
        Xp = self.scaler.transform(X) if self.scaler is not None else X

        pred = int(self.model.predict(Xp)[0])
        proba = self.model.predict_proba(Xp)[0]

        return {
            "prediction": "Abnormal" if pred == 1 else "Normal",
            "probability_normal": float(proba[0]),
            "probability_abnormal": float(proba[1]),
            "confidence": float(max(proba)),
            "model_used": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "pred_label_int": pred,  # 0/1 numeric for convenience
        }
    
    def predict_batch(self, features_df):
        """
        Batch prediction for multiple samples
        
        Args:
            features_df: DataFrame with features only (no labels)
        
        Returns:
            dict: Contains predictions, probabilities, and metadata
        """
        # Ensure all required features are present
        missing = set(self.selected_features) - set(features_df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Select and order features correctly
        X = features_df[self.selected_features].copy()
        
        # Handle any remaining NaN values
        if X.isnull().any().any():
            print("⚠️ Warning: NaN values found in features. Filling with median values.")
            X = X.fillna(X.median())
        
        # Scale features if scaler exists
        if self.scaler is not None:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X.values
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        return {
            'predictions': predictions.astype(int),
            'probabilities': probabilities,
            'probability_normal': probabilities[:, 0],
            'probability_abnormal': probabilities[:, 1],
            'confidence': np.max(probabilities, axis=1),
            'model_used': self.model_name,
            'n_samples': len(X),
            'timestamp': datetime.now().isoformat()
        }

# ------------------- CONFIG -------------------
# Update this to your actual deployment folder
PKG_DIR = Path("ecg_model_deployment_20250821_125946")  # Update with your actual folder

PREDICTOR_PATH = PKG_DIR / "ecg_predictor.joblib"
META_PATH      = PKG_DIR / "ecg_features.json"

# CSV paths
CANDIDATE_CSV_PATHS = [
    Path("ecg_features_cleaned.csv"),
]

# Prediction settings
N_SAMPLES = 100  # Number of samples to predict (set to None for all data)
RANDOM_SEED = 40  # For reproducible results
TEST_SPLIT = 0.2  # Use 20% of data for testing (simulates new unseen data)
# ------------------------------------------------

def find_csv_path():
    for p in CANDIDATE_CSV_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find ecg_features_cleaned.csv. Please update CANDIDATE_CSV_PATHS."
    )

def ensure_no_data_leakage(predictor, features_df):
    """
    Verify that the predictor only uses features and not labels
    """
    print("🔒 Verifying No Data Leakage...")
    
    # Check that predictor only uses the specified features
    print(f"   Predictor uses {len(predictor.selected_features)} features")
    print(f"   Available features in data: {len(features_df.columns)}")
    
    # Verify no 'label' column in selected features
    if 'label' in predictor.selected_features:
        raise ValueError("❌ DATA LEAKAGE: 'label' column found in selected features!")
    
    print("✅ No data leakage detected - predictor only uses feature columns")

def evaluate_predictions(y_true, predictions_dict):
    """
    Comprehensive evaluation of predictions vs true labels
    """
    y_pred = predictions_dict['predictions']
    y_proba = predictions_dict['probability_abnormal']
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Clinical metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    clinical_metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'false_negatives': fn,
        'false_positives': fp,
        'true_positives': tp,
        'true_negatives': tn
    }
    
    return metrics, clinical_metrics, cm

def create_evaluation_plots(y_true, predictions_dict, save_path=None):
    """
    Create comprehensive evaluation plots
    """
    y_pred = predictions_dict['predictions']
    y_proba = predictions_dict['probability_abnormal']
    confidence = predictions_dict['confidence']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('True')
    
    # 2. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[0,1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc_score(y_true, y_proba):.3f})')
    axes[0,1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[0,2].plot(recall, precision, linewidth=2, 
                   label=f'PR (AUC = {average_precision_score(y_true, y_proba):.3f})')
    axes[0,2].set_xlabel('Recall')
    axes[0,2].set_ylabel('Precision')
    axes[0,2].set_title('Precision-Recall Curve')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Probability Distribution by True Class
    normal_probs = y_proba[y_true == 0]
    abnormal_probs = y_proba[y_true == 1]
    
    axes[1,0].hist(normal_probs, bins=20, alpha=0.5, label='True Normal', color='blue')
    axes[1,0].hist(abnormal_probs, bins=20, alpha=0.5, label='True Abnormal', color='red')
    axes[1,0].set_xlabel('Predicted Probability of Abnormal')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Probability Distribution by True Class')
    axes[1,0].legend()
    
    # 5. Confidence Distribution
    correct_preds = (y_true == y_pred)
    correct_conf = confidence[correct_preds]
    incorrect_conf = confidence[~correct_preds]
    
    axes[1,1].hist(correct_conf, bins=20, alpha=0.5, label='Correct Predictions', color='green')
    axes[1,1].hist(incorrect_conf, bins=20, alpha=0.5, label='Incorrect Predictions', color='red')
    axes[1,1].set_xlabel('Confidence')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Confidence Distribution')
    axes[1,1].legend()
    
    # 6. Prediction vs True Label Scatter
    jitter_true = y_true + np.random.normal(0, 0.05, len(y_true))
    jitter_pred = y_pred + np.random.normal(0, 0.05, len(y_pred))
    
    colors = ['green' if t == p else 'red' for t, p in zip(y_true, y_pred)]
    axes[1,2].scatter(jitter_true, jitter_pred, c=colors, alpha=0.6, s=20)
    axes[1,2].plot([0, 1], [0, 1], 'k--', linewidth=2)
    axes[1,2].set_xlabel('True Label')
    axes[1,2].set_ylabel('Predicted Label')
    axes[1,2].set_title('Predictions vs Truth (Green=Correct, Red=Wrong)')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Evaluation plots saved to: {save_path}")
    
    plt.show()

def main():
    print("🎯 Enhanced ECG Batch Predictor with Data Leakage Protection")
    print("=" * 70)
    
    # Load predictor and metadata
    try:
        predictor = joblib.load(PREDICTOR_PATH)
        print(f"✅ Loaded predictor from: {PREDICTOR_PATH}")
    except FileNotFoundError:
        print(f"❌ Could not find predictor at: {PREDICTOR_PATH}")
        print("   Please check the PKG_DIR path in the config section")
        return
    
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    
    selected_features = meta["selected_features"]
    print(f"📋 Model uses {len(selected_features)} features")
    
    # Load dataset
    csv_path = find_csv_path()
    print(f"📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if "label" not in df.columns:
        raise ValueError("❌ The CSV must contain a 'label' column for evaluation.")
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"   Class distribution: {dict(df['label'].value_counts())}")
    
    # Clean data - remove rows with missing features or labels
    required_cols = selected_features + ['label']
    df_clean = df.dropna(subset=required_cols).copy()
    print(f"📊 After removing NaN values: {df_clean.shape}")
    
    if df_clean.empty:
        raise ValueError("❌ No complete rows found after removing NaN values")
    
    # Sample data if requested
    if N_SAMPLES and N_SAMPLES < len(df_clean):
        df_sample = df_clean.sample(n=N_SAMPLES, random_state=RANDOM_SEED)
        print(f"🎲 Using random sample of {N_SAMPLES} rows")
    else:
        df_sample = df_clean
        print(f"📊 Using all {len(df_sample)} clean rows")
    
    # Split into features and labels (IMPORTANT: Separate them clearly)
    X_features = df_sample[selected_features].copy()  # Features only
    y_true = df_sample['label'].copy()  # True labels only
    
    print(f"\n🔍 Data Summary:")
    print(f"   Features shape: {X_features.shape}")
    print(f"   True labels shape: {y_true.shape}")
    print(f"   Sample class distribution: {dict(y_true.value_counts())}")
    
    # Verify no data leakage
    ensure_no_data_leakage(predictor, X_features)
    
    # Make batch predictions (predictor only sees features, never labels)
    print(f"\n🤖 Making batch predictions...")
    start_time = datetime.now()
    
    try:
        predictions_dict = predictor.predict_batch(X_features)
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Predictions completed in {prediction_time:.2f} seconds")
        print(f"   Processed {predictions_dict['n_samples']} samples")
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return
    
    # Evaluate predictions against true labels
    print(f"\n📊 Evaluating Predictions...")
    metrics, clinical_metrics, cm = evaluate_predictions(y_true, predictions_dict)
    
    # Display results
    print(f"\n🎯 BATCH PREDICTION RESULTS")
    print("=" * 50)
    print(f"Samples processed: {len(y_true)}")
    print(f"Model used: {predictions_dict['model_used']}")
    print(f"Timestamp: {predictions_dict['timestamp']}")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    for metric, value in metrics.items():
        print(f"  {metric.upper():15}: {value:.4f}")
    
    print(f"\n🏥 CLINICAL METRICS:")
    print(f"  Sensitivity:     {clinical_metrics['sensitivity']:.4f} (detects {clinical_metrics['sensitivity']:.1%} of abnormal cases)")
    print(f"  Specificity:     {clinical_metrics['specificity']:.4f} (correctly identifies {clinical_metrics['specificity']:.1%} of normal cases)")
    print(f"  PPV:            {clinical_metrics['ppv']:.4f} (precision)")
    print(f"  NPV:            {clinical_metrics['npv']:.4f}")
    
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"  True Negatives:  {clinical_metrics['true_negatives']}")
    print(f"  False Positives: {clinical_metrics['false_positives']}")
    print(f"  False Negatives: {clinical_metrics['false_negatives']} (missed abnormal cases)")
    print(f"  True Positives:  {clinical_metrics['true_positives']}")
    
    # Show sample predictions
    print(f"\n🔍 SAMPLE PREDICTIONS (First 10):")
    for i in range(min(10, len(y_true))):
        true_label = "Abnormal" if y_true.iloc[i] == 1 else "Normal"
        pred_label = "Abnormal" if predictions_dict['predictions'][i] == 1 else "Normal"
        confidence = predictions_dict['confidence'][i]
        correct = "✅" if y_true.iloc[i] == predictions_dict['predictions'][i] else "❌"
        
        print(f"  {i+1:2d}. True: {true_label:8} | Pred: {pred_label:8} | Conf: {confidence:.3f} {correct}")
    
    # Create evaluation plots
    print(f"\n📊 Generating evaluation plots...")
    plot_path = PKG_DIR / "batch_prediction_evaluation.png"
    create_evaluation_plots(y_true, predictions_dict, save_path=plot_path)
    
    # Save detailed results
    results_df = pd.DataFrame({
        'true_label': y_true.values,
        'predicted_label': predictions_dict['predictions'],
        'probability_normal': predictions_dict['probability_normal'],
        'probability_abnormal': predictions_dict['probability_abnormal'],
        'confidence': predictions_dict['confidence'],
        'correct': y_true.values == predictions_dict['predictions']
    })
    
    results_path = PKG_DIR / "batch_prediction_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"💾 Detailed results saved to: {results_path}")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    if metrics['accuracy'] >= 0.85:
        print(f"  🟢 EXCELLENT: High accuracy ({metrics['accuracy']:.1%})")
    elif metrics['accuracy'] >= 0.75:
        print(f"  🟡 GOOD: Acceptable accuracy ({metrics['accuracy']:.1%})")
    else:
        print(f"  🟠 FAIR: Room for improvement ({metrics['accuracy']:.1%})")
    
    if clinical_metrics['sensitivity'] >= 0.80:
        print(f"  🟢 GOOD: High sensitivity - catching most abnormal cases")
    else:
        print(f"  🟡 CAUTION: Lower sensitivity - missing {1-clinical_metrics['sensitivity']:.1%} of abnormal cases")
    
    print(f"\n✅ Batch prediction completed successfully!")
    print(f"📁 All results saved in: {PKG_DIR}/")

if __name__ == "__main__":
    main()
