import pandas as pd
import numpy as np
import joblib  # Better than pickle for sklearn models
import json
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
from optuna_5 import optimize_models_with_optuna
import seaborn as sns
from scipy import stats
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Create a serializable prediction class
class ECGPredictor:
    """
    Serializable ECG prediction class that can be saved and loaded
    """
    def __init__(self, model, scaler, selected_features, model_name):
        self.model = model
        self.scaler = scaler
        self.selected_features = selected_features
        self.model_name = model_name
    
    def predict(self, features_dict):

        # Convert to DataFrame
        X_new = pd.DataFrame([features_dict])
        
        # Ensure all required features are present
        missing_features = set(self.selected_features) - set(X_new.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and order features
        X_new = X_new[self.selected_features]
        
        # Scale if needed
        if self.scaler is not None:
            X_new_processed = self.scaler.transform(X_new)
        else:
            X_new_processed = X_new
        
        # Predict
        prediction = self.model.predict(X_new_processed)[0]
        probability = self.model.predict_proba(X_new_processed)[0]
        
        # Calculate confidence and risk level
        confidence = max(probability)
        risk_level = "HIGH" if probability[1] > 0.8 else "MEDIUM" if probability[1] > 0.5 else "LOW"
        
        return {
            'prediction': 'Abnormal' if prediction == 1 else 'Normal',
            'probability_normal': float(probability[0]),
            'probability_abnormal': float(probability[1]),
            'confidence': float(confidence),
            'risk_level': risk_level,
            'model_used': self.model_name,
            'timestamp': datetime.now().isoformat()
        }
    
    def __call__(self, features_dict):
        """Make the class callable"""
        return self.predict(features_dict)

def enhanced_model_optimization(model, X_train, y_train, model_name):
    try:
        if 'LightGBM' in model_name or 'LGBM' in model_name:
            # Create a new model with enhanced parameters instead of modifying existing one
            from lightgbm import LGBMClassifier
            
            # Get current parameters if possible
            current_params = {}
            try:
                current_params = model.get_params()
            except:
                pass
            
            # Enhanced regularization parameters
            enhanced_params = {
                'reg_alpha': 0.1,           # Moderate L1 regularization
                'reg_lambda': 0.1,          # Moderate L2 regularization  
                'max_depth': 8,             # Slightly reduced depth
                'min_child_samples': 20,    # Increased minimum samples
                'feature_fraction': 0.9,    # Light feature sampling
                'bagging_fraction': 0.9,    # Light row sampling
                'bagging_freq': 1,
                'n_estimators': current_params.get('n_estimators', 100),
                'learning_rate': current_params.get('learning_rate', 0.1),
                'random_state': current_params.get('random_state', 42),
                'verbose': -1
            }
            
            # Merge with current parameters, giving priority to enhanced ones
            final_params = {**current_params, **enhanced_params}
            
            # Create new model with enhanced parameters
            enhanced_model = LGBMClassifier(**final_params)
            
            print(f"✅ Applied enhanced regularization to {model_name}")
            return enhanced_model
            
        else:
            # For non-LightGBM models, return as-is
            print(f"ℹ️  No specific enhancements for {model_name}, using original model")
            return model
            
    except Exception as e:
        print(f"⚠️  Could not apply enhancements: {str(e)}")
        print("   Using original model...")
        return model

def final_model_validation(df, best_model, best_model_name, selected_features, scaler=None):
    print("🎯 Enhanced ECG Model Validation & Deployment Preparation")
    print("=" * 70)
    print(f"Best Model: {best_model_name}")
    print(f"Selected Features: {len(selected_features)} features")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"ecg_model_deployment_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    X = df[selected_features]
    y = df['label']
    
    # Train-test split (same as optimization)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale if needed
    if scaler is not None:
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
    else:
        X_train_processed = X_train
        X_test_processed = X_test
    
    # Apply enhanced optimization to reduce overfitting
    print("\n🔧 Applying Enhanced Regularization...")
    print(f"Original model type: {type(best_model)}")
    enhanced_model = enhanced_model_optimization(best_model, X_train_processed, y_train, best_model_name)
    print(f"Enhanced model type: {type(enhanced_model)}")
    
    # Debug: Check if model is properly configured
    try:
        # Quick test fit on small subset
        test_indices = np.random.choice(len(X_train_processed), min(100, len(X_train_processed)), replace=False)
        enhanced_model.fit(X_train_processed[test_indices], y_train.iloc[test_indices])
        print("✅ Model configuration test passed")
    except Exception as e:
        print(f"❌ Model configuration test failed: {str(e)}")
        print("🔄 Reverting to original model...")
        enhanced_model = best_model
    
    # 1. COMPREHENSIVE PERFORMANCE EVALUATION
    print("\n📊 1. Enhanced Performance Metrics")
    print("=" * 50)
    
    # Cross-validation with multiple metrics
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall', 
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision'
    }
    
    # Add error handling for cross-validation
    try:
        cv_results = cross_validate(
            enhanced_model, X_train_processed, y_train,
            cv=cv, scoring=scoring_metrics, return_train_score=True,
            error_score='raise'  # This will give us more detailed error info
        )
    except Exception as e:
        print(f"❌ Cross-validation failed: {str(e)}")
        print("🔄 Falling back to original model...")
        
        # Fall back to original model
        enhanced_model = best_model
        cv_results = cross_validate(
            enhanced_model, X_train_processed, y_train,
            cv=cv, scoring=scoring_metrics, return_train_score=True
        )
    
    # Train final model
    final_model = enhanced_model
    final_model.fit(X_train_processed, y_train)
    
    # Predictions
    y_pred = final_model.predict(X_test_processed)
    y_proba = final_model.predict_proba(X_test_processed)[:, 1]
    
    # Calculate comprehensive metrics
    test_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'PR-AUC': average_precision_score(y_test, y_proba),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    
    # Display results
    print("10-Fold Cross-Validation Results (Enhanced Model):")
    for metric in scoring_metrics.keys():
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        print(f"  {metric.upper():15}: Train {train_scores.mean():.4f} (±{train_scores.std():.4f}) | "
              f"Test {test_scores.mean():.4f} (±{test_scores.std():.4f})")
    
    print(f"\nFinal Test Set Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric:15}: {value:.4f}")
    
    # 2. STATISTICAL CONFIDENCE INTERVALS
    print(f"\n📈 2. Statistical Confidence Analysis")
    print("=" * 50)
    
    # Bootstrap confidence intervals for ROC-AUC
    n_bootstrap = 1000
    roc_aucs = []
    
    np.random.seed(42)
    for i in range(n_bootstrap):
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        y_test_boot = y_test.iloc[indices]
        y_proba_boot = y_proba[indices]
        
        roc_auc_boot = roc_auc_score(y_test_boot, y_proba_boot)
        roc_aucs.append(roc_auc_boot)
    
    roc_aucs = np.array(roc_aucs)
    roc_ci_lower = np.percentile(roc_aucs, 2.5)
    roc_ci_upper = np.percentile(roc_aucs, 97.5)
    
    print(f"ROC-AUC: {test_metrics['ROC-AUC']:.4f}")
    print(f"95% Confidence Interval: [{roc_ci_lower:.4f}, {roc_ci_upper:.4f}]")
    print(f"Standard Error: {roc_aucs.std():.4f}")
    
    # 3. ENHANCED FEATURE IMPORTANCE ANALYSIS
    print(f"\n🔍 3. Feature Importance Analysis")
    print("=" * 50)
    
    feature_importance = None
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:20} {row['importance']:.4f}")
    
    # 4. ENHANCED ROBUSTNESS ANALYSIS
    print(f"\n🛡️ 4. Model Robustness Analysis")
    print("=" * 50)
    
    # Learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        final_model, X_train_processed, y_train,
        cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='roc_auc', n_jobs=-1
    )
    
    overfitting_gap = train_scores[-1].mean() - val_scores[-1].mean()
    
    print(f"Learning Curve Analysis (Enhanced Model):")
    print(f"  Training set performance: {train_scores[-1].mean():.4f} (±{train_scores[-1].std():.4f})")
    print(f"  Validation performance: {val_scores[-1].mean():.4f} (±{val_scores[-1].std():.4f})")
    print(f"  Overfitting gap: {overfitting_gap:.4f}")
    
    if overfitting_gap < 0.02:
        print("  ✅ Excellent generalization (minimal overfitting)")
    elif overfitting_gap < 0.05:
        print("  ⚠️ Good generalization (acceptable overfitting)")
    elif overfitting_gap < 0.08:
        print("  ⚠️ Moderate overfitting - consider more regularization")
    else:
        print("  ❌ High overfitting - requires additional regularization")
    
    # 5. CLINICAL RELEVANCE ANALYSIS
    print(f"\n🏥 5. Clinical Relevance Analysis")
    print("=" * 50)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Clinical metrics
    sensitivity = tp / (tp + fn)  # True Positive Rate (Recall)
    specificity = tn / (tn + fp)  # True Negative Rate
    ppv = tp / (tp + fp)          # Positive Predictive Value (Precision)
    npv = tn / (tn + fn)          # Negative Predictive Value
    
    clinical_metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'false_negatives': fn,
        'false_positives': fp
    }
    
    print(f"Clinical Performance Metrics:")
    print(f"  Sensitivity (Recall): {sensitivity:.4f} - Ability to detect abnormal ECGs")
    print(f"  Specificity:          {specificity:.4f} - Ability to correctly identify normal ECGs")
    print(f"  PPV (Precision):      {ppv:.4f} - Probability abnormal prediction is correct")
    print(f"  NPV:                  {npv:.4f} - Probability normal prediction is correct")
    
    print(f"\nClinical Risk Assessment:")
    print(f"  False Negatives: {fn} patients ({fn/len(y_test)*100:.1f}%) - Missed abnormal cases")
    print(f"  False Positives: {fp} patients ({fp/len(y_test)*100:.1f}%) - False alarms")
    
    # 6. SAVE MODEL FOR DEPLOYMENT (FIXED)
    print(f"\n💾 6. Model Deployment Preparation")
    print("=" * 50)
    
    # Save model components separately (FIXED APPROACH)
    model_filename = os.path.join(output_dir, "ecg_model.joblib")
    scaler_filename = os.path.join(output_dir, "ecg_scaler.joblib")
    features_filename = os.path.join(output_dir, "ecg_features.json")
    
    # Save using joblib (better for sklearn models)
    joblib.dump(final_model, model_filename)
    if scaler is not None:
        joblib.dump(scaler, scaler_filename)
    
    # Save features and metadata as JSON
    model_metadata = {
        'selected_features': selected_features,
        'model_name': best_model_name,
        'performance_metrics': {k: float(v) for k, v in test_metrics.items()},
        'clinical_metrics': {k: float(v) for k, v in clinical_metrics.items()},
        'training_info': {
            'training_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'feature_count': int(len(selected_features)),
            'class_distribution': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            'overfitting_gap': float(overfitting_gap),
            'roc_ci_lower': float(roc_ci_lower),
            'roc_ci_upper': float(roc_ci_upper)
        },
        'deployment_timestamp': timestamp
    }
    
    with open(features_filename, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Create and save prediction class (FIXED)
    predictor = ECGPredictor(final_model, scaler, selected_features, best_model_name)
    prediction_func_filename = os.path.join(output_dir, "ecg_predictor.joblib")
    
    try:
        joblib.dump(predictor, prediction_func_filename)
        print(f"✅ Prediction class saved successfully")
    except Exception as e:
        print(f"⚠️  Could not save prediction class: {str(e)}")
        print("   Saving model components separately...")
        
        # Alternative: save components separately
        joblib.dump(final_model, os.path.join(output_dir, "model_only.joblib"))
        if scaler:
            joblib.dump(scaler, scaler_filename)
        
        # Create a simple prediction script instead
        simple_predict_script = f
        with open(os.path.join(output_dir, "predict_function.py"), 'w') as f:
            f.write(simple_predict_script)
    
    # Create a deployment script
    deployment_script = f
    deployment_script_filename = os.path.join(output_dir, "deploy_model.py")
    with open(deployment_script_filename, 'w') as f:
        f.write(deployment_script)
    
    # Save feature importance if available
    if feature_importance is not None:
        importance_filename = os.path.join(output_dir, "feature_importance.csv")
        feature_importance.to_csv(importance_filename, index=False)
    
    print(f"✅ Enhanced Model Package Saved in: {output_dir}/")
    print(f"  • ecg_model.joblib - Main model")
    print(f"  • ecg_scaler.joblib - Feature scaler")
    print(f"  • ecg_features.json - Features and metadata")
    print(f"  • ecg_predict_function.joblib - Prediction function")
    print(f"  • deploy_model.py - Deployment script")
    if feature_importance is not None:
        print(f"  • feature_importance.csv - Feature importance rankings")
    
    # 7. CREATE COMPREHENSIVE VISUALIZATIONS
    print(f"\n📊 7. Generating Visualizations...")
    plt.figure(figsize=(20, 15))
    
    # 1. Confusion Matrix
    plt.subplot(3, 4, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 2. ROC Curve
    plt.subplot(3, 4, 2)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {test_metrics["ROC-AUC"]:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    plt.subplot(3, 4, 3)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, label=f'PR (AUC = {test_metrics["PR-AUC"]:.3f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Learning Curve
    plt.subplot(3, 4, 4)
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training Score', linewidth=2)
    plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation Score', linewidth=2)
    plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('ROC-AUC Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Feature Importance (if available)
    if feature_importance is not None:
        plt.subplot(3, 4, 5)
        top_features = feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importance')
        plt.gca().invert_yaxis()
    
    # 6. ROC-AUC Distribution (Bootstrap)
    plt.subplot(3, 4, 6)
    plt.hist(roc_aucs, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(test_metrics['ROC-AUC'], color='red', linestyle='--', label=f'Test ROC-AUC: {test_metrics["ROC-AUC"]:.3f}')
    plt.axvline(roc_ci_lower, color='orange', linestyle='--', alpha=0.7, label=f'95% CI: [{roc_ci_lower:.3f}, {roc_ci_upper:.3f}]')
    plt.axvline(roc_ci_upper, color='orange', linestyle='--', alpha=0.7)
    plt.xlabel('ROC-AUC')
    plt.ylabel('Frequency')
    plt.title('Bootstrap ROC-AUC Distribution')
    plt.legend()
    
    # 7. Cross-validation scores
    plt.subplot(3, 4, 7)
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_means = [cv_results[f'test_{metric}'].mean() for metric in metrics_to_plot]
    cv_stds = [cv_results[f'test_{metric}'].std() for metric in metrics_to_plot]
    
    x_pos = np.arange(len(metrics_to_plot))
    plt.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('10-Fold CV Performance')
    plt.xticks(x_pos, [m.upper() for m in metrics_to_plot])
    plt.grid(True, alpha=0.3)
    
    # 8. Probability Distribution
    plt.subplot(3, 4, 8)
    normal_probs = y_proba[y_test == 0]
    abnormal_probs = y_proba[y_test == 1]
    
    plt.hist(normal_probs, bins=20, alpha=0.5, label='Normal', color='blue')
    plt.hist(abnormal_probs, bins=20, alpha=0.5, label='Abnormal', color='red')
    plt.xlabel('Predicted Probability of Abnormal')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution by True Class')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(output_dir, "model_analysis_plots.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # ENHANCED SUMMARY REPORT
    print(f"\n📋 ENHANCED MODEL SUMMARY REPORT")
    print("=" * 70)
    print(f"Model Type: {best_model_name}")
    print(f"Features Used: {len(selected_features)} selected features")
    print(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")
    print(f"Deployment Package: {output_dir}/")
    print(f"")
    print(f"PERFORMANCE METRICS (Enhanced Model):")
    print(f"  • Test Accuracy: {test_metrics['Accuracy']:.1%}")
    print(f"  • Test ROC-AUC:  {test_metrics['ROC-AUC']:.4f} (95% CI: [{roc_ci_lower:.3f}, {roc_ci_upper:.3f}])")
    print(f"  • Sensitivity:   {sensitivity:.1%} (detects {sensitivity:.1%} of abnormal cases)")
    print(f"  • Specificity:   {specificity:.1%} (correctly identifies {specificity:.1%} of normal cases)")
    print(f"  • Overfitting:   {overfitting_gap:.4f} (lower is better)")
    print(f"")
    
    # Clinical interpretation
    print(f"CLINICAL INTERPRETATION:")
    if test_metrics['ROC-AUC'] >= 0.90:
        print(f"  🟢 EXCELLENT: Model shows excellent discriminative ability")
    elif test_metrics['ROC-AUC'] >= 0.80:
        print(f"  🟡 GOOD: Model shows good discriminative ability")
    else:
        print(f"  🟠 FAIR: Model shows fair discriminative ability")
    
    if overfitting_gap < 0.05:
        print(f"  🟢 ROBUST: Model shows good generalization")
    else:
        print(f"  🟡 CAUTION: Monitor model performance on new data")
    
    return {
        'final_model': final_model,
        'test_metrics': test_metrics,
        'clinical_metrics': clinical_metrics,
        'feature_importance': feature_importance,
        'model_metadata': model_metadata,
        'predict_function': predict_ecg,
        'output_directory': output_dir,
        'overfitting_gap': overfitting_gap
    }

# Main execution
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("ecg_features_cleaned.csv")
    
    # Run optimization to get the best model and params
    print("Running model optimization...")
    optuna_results = optimize_models_with_optuna(df, n_trials=100)
    
    # Extract required variables
    best_model = optuna_results['best_model']
    best_model_name = optuna_results['best_model_name']
    selected_features = optuna_results['selected_features']
    scaler = optuna_results['scaler']
    
    # Run enhanced final validation
    print("\nRunning enhanced model validation...")
    final_results = final_model_validation(df, best_model, best_model_name, selected_features, scaler)
    
    print("\n🎯 Enhanced ECG Classifier is ready for deployment!")
    print(f"📁 All files saved in: {final_results['output_directory']}")
    
    # Example of how to use the saved model
    print("\n📝 Usage Example:")
    print("```python")
    print("import joblib")
    print("predict_function = joblib.load('path/to/ecg_predict_function.joblib')")
    print("result = predict_function({'age': 65, 'III_skewness': 1.2, ...})")
    print("print(result)")
    print("```")