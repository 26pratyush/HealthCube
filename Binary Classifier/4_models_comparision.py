import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings('ignore')

def evaluate_baseline_models(df):
    # Selected top 15 features based on your analysis
    selected_features = [
        'age', 'aVL_std', 'aVL_range', 'aVF_skewness', 'signal_skewness', 
        'II_skewness', 'III_std', 'V1_skewness', 'V4_skewness', 'III_range', 
        'V1_std', 'III_skewness', 'signal_min', 'I_range', 'V3_range'
    ]
    
    print("🚀 ECG Baseline Model Comparison")
    print("=" * 50)
    print(f"Using {len(selected_features)} selected features")
    print(f"Features: {selected_features}")
    
    # Prepare data
    X = df[selected_features]
    y = df['label']
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    print(f"Class distribution - Train: {np.bincount(y_train)}")
    print(f"Class distribution - Test: {np.bincount(y_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            random_state=42, eval_metric='logloss', verbosity=0
        ),
        'LightGBM': LGBMClassifier(
            random_state=42, verbosity=-1, force_row_wise=True
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', probability=True, random_state=42
        )
    }
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    results = {}
    predictions = {}
    
    print("\n📊 Model Evaluation Results")
    print("=" * 80)
    
    for name, model in models.items():
        print(f"\n🔍 Evaluating {name}...")
        
        # Prepare data based on model requirements
        if name in ['Logistic Regression', 'SVM (RBF)']:
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        # Cross-validation
        cv_results = cross_validate(
            model, X_train_model, y_train, 
            cv=cv, scoring=scoring_metrics, return_train_score=False
        )
        
        # Fit model and make predictions
        model.fit(X_train_model, y_train)
        y_pred = model.predict(X_test_model)
        y_proba = model.predict_proba(X_test_model)[:, 1]
        
        # Calculate test metrics
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Store results
        results[name] = {
            'cv_mean': {metric: np.mean(cv_results[f'test_{metric}']) for metric in scoring_metrics},
            'cv_std': {metric: np.std(cv_results[f'test_{metric}']) for metric in scoring_metrics},
            'test': test_metrics,
            'model': model
        }
        
        predictions[name] = {
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        # Print results
        print(f"Cross-validation (5-fold):")
        for metric in scoring_metrics:
            mean_score = results[name]['cv_mean'][metric]
            std_score = results[name]['cv_std'][metric]
            print(f"  {metric:12}: {mean_score:.4f} (±{std_score:.4f})")
        
        print(f"Test set performance:")
        for metric, score in test_metrics.items():
            print(f"  {metric:12}: {score:.4f}")
    
    # Create comparison table
    print("\n📋 Model Comparison Summary")
    print("=" * 80)
    
    comparison_df = pd.DataFrame({
        model_name: {
            'CV Accuracy': f"{results[model_name]['cv_mean']['accuracy']:.4f} (±{results[model_name]['cv_std']['accuracy']:.4f})",
            'CV F1-Score': f"{results[model_name]['cv_mean']['f1']:.4f} (±{results[model_name]['cv_std']['f1']:.4f})",
            'CV ROC-AUC': f"{results[model_name]['cv_mean']['roc_auc']:.4f} (±{results[model_name]['cv_std']['roc_auc']:.4f})",
            'Test Accuracy': f"{results[model_name]['test']['accuracy']:.4f}",
            'Test F1-Score': f"{results[model_name]['test']['f1']:.4f}",
            'Test ROC-AUC': f"{results[model_name]['test']['roc_auc']:.4f}"
        }
        for model_name in models.keys()
    }).T
    
    print(comparison_df.to_string())
    
    # Find best model based on CV ROC-AUC
    best_model_name = max(results.keys(), 
                         key=lambda x: results[x]['cv_mean']['roc_auc'])
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"CV ROC-AUC: {results[best_model_name]['cv_mean']['roc_auc']:.4f} (±{results[best_model_name]['cv_std']['roc_auc']:.4f})")
    print(f"Test ROC-AUC: {results[best_model_name]['test']['roc_auc']:.4f}")
    
    # Create visualizations
    plt.figure(figsize=(20, 15))
    
    # 1. ROC Curves
    plt.subplot(2, 3, 1)
    for name in models.keys():
        fpr, tpr, _ = roc_curve(y_test, predictions[name]['y_proba'])
        auc = results[name]['test']['roc_auc']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curves
    plt.subplot(2, 3, 2)
    for name in models.keys():
        precision, recall, _ = precision_recall_curve(y_test, predictions[name]['y_proba'])
        avg_precision = np.mean(precision)
        plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Cross-validation scores comparison
    plt.subplot(2, 3, 3)
    cv_scores = pd.DataFrame({
        name: [results[name]['cv_mean'][metric] for metric in scoring_metrics]
        for name in models.keys()
    }, index=scoring_metrics)
    
    cv_scores.plot(kind='bar', ax=plt.gca())
    plt.title('Cross-Validation Scores Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 4. Confusion matrix for best model
    plt.subplot(2, 3, 4)
    cm = confusion_matrix(y_test, predictions[best_model_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 5. Feature importance for best model (if available)
    plt.subplot(2, 3, 5)
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': selected_features,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        plt.barh(range(len(feature_imp)), feature_imp['importance'])
        plt.yticks(range(len(feature_imp)), feature_imp['feature'])
        plt.title(f'Feature Importance - {best_model_name}')
        plt.xlabel('Importance')
    elif hasattr(best_model, 'coef_'):
        coef = np.abs(best_model.coef_[0])
        feature_imp = pd.DataFrame({
            'feature': selected_features,
            'importance': coef
        }).sort_values('importance', ascending=True)
        
        plt.barh(range(len(feature_imp)), feature_imp['importance'])
        plt.yticks(range(len(feature_imp)), feature_imp['feature'])
        plt.title(f'Feature Coefficients - {best_model_name}')
        plt.xlabel('|Coefficient|')
    else:
        plt.text(0.5, 0.5, f'Feature importance not available\nfor {best_model_name}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importance')
    
    # 6. Model performance comparison
    plt.subplot(2, 3, 6)
    test_scores = pd.DataFrame({
        'Accuracy': [results[name]['test']['accuracy'] for name in models.keys()],
        'F1-Score': [results[name]['test']['f1'] for name in models.keys()],
        'ROC-AUC': [results[name]['test']['roc_auc'] for name in models.keys()]
    }, index=models.keys())
    
    test_scores.plot(kind='bar', ax=plt.gca())
    plt.title('Test Set Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Classification report for best model
    print(f"\n📝 Detailed Classification Report - {best_model_name}")
    print("=" * 50)
    print(classification_report(y_test, predictions[best_model_name]['y_pred'], 
                              target_names=['Normal', 'Abnormal']))
    
    return {
        'results': results,
        'predictions': predictions,
        'best_model_name': best_model_name,
        'best_model': results[best_model_name]['model'],
        'selected_features': selected_features,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test
    }

df = pd.read_csv("ecg_features_cleaned.csv")

# Run the baseline evaluation
baseline_results = evaluate_baseline_models(df)

print("\n🎯 Next Steps:")
print("1. Run this baseline comparison")
print("2. Identify the best performing model") 
print("3. Proceed to hyperparameter optimization with Optuna")
print("4. Final model evaluation and deployment preparation")