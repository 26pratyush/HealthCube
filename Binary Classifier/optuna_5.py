import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("ecg_features_cleaned.csv")

def optimize_models_with_optuna(df, n_trials=100):
    # Selected features from previous analysis
    selected_features = [
        'age', 'aVL_std', 'aVL_range', 'aVF_skewness', 'signal_skewness', 
        'II_skewness', 'III_std', 'V1_skewness', 'V4_skewness', 'III_range', 
        'V1_std', 'III_skewness', 'signal_min', 'I_range', 'V3_range'
    ]
    
    print("🚀 ECG Model Optimization with Optuna")
    print("=" * 50)
    print(f"Features: {len(selected_features)} selected features")
    print(f"Optimization trials: {n_trials} per model")
    
    # Prepare data
    X = df[selected_features]
    y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    optimization_results = {}
    
    # 1. Optimize SVM (RBF) - Best baseline model
    print("\n🔍 Optimizing SVM (RBF)...")
    
    def svm_objective(trial):
        # Handle gamma parameter properly
        gamma_type = trial.suggest_categorical('gamma_type', ['preset', 'manual'])
        
        if gamma_type == 'preset':
            gamma_value = trial.suggest_categorical('gamma_preset', ['scale', 'auto'])
        else:
            gamma_value = trial.suggest_float('gamma_manual', 1e-6, 1e2, log=True)
        
        params = {
            'C': trial.suggest_float('C', 0.001, 1000, log=True),
            'gamma': gamma_value,
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42
        }
        
        model = SVC(**params)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()
    
    svm_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    svm_study.optimize(svm_objective, n_trials=n_trials, show_progress_bar=True)
    
    # Train best SVM with properly formatted parameters
    best_svm_params = svm_study.best_params.copy()
    
    # Extract gamma value based on type
    if best_svm_params['gamma_type'] == 'preset':
        gamma_value = best_svm_params['gamma_preset']
    else:
        gamma_value = best_svm_params['gamma_manual']
    
    # Create clean parameters for SVC
    svm_final_params = {
        'C': best_svm_params['C'],
        'gamma': gamma_value,
        'kernel': 'rbf',
        'probability': True,
        'random_state': 42
    }
    
    best_svm = SVC(**svm_final_params)
    best_svm.fit(X_train_scaled, y_train)
    svm_pred = best_svm.predict(X_test_scaled)
    svm_proba = best_svm.predict_proba(X_test_scaled)[:, 1]
    
    optimization_results['SVM (Optimized)'] = {
        'model': best_svm,
        'params': svm_final_params,  # Use clean parameters
        'cv_score': svm_study.best_value,
        'test_accuracy': accuracy_score(y_test, svm_pred),
        'test_f1': f1_score(y_test, svm_pred),
        'test_roc_auc': roc_auc_score(y_test, svm_proba),
        'predictions': svm_pred,
        'probabilities': svm_proba
    }
    
    print(f"Best SVM CV ROC-AUC: {svm_study.best_value:.4f}")
    print(f"Best SVM parameters: {svm_final_params}")  # Show clean parameters
    
    # 2. Optimize XGBoost
    print("\n🔍 Optimizing XGBoost...")
    
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
        
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()
    
    xgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=True)
    
    # Train best XGBoost
    best_xgb = XGBClassifier(**xgb_study.best_params, random_state=42)
    best_xgb.fit(X_train, y_train)
    xgb_pred = best_xgb.predict(X_test)
    xgb_proba = best_xgb.predict_proba(X_test)[:, 1]
    
    optimization_results['XGBoost (Optimized)'] = {
        'model': best_xgb,
        'params': xgb_study.best_params,
        'cv_score': xgb_study.best_value,
        'test_accuracy': accuracy_score(y_test, xgb_pred),
        'test_f1': f1_score(y_test, xgb_pred),
        'test_roc_auc': roc_auc_score(y_test, xgb_proba),
        'predictions': xgb_pred,
        'probabilities': xgb_proba
    }
    
    print(f"Best XGBoost CV ROC-AUC: {xgb_study.best_value:.4f}")
    print(f"Best XGBoost parameters: {xgb_study.best_params}")
    
    # 3. Optimize LightGBM
    print("\n🔍 Optimizing LightGBM...")
    
    def lgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'random_state': 42,
            'verbosity': -1,
            'force_row_wise': True
        }
        
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()
    
    lgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    lgb_study.optimize(lgb_objective, n_trials=n_trials, show_progress_bar=True)
    
    # Train best LightGBM
    best_lgb = LGBMClassifier(**lgb_study.best_params, random_state=42)
    best_lgb.fit(X_train, y_train)
    lgb_pred = best_lgb.predict(X_test)
    lgb_proba = best_lgb.predict_proba(X_test)[:, 1]
    
    optimization_results['LightGBM (Optimized)'] = {
        'model': best_lgb,
        'params': lgb_study.best_params,
        'cv_score': lgb_study.best_value,
        'test_accuracy': accuracy_score(y_test, lgb_pred),
        'test_f1': f1_score(y_test, lgb_pred),
        'test_roc_auc': roc_auc_score(y_test, lgb_proba),
        'predictions': lgb_pred,
        'probabilities': lgb_proba
    }
    
    print(f"Best LightGBM CV ROC-AUC: {lgb_study.best_value:.4f}")
    print(f"Best LightGBM parameters: {lgb_study.best_params}")
    
    # Results comparison
    print("\n📊 Optimized Models Comparison")
    print("=" * 80)
    
    comparison_df = pd.DataFrame({
        model_name: {
            'CV ROC-AUC': f"{results['cv_score']:.4f}",
            'Test Accuracy': f"{results['test_accuracy']:.4f}",
            'Test F1-Score': f"{results['test_f1']:.4f}",
            'Test ROC-AUC': f"{results['test_roc_auc']:.4f}"
        }
        for model_name, results in optimization_results.items()
    }).T
    
    print(comparison_df.to_string())
    
    # Find best optimized model
    best_optimized_name = max(optimization_results.keys(), 
                            key=lambda x: optimization_results[x]['test_roc_auc'])
    best_optimized = optimization_results[best_optimized_name]
    
    print(f"\n🏆 Best Optimized Model: {best_optimized_name}")
    print(f"Test ROC-AUC: {best_optimized['test_roc_auc']:.4f}")
    print(f"Test Accuracy: {best_optimized['test_accuracy']:.4f}")
    print(f"Test F1-Score: {best_optimized['test_f1']:.4f}")
    
    # Detailed classification report
    print(f"\n📝 Detailed Classification Report - {best_optimized_name}")
    print("=" * 60)
    print(classification_report(y_test, best_optimized['predictions'], 
                              target_names=['Normal', 'Abnormal']))
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Model comparison
    plt.subplot(2, 3, 1)
    metrics_df = pd.DataFrame({
        name: [results['test_accuracy'], results['test_f1'], results['test_roc_auc']]
        for name, results in optimization_results.items()
    }, index=['Accuracy', 'F1-Score', 'ROC-AUC'])
    
    metrics_df.plot(kind='bar', ax=plt.gca())
    plt.title('Optimized Models Performance')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 2. Confusion matrix for best model
    plt.subplot(2, 3, 2)
    cm = confusion_matrix(y_test, best_optimized['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix\n{best_optimized_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 3. ROC curve for best model
    plt.subplot(2, 3, 3)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, best_optimized['probabilities'])
    plt.plot(fpr, tpr, label=f'{best_optimized_name} (AUC = {best_optimized["test_roc_auc"]:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Best Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4-6. Optimization history for each model
    studies = [svm_study, xgb_study, lgb_study]
    study_names = ['SVM', 'XGBoost', 'LightGBM']
    
    for i, (study, name) in enumerate(zip(studies, study_names), 4):
        plt.subplot(2, 3, i)
        plt.plot(study.trials_dataframe()['value'])
        plt.title(f'{name} Optimization History')
        plt.xlabel('Trial')
        plt.ylabel('CV ROC-AUC')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'optimization_results': optimization_results,
        'best_model_name': best_optimized_name,
        'best_model': best_optimized['model'],
        'selected_features': selected_features,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test,
        'studies': {
            'svm': svm_study,
            'xgb': xgb_study, 
            'lgb': lgb_study
        }
    }

# Run optimization
optuna_results = optimize_models_with_optuna(df, n_trials=100)

print("\n🎯 Next Steps After Optimization:")
print("1. Run this optimization (will take 5-10 minutes)")
print("2. Compare optimized vs baseline performance")
print("3. Final model validation and deployment preparation")
print("4. Feature importance analysis of final model")