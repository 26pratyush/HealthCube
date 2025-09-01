import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Load your data (replace 'your_file.csv' with your actual filename)
df = pd.read_csv('ecg_features_cleaned.csv')

def analyze_feature_importance(df):
    # Prepare features and target
    # Exclude non-feature columns
    feature_cols = [col for col in df.columns if col not in ['label', 'filename', 'patient_id']]
    X = df[feature_cols]
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for some analyses
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("🔍 ECG Feature Importance Analysis")
    print("=" * 50)
    print(f"Total features: {len(feature_cols)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # 1. Random Forest Feature Importance
    print("\n📊 1. Random Forest Feature Importance")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    rf_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 15 Most Important Features (Random Forest):")
    print(rf_importance.head(15).to_string(index=False))
    
    # 2. Statistical Feature Selection (F-test)
    print("\n📈 2. Statistical Feature Selection (F-test)")
    f_selector = SelectKBest(score_func=f_classif, k='all')
    f_selector.fit(X_train_scaled, y_train)
    
    f_scores = pd.DataFrame({
        'feature': feature_cols,
        'f_score': f_selector.scores_,
        'p_value': f_selector.pvalues_
    }).sort_values('f_score', ascending=False)
    
    print("Top 15 Features by F-score:")
    print(f_scores.head(15)[['feature', 'f_score']].to_string(index=False))
    
    # 3. Correlation with target
    print("\n🎯 3. Correlation with Target Variable")
    correlations = df[feature_cols + ['label']].corr()['label'].abs().sort_values(ascending=False)[1:]  # Exclude self-correlation
    
    print("Top 15 Features by Absolute Correlation with Target:")
    for feature, corr in correlations.head(15).items():
        print(f"{feature:25} {corr:.4f}")
    
    # 4. Identify highly correlated features (for removal)
    print("\n🔗 4. Highly Correlated Feature Pairs (|r| > 0.8)")
    corr_matrix = df[feature_cols].corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    )
    
    high_corr_pairs = []
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            if upper_tri.loc[idx, col] > 0.8:
                high_corr_pairs.append((idx, col, upper_tri.loc[idx, col]))
    
    if high_corr_pairs:
        print("Highly correlated pairs (consider removing one from each pair):")
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
            print(f"{feat1:25} <-> {feat2:25} (r = {corr:.3f})")
    else:
        print("No highly correlated feature pairs found.")
    
    # 5. Recommended feature selection
    print("\n✅ 5. Recommended Feature Selection")
    
    # Combine different importance measures
    feature_ranking = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf.feature_importances_,
        'f_score': f_selector.scores_,
        'target_corr': [abs(df[feature_cols + ['label']].corr()['label'][feat]) for feat in feature_cols]
    })
    
    # Normalize scores to 0-1 range
    feature_ranking['rf_importance_norm'] = (feature_ranking['rf_importance'] - feature_ranking['rf_importance'].min()) / (feature_ranking['rf_importance'].max() - feature_ranking['rf_importance'].min())
    feature_ranking['f_score_norm'] = (feature_ranking['f_score'] - feature_ranking['f_score'].min()) / (feature_ranking['f_score'].max() - feature_ranking['f_score'].min())
    feature_ranking['target_corr_norm'] = (feature_ranking['target_corr'] - feature_ranking['target_corr'].min()) / (feature_ranking['target_corr'].max() - feature_ranking['target_corr'].min())
    
    # Combined score (equal weights)
    feature_ranking['combined_score'] = (
        feature_ranking['rf_importance_norm'] + 
        feature_ranking['f_score_norm'] + 
        feature_ranking['target_corr_norm']
    ) / 3
    
    feature_ranking = feature_ranking.sort_values('combined_score', ascending=False)
    
    # Suggest different feature set sizes
    print("\nTop 20 Features (Combined Ranking):")
    top_20_features = feature_ranking.head(20)['feature'].tolist()
    for i, feature in enumerate(top_20_features, 1):
        print(f"{i:2d}. {feature}")
    
    print(f"\n📋 Feature Selection Recommendations:")
    print(f"• Top 10 features: {top_20_features[:10]}")
    print(f"• Top 15 features: {top_20_features[:15]}")
    print(f"• Top 20 features: {top_20_features[:20]}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Top 20 RF importance
    plt.subplot(2, 2, 1)
    top_rf = rf_importance.head(20)
    plt.barh(range(len(top_rf)), top_rf['importance'])
    plt.yticks(range(len(top_rf)), top_rf['feature'])
    plt.xlabel('Random Forest Importance')
    plt.title('Top 20 Features - RF Importance')
    plt.gca().invert_yaxis()
    
    # Plot 2: Top 20 F-scores
    plt.subplot(2, 2, 2)
    top_f = f_scores.head(20)
    plt.barh(range(len(top_f)), top_f['f_score'])
    plt.yticks(range(len(top_f)), top_f['feature'])
    plt.xlabel('F-Score')
    plt.title('Top 20 Features - F-Score')
    plt.gca().invert_yaxis()
    
    # Plot 3: Target correlation
    plt.subplot(2, 2, 3)
    top_corr = correlations.head(20)
    plt.barh(range(len(top_corr)), top_corr.values)
    plt.yticks(range(len(top_corr)), top_corr.index)
    plt.xlabel('|Correlation with Target|')
    plt.title('Top 20 Features - Target Correlation')
    plt.gca().invert_yaxis()
    
    # Plot 4: Combined ranking
    plt.subplot(2, 2, 4)
    top_combined = feature_ranking.head(20)
    plt.barh(range(len(top_combined)), top_combined['combined_score'])
    plt.yticks(range(len(top_combined)), top_combined['feature'])
    plt.xlabel('Combined Score')
    plt.title('Top 20 Features - Combined Ranking')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'top_10_features': top_20_features[:10],
        'top_15_features': top_20_features[:15], 
        'top_20_features': top_20_features[:20],
        'all_rankings': feature_ranking,
        'high_corr_pairs': high_corr_pairs,
        'rf_model': rf,
        'scaler': scaler
    }

# Run the analysis
results = analyze_feature_importance(df)

print("\n🚀 Next Steps:")
print("1. Run this analysis on your dataset")
print("2. Choose a feature set size (10, 15, or 20 features)")  
print("3. Proceed to baseline modeling with selected features")
