import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load and analyze your current dataset
def analyze_current_dataset(df_path_or_dataframe):
    """
    Comprehensive analysis of your VitalV2 dataset to determine if we can proceed
    or need to regenerate data
    """
    
    # If path is provided, load the data
    if isinstance(df_path_or_dataframe, str):
        df = pd.read_csv(df_path_or_dataframe)
    else:
        df = df_path_or_dataframe.copy()
    
    print("=== VITALV2 DATASET ANALYSIS ===\n")
    
    # Basic dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    
    # Analyze class distribution
    print("\n1. CLASS DISTRIBUTION ANALYSIS")
    print("-" * 40)
    k_status_dist = df['k_status'].value_counts()
    k_status_pct = df['k_status'].value_counts(normalize=True) * 100
    
    print("Current distribution:")
    for status in k_status_dist.index:
        count = k_status_dist[status]
        pct = k_status_pct[status]
        print(f"  {status}: {count} samples ({pct:.1f}%)")
    
    # Analyze potassium value ranges
    print("\n2. POTASSIUM VALUE DISTRIBUTION")
    print("-" * 40)
    print(f"K+ range: {df['potassium'].min():.1f} - {df['potassium'].max():.1f} mEq/L")
    print(f"K+ mean ± std: {df['potassium'].mean():.1f} ± {df['potassium'].std():.1f}")
    
    # Critical ranges analysis
    hypokalemia_severe = (df['potassium'] < 3.0).sum()
    hypokalemia_moderate = ((df['potassium'] >= 3.0) & (df['potassium'] < 3.5)).sum()
    normal_range = ((df['potassium'] >= 3.5) & (df['potassium'] <= 5.0)).sum()
    hyperkalemia_mild = ((df['potassium'] > 5.0) & (df['potassium'] <= 5.5)).sum()
    hyperkalemia_severe = (df['potassium'] > 5.5).sum()
    
    print("\nClinical severity breakdown:")
    print(f"  Severe hypokalemia (<3.0): {hypokalemia_severe} samples")
    print(f"  Moderate hypokalemia (3.0-3.4): {hypokalemia_moderate} samples")
    print(f"  Normal (3.5-5.0): {normal_range} samples")
    print(f"  Mild hyperkalemia (5.1-5.5): {hyperkalemia_mild} samples")
    print(f"  Severe hyperkalemia (>5.5): {hyperkalemia_severe} samples")
    
    # Data quality analysis
    print("\n3. DATA QUALITY ASSESSMENT")
    print("-" * 40)
    
    # Missing values
    missing_summary = df.isnull().sum()
    missing_features = missing_summary[missing_summary > 0]
    
    if len(missing_features) > 0:
        print("Features with missing values:")
        for feature, count in missing_features.items():
            pct = (count / len(df)) * 100
            print(f"  {feature}: {count} ({pct:.1f}%)")
    else:
        print("✅ No missing values found")
    
    # Signal quality analysis
    if 'signal_quality' in df.columns:
        print(f"\nSignal quality range: {df['signal_quality'].min():.3f} - {df['signal_quality'].max():.3f}")
        low_quality = (df['signal_quality'] < 0.6).sum()
        print(f"Low quality signals (<0.6): {low_quality} samples ({low_quality/len(df)*100:.1f}%)")
    
    # Feature analysis
    print("\n4. FEATURE SET ANALYSIS")
    print("-" * 40)
    
    # Categorize features
    ecg_features = [col for col in df.columns if any(x in col.lower() for x in ['ecg', 'rate', 'qt', 'pr', 'qrs', 'duration', 'amplitude'])]
    hrv_features = [col for col in df.columns if col.startswith('HRV_')]
    clinical_features = ['caseid', 'potassium', 'k_status', 'segment_type', 'segment_duration', 'signal_quality', 'lab_time', 'time_to_lab']
    
    print(f"ECG-related features: {len(ecg_features)}")
    print(f"HRV features: {len(hrv_features)}")
    print(f"Clinical metadata: {len(clinical_features)}")
    
    # Check for key clinical features
    key_features = ['qt_interval_ms', 'qtc_bazett_ms', 't_wave_amplitude_median', 'r_wave_amplitude_median']
    missing_key = [f for f in key_features if f not in df.columns]
    if missing_key:
        print(f"⚠️  Missing key clinical features: {missing_key}")
    else:
        print("✅ Key clinical features present")
    
    return df, {
        'class_distribution': k_status_dist,
        'severe_cases': hypokalemia_severe + hyperkalemia_severe,
        'data_quality_issues': len(missing_features) > 0,
        'usable_for_redesign': True  # We'll determine this
    }

# Step 2: Determine if dataset is usable or needs regeneration
def assess_dataset_viability(df, analysis_results):
    """
    Determine if we can proceed with current dataset or need to regenerate
    """
    print("\n=== DATASET VIABILITY ASSESSMENT ===\n")
    
    issues = []
    recommendations = []
    can_proceed = True
    
    # Check 1: Class imbalance severity
    class_dist = analysis_results['class_distribution']
    hyperkalemia_pct = (class_dist.get('hyperkalemia', 0) / len(df)) * 100
    
    if hyperkalemia_pct < 10:
        issues.append(f"Severe hyperkalemia underrepresentation: {hyperkalemia_pct:.1f}%")
        if hyperkalemia_pct < 5:
            recommendations.append("CRITICAL: Need more hyperkalemia cases")
            can_proceed = False
    
    # Check 2: Severe cases availability
    severe_cases = analysis_results['severe_cases']
    if severe_cases < 50:
        issues.append(f"Too few severe cases: {severe_cases}")
        recommendations.append("Need more K+ < 3.0 or K+ > 5.5 cases")
        can_proceed = False
    
    # Check 3: Data quality
    if 'signal_quality' in df.columns:
        high_quality_pct = (df['signal_quality'] >= 0.6).mean() * 100
        if high_quality_pct < 80:
            issues.append(f"Low signal quality: {100-high_quality_pct:.1f}% poor quality")
            recommendations.append("Filter out low quality signals")
    
    # Check 4: Feature completeness
    if analysis_results['data_quality_issues']:
        issues.append("Missing values in features")
        recommendations.append("Handle missing values or exclude incomplete records")
    
    # Assessment summary
    print("ISSUES IDENTIFIED:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print(f"\n{'✅ CAN PROCEED' if can_proceed else '❌ NEED DATA REGENERATION'}")
    print(f"Confidence level: {'High' if can_proceed and len(issues) <= 2 else 'Medium' if can_proceed else 'Low'}")
    
    return can_proceed, issues, recommendations

# Step 3: Quick dataset preprocessing for immediate use
def quick_dataset_prep(df):
    """
    Quick preprocessing to make dataset immediately usable while we plan improvements
    """
    print("\n=== QUICK DATASET PREPARATION ===\n")
    
    df_clean = df.copy()
    
    # 1) ECG targeted imputations (avoid inplace to reduce chained-assignment risk)
    ecg_cols = ['p_duration_ms','pr_interval_ms','qrs_duration_ms','qt_interval_ms',
                't_duration_ms','tpeak_tend_ms','qtc_bazett_ms','qtc_fridericia_ms']
    for col in ecg_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            med = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(med)
            print(f"Filled {col} missing values with median: {med:.3f}")

    # 2) Consistent signal quality filter (use 0.6 everywhere)
    if 'signal_quality' in df_clean.columns:
        initial = len(df_clean)
        df_clean = df_clean[df_clean['signal_quality'] >= 0.6].copy()
        print(f"Removed {initial - len(df_clean)} low-quality signals (signal_quality < 0.6)")

    # 3) Feature engineering (unchanged)
    # ... t_r_ratio_enhanced, qt_hr_interaction, hrv_complexity_score ...

    # 4) Severity labels (unchanged thresholds; align your printed text to >5.0 == 5.1+)
    def get_severity_label(k):
        if k < 3.0:           return 'severe_hypo'
        elif k < 3.5:         return 'moderate_hypo'
        elif k <= 5.0:        return 'normal'
        elif k <= 5.5:        return 'mild_hyper'
        else:                 return 'severe_hyper'
    df_clean['k_severity'] = df_clean['potassium'].apply(get_severity_label)

    # 5) Feature selection and robust global imputation (for ALL numeric features)
    exclude = ['caseid','segment_type','segment_duration','lab_time','time_to_lab',
            'potassium','k_status','k_severity','signal_quality']
    feature_cols = [c for c in df_clean.columns if c not in exclude]
    X = df_clean[feature_cols].copy()

    # Impute any remaining numeric NaNs with per-column medians
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    feature_medians = X[numeric_cols].median(numeric_only=True).to_dict()
    X[numeric_cols] = X[numeric_cols].fillna(feature_medians)

    # OPTIONAL: sanitize inf/-inf
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(feature_medians)

    y_regression     = df_clean['potassium']
    y_classification = df_clean['k_severity']

    # 6) Persist schema for downstream steps (feature order + medians)
    import json
    with open('feature_schema.json', 'w') as f:
        json.dump({
            "feature_cols": feature_cols,
            "feature_medians": feature_medians
        }, f, indent=2)
    print("Saved feature schema to feature_schema.json")

    
    # 4. Create severity-based target
    def get_severity_label(k_value):
        if k_value < 3.0:
            return 'severe_hypo'
        elif k_value < 3.5:
            return 'moderate_hypo' 
        elif k_value <= 5.0:
            return 'normal'
        elif k_value <= 5.5:
            return 'mild_hyper'
        else:
            return 'severe_hyper'
    
    df_clean['k_severity'] = df_clean['potassium'].apply(get_severity_label)
    
    # 5. Feature selection for modeling
    # Exclude metadata columns
    exclude_cols = ['caseid', 'segment_type', 'segment_duration', 'lab_time', 'time_to_lab', 
                   'potassium', 'k_status', 'k_severity', 'signal_quality']
    
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    X = df_clean[feature_cols]
    y_regression = df_clean['potassium']
    y_classification = df_clean['k_severity']
    
    print(f"\nFinal dataset preparation:")
    print(f"  Samples: {len(df_clean)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target distribution:")
    
    severity_dist = df_clean['k_severity'].value_counts()
    for severity, count in severity_dist.items():
        pct = (count / len(df_clean)) * 100
        print(f"    {severity}: {count} ({pct:.1f}%)")
    
    return df_clean, X, y_regression, y_classification, feature_cols

# Main execution - Run the full analysis on your dataset
if __name__ == "__main__":
    print("=== VITALV2 DATASET ANALYSIS & PREPARATION ===\n")
    
    try:
        # Load your actual dataset
        print("Loading dataset: '1_cleaned_dataset.csv'")
        df = pd.read_csv('1_cleaned_dataset.csv')
        print(f"✅ Successfully loaded {len(df)} samples\n")
        
        # Step 1: Comprehensive dataset analysis
        print("STEP 1: Analyzing current dataset...")
        df_analyzed, analysis_results = analyze_current_dataset(df)
        
        # Step 2: Assess if we can proceed or need regeneration
        print("\nSTEP 2: Assessing dataset viability...")
        can_proceed, issues, recommendations = assess_dataset_viability(df_analyzed, analysis_results)
        
        # Step 3: Prepare dataset for immediate use
        if can_proceed:
            print("\nSTEP 3: Preparing dataset for modeling...")
            df_clean, X, y_regression, y_classification, feature_cols = quick_dataset_prep(df_analyzed)
            
            print(f"\n🎯 DATASET READY FOR MODELING!")
            print(f"   Shape: {X.shape}")
            print(f"   Features: {len(feature_cols)}")
            print(f"   Ready for Step 2: Clinical Model Development")
            
            # Save prepared dataset for next steps
            df_clean.to_csv('vital_prepared_dataset.csv', index=False)
            print(f"   Saved prepared dataset as 'vital_prepared_dataset.csv'")
            
        else:
            print(f"\n⚠️  DATASET NEEDS ENHANCEMENT")
            print("   We'll proceed with data augmentation strategies in Step 2")

            print("\nSTEP 3: Preparing dataset for modeling...")
            df_clean, X, y_regression, y_classification, feature_cols = quick_dataset_prep(df_analyzed)
            
            print(f"\n🎯 DATASET READY FOR MODELING!")
            print(f"   Shape: {X.shape}")
            print(f"   Features: {len(feature_cols)}")
            print(f"   Ready for Step 2: Clinical Model Development")
            
            # Save prepared dataset for next steps
            df_clean.to_csv('vital_prepared_dataset.csv', index=False)
            print(f"   Saved prepared dataset as 'vital_prepared_dataset.csv'")
            
    except FileNotFoundError:
        print("❌ Error: Could not find '1_cleaned_dataset.csv'")
        print("   Make sure the file is in the same directory as this script")
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        print("   Check file format and column names")
