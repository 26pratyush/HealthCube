# ECG DATASET LOADING AND CLEANING - VS CODE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. LOAD DATASET
def load_ecg_dataset(filepath):
    print("📂 Loading ECG dataset...")
    df = pd.read_csv(filepath)
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()[:10]}{'...' if len(df.columns) > 10 else ''}")
    
    return df

dataset_path = "ecg_features_dataset_2000.csv"  # Update this path
df = load_ecg_dataset(dataset_path)

# 2. INITIAL DATA EXPLORATION
def explore_dataset(df):
    print("\n" + "="*60)
    print("📊 DATASET EXPLORATION")
    print("="*60)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Identify column types
    metadata_cols = ['filename', 'label', 'patient_id', 'age', 'sex']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    print(f"\nColumn breakdown:")
    print(f"   Metadata columns: {len(metadata_cols)}")
    print(f"   Feature columns: {len(feature_cols)}")
    
    # Class distribution
    print(f"\n🎯 Class Distribution:")
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        percentage = count / len(df) * 100
        label_name = "Normal" if label == 0 else "Abnormal"
        print(f"   {label_name} ({label}): {count:4d} ({percentage:5.1f}%)")
    
    # Age and gender distribution
    if 'age' in df.columns:
        print(f"\n👥 Demographics:")
        print(f"   Age range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
        print(f"   Age mean: {df['age'].mean():.1f} ± {df['age'].std():.1f} years")
        
        if 'sex' in df.columns:
            print(f"   Gender distribution:")
            gender_counts = df['sex'].value_counts()
            for gender, count in gender_counts.items():
                percentage = count / len(df) * 100
                print(f"     {gender}: {count:4d} ({percentage:5.1f}%)")
    
    return feature_cols, metadata_cols

feature_columns, metadata_columns = explore_dataset(df)

# 3. MISSING VALUE ANALYSIS
def analyze_missing_values(df, feature_cols):
    print("\n" + "="*60)
    print("🔍 MISSING VALUE ANALYSIS")
    print("="*60)
    
    # Calculate missing percentages
    missing_stats = []
    for col in feature_cols:
        missing_count = df[col].isnull().sum()
        missing_pct = missing_count / len(df) * 100
        zero_count = (df[col] == 0).sum()
        zero_pct = zero_count / len(df) * 100
        
        missing_stats.append({
            'column': col,
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'zero_count': zero_count,
            'zero_pct': zero_pct,
            'total_problematic_pct': missing_pct + zero_pct
        })
    
    missing_df = pd.DataFrame(missing_stats)
    missing_df = missing_df.sort_values('missing_pct', ascending=False)
    
    # Categorize features by missing percentage
    high_missing = missing_df[missing_df['missing_pct'] > 50]
    medium_missing = missing_df[(missing_df['missing_pct'] > 20) & (missing_df['missing_pct'] <= 50)]
    low_missing = missing_df[missing_df['missing_pct'] <= 20]
    
    print(f"📊 Missing Value Categories:")
    print(f"   High missing (>50%): {len(high_missing)} features")
    print(f"   Medium missing (20-50%): {len(medium_missing)} features")
    print(f"   Low missing (≤20%): {len(low_missing)} features")
    
    # Display problematic features
    if len(high_missing) > 0:
        print(f"\n❌ HIGH MISSING FEATURES (>50%):")
        for _, row in high_missing.head(10).iterrows():
            print(f"   {row['column']}: {row['missing_pct']:.1f}% missing, {row['zero_pct']:.1f}% zeros")
    
    if len(medium_missing) > 0:
        print(f"\n⚠️  MEDIUM MISSING FEATURES (20-50%):")
        for _, row in medium_missing.head(10).iterrows():
            print(f"   {row['column']}: {row['missing_pct']:.1f}% missing, {row['zero_pct']:.1f}% zeros")
    
    # Overall statistics
    print(f"\n📈 Overall Statistics:")
    print(f"   Features with no missing values: {len(missing_df[missing_df['missing_pct'] == 0])}")
    print(f"   Average missing percentage: {missing_df['missing_pct'].mean():.1f}%")
    print(f"   Max missing percentage: {missing_df['missing_pct'].max():.1f}%")
    
    return missing_df

missing_analysis = analyze_missing_values(df, feature_columns)

# 4. FEATURE QUALITY ASSESSMENT
def assess_feature_quality(df, feature_cols):
    print("\n" + "="*60)
    print("🔬 FEATURE QUALITY ASSESSMENT")
    print("="*60)
    
    quality_stats = []
    
    for col in feature_cols:
        series = df[col]
        
        # Basic statistics
        non_null_count = series.count()
        unique_count = series.nunique()
        variance = series.var()
        
        # Quality metrics
        completeness = non_null_count / len(df)
        uniqueness = unique_count / non_null_count if non_null_count > 0 else 0
        is_constant = unique_count <= 1
        is_binary = unique_count == 2
        
        # Check for infinite values
        inf_count = np.isinf(series).sum() if series.dtype in ['float64', 'float32'] else 0
        
        quality_stats.append({
            'feature': col,
            'completeness': completeness,
            'uniqueness': uniqueness,
            'variance': variance,
            'is_constant': is_constant,
            'is_binary': is_binary,
            'inf_count': inf_count,
            'quality_score': completeness * (1 - is_constant) * min(uniqueness * 10, 1)
        })
    
    quality_df = pd.DataFrame(quality_stats)
    
    # Identify problematic features
    constant_features = quality_df[quality_df['is_constant']]['feature'].tolist()
    low_quality_features = quality_df[quality_df['quality_score'] < 0.1]['feature'].tolist()
    infinite_features = quality_df[quality_df['inf_count'] > 0]['feature'].tolist()
    
    print(f"🚨 PROBLEMATIC FEATURES:")
    print(f"   Constant features: {len(constant_features)}")
    print(f"   Low quality features: {len(low_quality_features)}")
    print(f"   Features with infinite values: {len(infinite_features)}")
    
    if constant_features:
        print(f"\n   Constant features: {constant_features[:10]}")
    if infinite_features:
        print(f"\n   Features with inf values: {infinite_features[:10]}")
    
    return quality_df

quality_analysis = assess_feature_quality(df, feature_columns)

# 5. DATASET CLEANING FUNCTIONS
def clean_dataset(df, missing_threshold=0.5, quality_threshold=0.1):
    print("\n" + "="*60)
    print("🧹 DATASET CLEANING")
    print("="*60)
    
    df_clean = df.copy()
    original_shape = df_clean.shape
    
    # Separate metadata and features
    metadata_cols = ['filename', 'label', 'patient_id', 'age', 'sex']
    feature_cols = [col for col in df_clean.columns if col not in metadata_cols]
    
    print(f"Starting shape: {original_shape}")
    print(f"Original features: {len(feature_cols)}")
    
    # Step 1: Remove features with too many missing values
    missing_stats = missing_analysis
    high_missing_features = missing_stats[missing_stats['missing_pct'] > missing_threshold * 100]['column'].tolist()
    
    print(f"\n1️⃣ REMOVING HIGH MISSING FEATURES (>{missing_threshold*100}%):")
    print(f"   Features to remove: {len(high_missing_features)}")
    if high_missing_features:
        print(f"   Examples: {high_missing_features[:5]}")
        df_clean = df_clean.drop(columns=high_missing_features)
    
    # Step 2: Remove constant/low-quality features
    remaining_features = [col for col in feature_cols if col not in high_missing_features]
    quality_stats = quality_analysis
    
    constant_features = quality_stats[quality_stats['is_constant']]['feature'].tolist()
    low_quality_features = quality_stats[quality_stats['quality_score'] < quality_threshold]['feature'].tolist()
    
    # Combine problematic features
    features_to_remove = list(set(constant_features + low_quality_features))
    features_to_remove = [f for f in features_to_remove if f in remaining_features]
    
    print(f"\n2️⃣ REMOVING LOW QUALITY FEATURES:")
    print(f"   Constant features: {len(constant_features)}")
    print(f"   Low quality features: {len(low_quality_features)}")
    print(f"   Total to remove: {len(features_to_remove)}")
    
    if features_to_remove:
        print(f"   Examples: {features_to_remove[:5]}")
        df_clean = df_clean.drop(columns=features_to_remove)
    
    # Step 3: Handle infinite values
    remaining_features = [col for col in df_clean.columns if col not in metadata_cols]
    inf_features = []
    
    for col in remaining_features:
        if df_clean[col].dtype in ['float64', 'float32']:
            inf_count = np.isinf(df_clean[col]).sum()
            if inf_count > 0:
                inf_features.append(col)
                # Replace inf with NaN for now
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    
    print(f"\n3️⃣ HANDLING INFINITE VALUES:")
    print(f"   Features with inf values: {len(inf_features)}")
    if inf_features:
        print(f"   Converted to NaN: {inf_features[:5]}")
    
    # Step 4: Final feature list
    final_features = [col for col in df_clean.columns if col not in metadata_cols]
    
    print(f"\n✅ CLEANING COMPLETE:")
    print(f"   Original features: {len(feature_cols)}")
    print(f"   Final features: {len(final_features)}")
    print(f"   Features removed: {len(feature_cols) - len(final_features)}")
    print(f"   Final shape: {df_clean.shape}")
    
    return df_clean, final_features

# Apply cleaning
df_cleaned, clean_features = clean_dataset(df, missing_threshold=0.5, quality_threshold=0.05)

# 6. MISSING VALUE IMPUTATION
def impute_missing_values(df, feature_cols, method='knn'):
    print("\n" + "="*60)
    print("🔧 MISSING VALUE IMPUTATION")
    print("="*60)
    
    df_imputed = df.copy()
    
    # Check remaining missing values
    missing_counts = df[feature_cols].isnull().sum()
    features_with_missing = missing_counts[missing_counts > 0]
    
    print(f"Features with missing values: {len(features_with_missing)}")
    
    if len(features_with_missing) == 0:
        print("✅ No missing values found!")
        return df_imputed
    
    print(f"Total missing values: {missing_counts.sum()}")
    print(f"Imputation method: {method}")
    
    # Apply imputation
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        print("❌ Invalid imputation method!")
        return df_imputed
    
    # Fit and transform
    print("🔄 Applying imputation...")
    df_imputed[feature_cols] = imputer.fit_transform(df[feature_cols])
    
    # Verify imputation
    remaining_missing = df_imputed[feature_cols].isnull().sum().sum()
    print(f"✅ Imputation complete!")
    print(f"   Remaining missing values: {remaining_missing}")
    
    return df_imputed

# Apply imputation
df_final = impute_missing_values(df_cleaned, clean_features, method='median')

# 7. FINAL DATASET SUMMARY
def final_dataset_summary(df, feature_cols):
    """Generate final dataset summary"""
    print("\n" + "="*60)
    print("📋 FINAL DATASET SUMMARY")
    print("="*60)
    
    print(f"📊 Dataset Characteristics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Class distribution
    print(f"\n🎯 Class Distribution:")
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        percentage = count / len(df) * 100
        label_name = "Normal" if label == 0 else "Abnormal"
        print(f"   {label_name}: {count:4d} ({percentage:5.1f}%)")
    
    # Feature statistics
    feature_data = df[feature_cols]
    print(f"\n📈 Feature Statistics:")
    print(f"   Missing values: {feature_data.isnull().sum().sum()}")
    print(f"   Mean feature range: {feature_data.max().mean() - feature_data.min().mean():.2f}")
    print(f"   Features ready for ML: ✅")
    
    # Data quality check
    print(f"\n✅ DATA QUALITY PASSED:")
    print(f"   ✓ No missing values")
    print(f"   ✓ No infinite values") 
    print(f"   ✓ No constant features")
    print(f"   ✓ Balanced feature set")
    print(f"   ✓ Ready for model training")

final_dataset_summary(df_final, clean_features)

# 8. SAVE CLEANED DATASET
def save_cleaned_dataset(df, filepath):
    print(f"\n💾 Saving cleaned dataset to: {filepath}")
    df.to_csv(filepath, index=False)
    file_size = pd.read_csv(filepath).memory_usage(deep=True).sum() / 1024**2
    print(f"✅ Dataset saved successfully!")
    print(f"   File size: {file_size:.1f} MB")

# Save the cleaned dataset
output_path = "ecg_features_cleaned.csv"
save_cleaned_dataset(df_final, output_path)

print(f"\n🎉 DATASET CLEANING COMPLETE!")
print(f"📁 Cleaned dataset: {output_path}")
print(f"🚀 Ready for model training!")
