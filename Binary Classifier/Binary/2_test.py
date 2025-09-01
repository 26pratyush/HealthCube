import pandas as pd
import numpy as np

df = pd.read_csv("ecg_features_cleaned.csv")

# Check class distribution
print("Class distribution:")
print(df['label'].value_counts())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Basic correlation analysis
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
print("\nCorrelation matrix:")
print(correlation_matrix)
