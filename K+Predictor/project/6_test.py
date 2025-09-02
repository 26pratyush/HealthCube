# 6_test.py — evaluate saved model on prepared/enhanced data
import os, sys, importlib.util
import pandas as pd
import numpy as np
from deployment_5 import VitalV2ClinicalDeployment

MODEL_PATH = "vital_clinical_model.pkl"
FEAT_PATH  = "vital_enhanced_features.csv"
TGT_PATH   = "vital_enhanced_targets.csv"
PREP_PATH  = "vital_prepared_dataset.csv"
RAW_FALLBACK = "1_cleaned_dataset.csv"
N_SAMPLES = 30

def _load_eval_frame():
    if os.path.exists(FEAT_PATH) and os.path.exists(TGT_PATH):
        X = pd.read_csv(FEAT_PATH)
        y = pd.read_csv(TGT_PATH)
        df = pd.concat([X, y], axis=1)
        return df
    if os.path.exists(PREP_PATH):
        return pd.read_csv(PREP_PATH)
    if os.path.exists(RAW_FALLBACK):
        return pd.read_csv(RAW_FALLBACK)
    raise FileNotFoundError("No input data found.")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}"); return

    d = VitalV2ClinicalDeployment(MODEL_PATH)
    if d.predictor is None:
        print("❌ Failed to load predictor"); return

    df = _load_eval_frame()

    feature_cols = d.feature_names
    if not feature_cols:
        print("❌ Model has no feature_names"); return
    print(f"Model expects {len(feature_cols)} features")

    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feature_cols].copy()
    X = X.select_dtypes(include=[np.number]).reindex(columns=feature_cols)
    X = X.replace([np.inf, -np.inf], np.nan)

    schema_medians = getattr(d.predictor, "schema_medians", {}) or {}
    med = {}
    if schema_medians:
        med = {c: schema_medians[c] for c in feature_cols if c in schema_medians}
    fallback = X.median(numeric_only=True).to_dict()
    for c in feature_cols:
        if c not in med:
            med[c] = fallback.get(c, 0.0)
    X = X.fillna(med)

    clean = df.copy()
    for c in feature_cols:
        clean[c] = X[c]

    clean_nonempty = clean.dropna(subset=feature_cols, how="all")
    sample = clean_nonempty.sample(n=min(N_SAMPLES, len(clean_nonempty)), random_state=42)

    results = []
    for idx, row in sample.iterrows():
        pid = f"case{int(row.get('caseid', idx))}_{idx}"
        feat = {c: float(row[c]) if pd.notna(row[c]) else np.nan for c in feature_cols}
        res = d.predict_k_level(feat, pid)
        pred = res.get("prediction", {})
        results.append({
            "patient_id": pid,
            "actual_K": float(row.get("potassium", np.nan)) if pd.notna(row.get("potassium", np.nan)) else np.nan,
            "actual_severity": row.get("k_clinical_severity", ""),
            "predicted_class": pred.get("severity_class", "ERROR"),
            "predicted_K": pred.get("estimated_k_level", np.nan),
            "confidence": pred.get("confidence_score", np.nan),
        })

    out = pd.DataFrame(results)
    with pd.option_context("display.max_rows", None, "display.width", 150):
        print(f"\n=== Prediction vs Actual ({len(out)} patients) ===\n")
        print(out.to_string(index=False))

    print("Expected:", feature_cols[:10])
    if not sample.empty:
        last_feat = {k: sample.iloc[0][k] for k in feature_cols[:10]}
        print("Row features (example):", list(last_feat.keys()))

if __name__ == "__main__":
    main()
