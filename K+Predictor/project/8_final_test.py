# 6_eval_full.py — Evaluate saved model across the full dataset
# Usage:
#   python 6_eval_full.py                # default tol=0.2
#   python 6_eval_full.py --tol 0.15     # custom tolerance
#   python 6_eval_full.py --limit 1000   # evaluate on first 1000 rows
# Notes:
# - Requires: numpy, pandas, scikit-learn, joblib (via your deployment class)
# - Assumes the same file names/structure you shared.

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error
)

from deployment_5 import VitalV2ClinicalDeployment

MODEL_PATH   = "vital_clinical_model.pkl"
FEAT_PATH    = "vital_enhanced_features.csv"
TGT_PATH     = "vital_enhanced_targets.csv"
PREP_PATH    = "vital_prepared_dataset.csv"
RAW_FALLBACK = "1_cleaned_dataset.csv"

def load_eval_frame():
    """Load the widest available evaluation frame (features + targets if possible)."""
    if os.path.exists(FEAT_PATH) and os.path.exists(TGT_PATH):
        X = pd.read_csv(FEAT_PATH)
        y = pd.read_csv(TGT_PATH)
        # Avoid duplicate columns when concat:
        y_only = y[[c for c in y.columns if c not in X.columns]]
        df = pd.concat([X, y_only], axis=1)
        return df
    if os.path.exists(PREP_PATH):
        return pd.read_csv(PREP_PATH)
    if os.path.exists(RAW_FALLBACK):
        return pd.read_csv(RAW_FALLBACK)
    raise FileNotFoundError("No input data found (features/targets/prepared/raw).")

def sanitize_features(frame: pd.DataFrame, feature_cols, schema_medians=None):
    """Ensure required feature columns exist, numeric, imputed with schema medians (fallback: column median)."""
    df = frame.copy()

    # Ensure all required feature columns exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Keep only numeric values for model inputs; coerce others
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace inf/-inf with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Build imputation map: schema medians first, else column median, else 0.0
    med = {}
    schema_medians = schema_medians or {}
    for c in feature_cols:
        if c in schema_medians and np.isfinite(schema_medians[c]):
            med[c] = float(schema_medians[c])
        else:
            col_med = df[c].median()
            med[c] = float(col_med) if pd.notna(col_med) and np.isfinite(col_med) else 0.0

    df[feature_cols] = df[feature_cols].fillna(med)
    return df

def evaluate(d, df, feature_cols, tol=0.2, limit=None):
    """Run predictions and compute metrics for K regression and severity classification."""
    # Optionally limit rows (for quick runs)
    if isinstance(limit, int) and limit > 0:
        df = df.head(limit).copy()

    # Prediction loop
    schema_medians = getattr(d.predictor, "schema_medians", {}) or {}
    df = sanitize_features(df, feature_cols, schema_medians=schema_medians)

    # Prepare ground truth columns (they might not exist in all inputs)
    true_k_col = "potassium"
    true_sev_col = "k_clinical_severity"
    if true_k_col not in df.columns:
        df[true_k_col] = np.nan
    if true_sev_col not in df.columns:
        df[true_sev_col] = ""

    preds = []
    for idx, row in df.iterrows():
        pid = f"case{int(row.get('caseid', idx))}_{idx}"
        feats = {c: float(row[c]) if pd.notna(row[c]) else np.nan for c in feature_cols}

        res = d.predict_k_level(feats, pid)
        pred = (res or {}).get("prediction", {})

        preds.append({
            "patient_id": pid,
            "actual_K": float(row[true_k_col]) if pd.notna(row[true_k_col]) else np.nan,
            "actual_severity": (row[true_sev_col] if isinstance(row[true_sev_col], str) else ""),
            "predicted_K": pred.get("estimated_k_level", np.nan),
            "predicted_class": pred.get("severity_class", "NA"),
            "confidence": pred.get("confidence_score", np.nan),
        })

    out = pd.DataFrame(preds)

    # ----- K-value metrics -----
    # Keep only rows with both actual and predicted K
    k_mask = np.isfinite(out["actual_K"]) & np.isfinite(pd.to_numeric(out["predicted_K"], errors="coerce"))
    k_df = out[k_mask].copy()
    k_mae = k_rmse = k_within_tol = k_within_0_1 = np.nan
    n_k = 0

    if not k_df.empty:
        y_true_k = k_df["actual_K"].astype(float).values
        y_pred_k = k_df["predicted_K"].astype(float).values
        k_mae = mean_absolute_error(y_true_k, y_pred_k)
        k_rmse = mean_squared_error(y_true_k, y_pred_k, squared=False)
        n_k = len(k_df)
        k_within_tol = float(np.mean(np.abs(y_pred_k - y_true_k) <= tol))
        k_within_0_1 = float(np.mean(np.abs(y_pred_k - y_true_k) <= 0.1))

    # ----- Severity classification metrics -----
    c_mask = (out["actual_severity"].astype(str).str.len() > 0) & (out["predicted_class"].astype(str).str.len() > 0)
    c_df = out[c_mask].copy()
    cls_acc = cls_f1_macro = None
    cls_cm = None
    cls_report = ""
    n_c = 0

    if not c_df.empty:
        y_true_c = c_df["actual_severity"].astype(str).values
        y_pred_c = c_df["predicted_class"].astype(str).values

        # Align labels if needed (classification_report will handle unseen labels gracefully)
        labels = sorted(list(set(list(y_true_c) + list(y_pred_c))))
        cls_acc = accuracy_score(y_true_c, y_pred_c)
        cls_f1_macro = f1_score(y_true_c, y_pred_c, average="macro")
        cls_cm = confusion_matrix(y_true_c, y_pred_c, labels=labels)
        cls_report = classification_report(y_true_c, y_pred_c, labels=labels, digits=3)
        n_c = len(c_df)

    # ----- Print summary -----
    print("\n================  EVALUATION SUMMARY  ================")
    print(f"Total rows evaluated: {len(out)}")
    print(f"Rows with K ground-truth: {n_k}")
    print(f"Rows with severity ground-truth: {n_c}")

    print("\n--- K-value (Regression) ---")
    if n_k > 0:
        print(f"MAE:  {k_mae:.3f}")
        print(f"RMSE: {k_rmse:.3f}")
        print(f"Within ±{tol:.2f} mmol/L: {k_within_tol*100:.1f}%")
        print(f"Within ±0.10 mmol/L:      {k_within_0_1*100:.1f}%")
    else:
        print("No valid K pairs to evaluate.")

    print("\n--- Severity (Classification) ---")
    if n_c > 0:
        print(f"Accuracy:   {cls_acc:.3f}")
        print(f"F1 (macro): {cls_f1_macro:.3f}")
        print("\nConfusion matrix (rows=true, cols=pred):")
        labels = sorted(list(set(list(c_df["actual_severity"].astype(str)) + list(c_df["predicted_class"].astype(str)))))
        print("Labels:", labels)
        print(pd.DataFrame(cls_cm, index=labels, columns=labels))
        print("\nClassification report:")
        print(cls_report)
    else:
        print("No valid severity pairs to evaluate.")

    # Show a small head of the output for sanity
    print("\n--- Sample of predictions ---")
    with pd.option_context("display.max_rows", 10, "display.width", 180):
        print(out.head(10).to_string(index=False))

    return out

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate K+ regression and severity classification over the dataset.")
    ap.add_argument("--tol", type=float, default=0.2, help="Tolerance for K-value accuracy (default: 0.2 mmol/L).")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only the first N rows (0 = all).")
    return ap.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    d = VitalV2ClinicalDeployment(MODEL_PATH)
    if d.predictor is None:
        print("❌ Failed to load predictor")
        return

    df = load_eval_frame()
    feature_cols = d.feature_names
    if not feature_cols:
        print("❌ Model has empty feature_names.")
        return

    print(f"Loaded data with {len(df)} rows.")
    print(f"Model expects {len(feature_cols)} features.")
    _ = evaluate(d, df, feature_cols, tol=float(args.tol), limit=args.limit)

if __name__ == "__main__":
    main()
