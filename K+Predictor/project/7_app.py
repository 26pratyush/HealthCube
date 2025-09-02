
import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from deployment_5 import VitalV2ClinicalDeployment

MODEL_PATH = "vital_clinical_model.pkl"
FEAT_PATH  = "vital_enhanced_features.csv"
TGT_PATH   = "vital_enhanced_targets.csv"
PREP_PATH  = "vital_prepared_dataset.csv"
RAW_FALLBACK = "1_cleaned_dataset.csv"

st.set_page_config(page_title="K+ Severity — Inference Explorer", page_icon="🧪", layout="wide")

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        st.stop()
    d = VitalV2ClinicalDeployment(path)
    if d.predictor is None:
        st.error("Failed to load predictor from model file.")
        st.stop()
    return d

@st.cache_data(show_spinner=False)
def load_frame():
    if os.path.exists(FEAT_PATH) and os.path.exists(TGT_PATH):
        X = pd.read_csv(FEAT_PATH)
        y = pd.read_csv(TGT_PATH)
        df = pd.concat([X, y], axis=1)
        src = "enhanced"
    elif os.path.exists(PREP_PATH):
        df = pd.read_csv(PREP_PATH)
        src = "prepared"
    elif os.path.exists(RAW_FALLBACK):
        df = pd.read_csv(RAW_FALLBACK)
        src = "raw"
    else:
        st.error("No input data found (expected one of the CSVs).")
        st.stop()
    return df, src

def sanitize_features(df: pd.DataFrame, feature_cols, medians: dict) -> pd.DataFrame:
    work = df.copy()
    for c in feature_cols:
        if c not in work.columns:
            work[c] = np.nan
    X = work[feature_cols].copy()
    X = X.select_dtypes(include=[np.number]).reindex(columns=feature_cols)
    X = X.replace([np.inf, -np.inf], np.nan)
    med = {c: medians[c] for c in feature_cols if c in medians}
    fallback = X.median(numeric_only=True).to_dict()
    for c in feature_cols:
        if c not in med:
            med[c] = fallback.get(c, 0.0)
    X = X.fillna(med)
    for c in feature_cols:
        work[c] = X[c]
    return work

def make_html_report(patient_id, pred, true_sev, true_k, feature_subset: pd.Series):
    import numpy as np
    from datetime import datetime

    label = pred.get("severity_class", "NA")
    conf  = pred.get("confidence_score", np.nan)
    est_k = pred.get("estimated_k_level", np.nan)

    conf_str = f"{conf:.3f}" if isinstance(conf, (int, float, np.floating)) else str(conf)
    est_k_str = f"{est_k:.3f}" if isinstance(est_k, (int, float, np.floating)) else str(est_k)
    true_k_str = (f"{float(true_k):.3f}" if isinstance(true_k, (int, float, np.floating)) and np.isfinite(true_k)
                  else "NA")

    delta_str = ""
    if isinstance(true_k, (int, float, np.floating)) and np.isfinite(true_k) and isinstance(est_k, (int, float, np.floating)):
        delta_str = f"(Δ={float(est_k) - float(true_k):+.2f})"

    match = None
    if isinstance(true_sev, str) and true_sev:
        match = "MATCH" if (true_sev == label) else "MISMATCH"

    styles = """
    <style>
    body { font-family: Arial, sans-serif; padding: 20px; }
    .hdr { font-size: 22px; font-weight: bold; margin-bottom: 6px; }
    .sub { color: #555; margin-bottom: 16px; }
    .row { margin: 10px 0; }
    .kv { display: grid; grid-template-columns: 220px 1fr; gap: 8px 16px; }
    .ok { color: #0b7; font-weight: bold; }
    .bad { color: #c33; font-weight: bold; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; }
    th { background: #f5f5f5; }
    </style>
    """
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in feature_subset.items())
    match_badge = "" if match is None else (f"<span class='ok'>{match}</span>" if match == "MATCH" else f"<span class='bad'>{match}</span>")

    html = f"""
    <html><head>{styles}</head><body>
    <div class='hdr'>K+ Severity — Prediction Report</div>
    <div class='sub'>Generated {datetime.now().isoformat(timespec='seconds')}</div>

    <div class='row kv'>
      <div>Estimated K (mmol/L)</div><div>{est_k_str}</div>
      <div>True K (mmol/L)</div><div>{true_k_str} {delta_str}</div>
      <div>Predicted Severity</div><div>{label}</div>
      <div>True Severity</div><div>{true_sev if true_sev else 'NA'} {match_badge}</div>
      <div>Patient ID</div><div>{patient_id}</div>
      <div>Confidence</div><div>{conf_str}</div>
    </div>

    <div class='row'>
      <div class='hdr' style='font-size:18px;'>Feature Snapshot</div>
      <table>
        <thead><tr><th>Feature</th><th>Value</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    </body></html>
    """
    return html


st.title("🧪 K+ Severity — Inference Explorer")

d = load_model(MODEL_PATH)
df, source = load_frame()

feature_cols = d.feature_names
if not feature_cols:
    st.error("Model has empty feature_names.")
    st.stop()

schema_medians = getattr(d.predictor, "schema_medians", {}) or {}
df_clean = sanitize_features(df, feature_cols, schema_medians)

left, right = st.columns([1, 2])
with left:
    st.markdown("**Data source**: " + source)
    default_sample = min(200, len(df_clean))
    sample_size = st.slider("Sample size", min_value=20, max_value=min(2000, len(df_clean)), value=default_sample, step=10)
    seed = st.number_input("Random seed", value=42, step=1)
    sample_idx = df_clean.sample(n=sample_size, random_state=int(seed)).index.tolist()
    chosen_idx = st.selectbox("Choose an index from sample", options=sample_idx, index=0)
    st.write(f"Selected index: `{chosen_idx}`")

with right:
    # --- small style tweaks for bigger labels ---
    st.markdown(
        """
    <style>
    .big-label { 
        font-size: 24px; 
        font-weight: 500; 
        line-height: 1.1; 
        margin-top: -8px; 
    }
    </style>
        """,
        unsafe_allow_html=True,
    )

    # Ground truth from row
    true_k = df_clean.loc[chosen_idx].get("potassium", np.nan)
    true_sev = df_clean.loc[chosen_idx].get("k_clinical_severity", "")

    patient_id = f"case{int(df_clean.loc[chosen_idx].get('caseid', chosen_idx))}_{chosen_idx}"

    # Build features and predict
    row_features = {c: float(df_clean.loc[chosen_idx, c]) for c in feature_cols}
    res = d.predict_k_level(row_features, patient_id=patient_id)
    pred = res.get("prediction", {})
    pred_label = pred.get("severity_class", "NA")
    pred_conf = pred.get("confidence_score", np.nan)
    est_k = pred.get("estimated_k_level", np.nan)

    # Safe strings
    conf_str = f"{pred_conf:.3f}" if isinstance(pred_conf, (int, float)) and np.isfinite(pred_conf) else str(pred_conf)
    est_str = f"{float(est_k):.1f}" if isinstance(est_k, (int, float)) and np.isfinite(est_k) else str(est_k)
    true_k_str = f"{float(true_k):.1f}" if isinstance(true_k, (int, float)) and np.isfinite(true_k) else "NA"

    # Delta vs true
    delta_str = None
    if isinstance(true_k, (int, float)) and np.isfinite(true_k) and isinstance(est_k, (int, float)) and np.isfinite(est_k):
        delta_val = float(est_k) - float(true_k)
        delta_str = f"{delta_val:+.2f} vs True"

    # ---------- ROW 1 (Top-right): Estimated K, True K, Confidence ----------
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Estimated K (mmol/L)", est_str, delta_str if delta_str is not None else None)
    r1c2.metric("True K (mmol/L)", true_k_str)
    r1c3.metric("Confidence", conf_str)

    # ---------- ROW 2: Predicted vs True Severity ----------
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("**Predicted Severity**")
        st.markdown(f"<div class='big-label'>{pred_label}</div>", unsafe_allow_html=True)

    with r2c2:
        match = isinstance(true_sev, str) and len(true_sev) > 0 and (true_sev == pred_label)
        pill = "Match ✅" if match else ("Mismatch ❌" if isinstance(true_sev, str) and len(true_sev) > 0 else "Unknown")
        pill_cls = "pill ok" if match else ("pill no" if pill.startswith("MISMATCH") else "pill")
        true_sev_show = true_sev if isinstance(true_sev, str) and true_sev else "NA"
        st.markdown("**True Severity**")
        st.markdown(f"<div class='big-label'>{true_sev_show} &nbsp; <span class='{pill_cls}'>{pill}</span></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:40px;'></div>", unsafe_allow_html=True)
    # ---------- Feature snapshot & report ----------
    subset_view = pd.Series({k: row_features[k] for k in feature_cols[:20]})
    st.subheader("Feature Snapshot (first 20)")
    st.dataframe(subset_view.to_frame("value"))

    html = make_html_report(patient_id, pred, true_sev, true_k, subset_view)
    buf = io.BytesIO(html.encode("utf-8"))
    st.download_button("Download HTML Report", data=buf,
                       file_name=f"report_{patient_id}.html", mime="text/html")

st.caption("Tip: adjust the seed and sample size to explore different rows. Missing features are imputed using the model's saved medians.")
