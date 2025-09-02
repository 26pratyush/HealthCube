from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

try:
    from imblearn.over_sampling import SMOTE
    _HAS_IMBLEARN = True
except Exception:
    _HAS_IMBLEARN = False

VALID_LABELS = ["severe_hypo", "moderate_hypo", "normal", "mild_hyper", "severe_hyper"]


@dataclass
class ClinicalKPotassiumPredictor:
    feature_names: List[str] = field(default_factory=list)
    label_encoder: Optional[LabelEncoder] = None
    models: Dict[str, object] = field(default_factory=dict)
    schema_medians: Dict[str, float] = field(default_factory=dict)

    def _force_numeric_and_impute(self, X_in: pd.DataFrame) -> pd.DataFrame:
        X = X_in.select_dtypes(include=[np.number]).copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        med = {}
        if self.schema_medians:
            med = {k: v for k, v in self.schema_medians.items() if k in X.columns}
        fallback = X.median(numeric_only=True).to_dict()
        for col in X.columns:
            if col not in med:
                med[col] = fallback.get(col, 0.0)
        all_nan = [c for c in X.columns if X[c].isna().all()]
        for c in all_nan:
            med[c] = 0.0
        X = X.fillna(med)
        if X.isna().any().any():
            raise ValueError("NaNs remain after imputation.")
        return X

    def load_enhanced_data(self, features_path="vital_enhanced_features.csv",
                           targets_path="vital_enhanced_targets.csv",
                           schema_path="feature_schema.json") -> pd.DataFrame:
        X = pd.read_csv(features_path)
        y_df = pd.read_csv(targets_path)
        if "k_clinical_severity" not in y_df.columns:
            raise ValueError("k_clinical_severity missing.")
        y = y_df["k_clinical_severity"].astype(str)
        mask = y.isin(VALID_LABELS)
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
        try:
            with open(schema_path, "r") as f:
                schema = json.load(f)
            self.schema_medians = schema.get("feature_medians", {})
        except Exception:
            self.schema_medians = {}
        self.X = self._force_numeric_and_impute(X)
        self.feature_names = list(self.X.columns)
        self.y_severity = y
        self.y_regression = y_df["potassium"].loc[mask].reset_index(drop=True) if "potassium" in y_df.columns else None
        le = LabelEncoder()
        le.fit(VALID_LABELS)
        self.label_encoder = le
        return pd.concat([self.X, self.y_severity.rename("k_clinical_severity")], axis=1)

    def train_specialist_models(self, X: pd.DataFrame, y: pd.Series,
                                cv_splits: int = 5, random_state: int = 42) -> Dict[str, float]:
        y_hypo = y.isin(["severe_hypo", "moderate_hypo"]).astype(int)
        y_hyper = y.isin(["mild_hyper", "severe_hyper"]).astype(int)
        y_highrisk = y.isin(["severe_hypo", "severe_hyper"]).astype(int)
        models_cfg = {
            "hypo_detector": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state),
            "hyper_detector": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state),
            "high_risk_detector": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state),
            "severity_classifier": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=random_state),
        }
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        models_cfg["hypo_detector"].fit(X, y_hypo)
        models_cfg["hyper_detector"].fit(X, y_hyper)
        models_cfg["high_risk_detector"].fit(X, y_highrisk)
        y_multi = self.label_encoder.transform(y)
        models_cfg["severity_classifier"].fit(X, y_multi)
        cv_scores = {
            "hypo_detector_auc": cross_val_score(models_cfg["hypo_detector"], X, y_hypo, cv=cv, scoring="roc_auc").mean(),
            "hyper_detector_auc": cross_val_score(models_cfg["hyper_detector"], X, y_hyper, cv=cv, scoring="roc_auc").mean(),
            "high_risk_detector_auc": cross_val_score(models_cfg["high_risk_detector"], X, y_highrisk, cv=cv, scoring="roc_auc").mean(),
            "severity_classifier_acc": cross_val_score(models_cfg["severity_classifier"], X, y_multi, cv=cv, scoring="accuracy").mean(),
        }
        self.models = models_cfg
        return cv_scores

    def ensemble_predict(self, X_in: pd.DataFrame | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = pd.DataFrame(X_in, columns=self.feature_names) if isinstance(X_in, np.ndarray) else X_in.reindex(columns=self.feature_names, fill_value=0)
        clf = self.models["severity_classifier"]
        proba_mc = clf.predict_proba(X)
        base_ids = np.argmax(proba_mc, axis=1)
        base_labels = self.label_encoder.inverse_transform(base_ids)
        base_conf = proba_mc[np.arange(len(X)), base_ids]
        prob_hypo = self.models["hypo_detector"].predict_proba(X)[:, 1]
        prob_hyper = self.models["hyper_detector"].predict_proba(X)[:, 1]
        prob_high = self.models["high_risk_detector"].predict_proba(X)[:, 1]
        final_pred, final_conf = [], []
        for i, pred in enumerate(base_labels):
            p = float(base_conf[i])
            if prob_high[i] > 0.80:
                if pred == "mild_hyper" and prob_hyper[i] > 0.60:
                    pred, p = "severe_hyper", max(p, prob_hyper[i], prob_high[i])
                elif pred == "moderate_hypo" and prob_hypo[i] > 0.60:
                    pred, p = "severe_hypo", max(p, prob_hypo[i], prob_high[i])
                elif pred == "normal":
                    if prob_hyper[i] >= prob_hypo[i] and prob_hyper[i] > 0.60:
                        pred, p = "mild_hyper", max(p, prob_hyper[i])
                    elif prob_hypo[i] > 0.60:
                        pred, p = "moderate_hypo", max(p, prob_hypo[i])
            final_pred.append(pred); final_conf.append(p)
        return np.array(final_pred), np.array(final_conf)

    def clinical_evaluation(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        X_tr, X_te, y_tr, y_te = train_test_split(self.X, self.y_severity, test_size=test_size, stratify=self.y_severity, random_state=random_state)
        if _HAS_IMBLEARN:
            y_tr_enc = self.label_encoder.transform(y_tr)
            sm = SMOTE(k_neighbors=min(3, max(1, len(np.unique(y_tr_enc)) - 1))), 
            sm = SMOTE(k_neighbors=min(3, max(1, len(np.unique(y_tr_enc)) - 1))),  # ensure no tuple
            sm = SMOTE(k_neighbors=min(3, max(1, len(np.unique(y_tr_enc)) - 1)))
            X_tr_res, y_tr_res_enc = sm.fit_resample(X_tr, y_tr_enc)
            y_tr_res = pd.Series(self.label_encoder.inverse_transform(y_tr_res_enc), name="k_clinical_severity")
        else:
            X_tr_res, y_tr_res = X_tr, y_tr
        self.train_specialist_models(X_tr_res, y_tr_res)
        y_pred, _ = self.ensemble_predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        danger = ["severe_hypo", "severe_hyper"]
        m = y_te.isin(danger).values
        sens = float((y_pred[m] == y_te.values[m]).mean()) if m.sum() > 0 else None
        print("Accuracy:", round(acc, 3))
        print("Dangerous Sensitivity:", None if sens is None else round(sens, 3))
        print(classification_report(y_te, y_pred, labels=VALID_LABELS, zero_division=0))
        print(confusion_matrix(y_te, y_pred, labels=VALID_LABELS))
        return {"overall_accuracy": float(acc), "dangerous_sensitivity": sens}


def run_complete_clinical_modeling(features_path="vital_enhanced_features.csv",
                                   targets_path="vital_enhanced_targets.csv") -> Tuple[ClinicalKPotassiumPredictor, Dict[str, float]]:
    predictor = ClinicalKPotassiumPredictor()
    predictor.load_enhanced_data(features_path, targets_path)
    metrics = predictor.clinical_evaluation()
    return predictor, metrics


if __name__ == "__main__":
    pred, metrics = run_complete_clinical_modeling()
    from deployment_5 import VitalV2ClinicalDeployment
    deployer = VitalV2ClinicalDeployment()
    deployer.save_model(pred, filepath="vital_clinical_model.pkl")
    print("Saved: vital_clinical_model.pkl")
