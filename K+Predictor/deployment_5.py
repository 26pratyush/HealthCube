from __future__ import annotations
import os, sys, json, importlib.util
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import joblib

SEVERITY_TO_K = {
    "severe_hypo": 2.8,
    "moderate_hypo": 3.3,
    "normal": 4.2,
    "mild_hyper": 5.3,
    "severe_hyper": 6.0,
}

def _install_main_alias():
    here = os.path.dirname(os.path.abspath(__file__))
    tr_path = os.path.join(here, "4_resampling_&_modelling.py")
    if os.path.exists(tr_path):
        spec = importlib.util.spec_from_file_location("vital_train", tr_path)
        vital_train = importlib.util.module_from_spec(spec)
        sys.modules["vital_train"] = vital_train
        spec.loader.exec_module(vital_train)
        sys.modules['__main__'].ClinicalKPotassiumPredictor = getattr(vital_train, "ClinicalKPotassiumPredictor", object)

@dataclass
class VitalV2ClinicalDeployment:
    model_path: Optional[str] = None
    predictor: Any = None

    def __post_init__(self):
        if self.model_path:
            self.load_model(self.model_path)

    @property
    def feature_names(self) -> List[str]:
        return getattr(self.predictor, "feature_names", []) if self.predictor is not None else []

    def load_model(self, path: str) -> None:
        _install_main_alias()
        self.predictor = joblib.load(path)

    def save_model(self, predictor_obj: Any, filepath: str = "vital_clinical_model.pkl") -> None:
        joblib.dump(predictor_obj, filepath)

    def _impute_and_order(self, feat: Dict[str, Any]) -> pd.DataFrame:
        cols = self.feature_names
        X = pd.DataFrame([{c: feat.get(c, np.nan) for c in cols}], columns=cols)
        X = X.select_dtypes(include=[np.number]).reindex(columns=cols)
        X = X.replace([np.inf, -np.inf], np.nan)
        schema_medians = getattr(self.predictor, "schema_medians", {}) or {}
        med = {c: schema_medians[c] for c in cols if c in schema_medians}
        fallback = X.median(numeric_only=True).to_dict()
        for c in cols:
            if c not in med:
                med[c] = fallback.get(c, 0.0)
        X = X.fillna(med)
        return X

    def predict_k_level(self, features: Dict[str, Any], patient_id: str = "case") -> Dict[str, Any]:
        if self.predictor is None:
            return {"error": "model_not_loaded"}
        X = self._impute_and_order(features)
        labels, conf = self.predictor.ensemble_predict(X)
        label = str(labels[0])
        confidence = float(conf[0])
        est_k = float(SEVERITY_TO_K.get(label, 4.2))
        return {
            "patient_id": patient_id,
            "prediction": {
                "severity_class": label,
                "confidence_score": confidence,
                "estimated_k_level": est_k
            }
        }

if __name__ == "__main__":
    d = VitalV2ClinicalDeployment("vital_clinical_model.pkl")
    print({"loaded": d.predictor is not None, "n_features": len(d.feature_names)})
