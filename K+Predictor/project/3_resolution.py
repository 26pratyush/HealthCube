# 3_resolution.py — schema-locked enhancement & (optional) classification-only resampling
# -------------------------------------------------------------------------------
# - Consumes Step-1 outputs: vital_prepared_dataset.csv (+ optional feature_schema.json)
# - Enforces the same feature order & medians to prevent train/deploy drift
# - Generates targets (severity) from potassium without overwriting potassium
# - NO SCALING here (avoid leakage) — do it inside CV / model pipelines
# - Optional SMOTE for classification ONLY; regression set remains original
# -------------------------------------------------------------------------------

from __future__ import annotations
import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from imblearn.over_sampling import SMOTE
    _HAS_IMBLEARN = True
except Exception:
    _HAS_IMBLEARN = False
    warnings.warn("imblearn not found: classification resampling will be unavailable.", RuntimeWarning)


# ----------------------------- Helpers / Config -----------------------------

SEVERITY_LABELS = ["severe_hypo", "moderate_hypo", "normal", "mild_hyper", "severe_hyper"]

EXCLUDE_COLS_DEFAULT = {
    "caseid", "segment_type", "segment_duration",
    "lab_time", "time_to_lab", "signal_quality",
    "k_status", "k_severity", "k_clinical_severity",  # labels/aux
    "high_risk", "arrhythmia_risk",                   # potential aux labels
    "potassium"                                       # target (regression)
}


def potassium_to_severity(k: float) -> str:
    """Map serum potassium to discrete clinical severity (consistent with Step-1 thresholds)."""
    if pd.isna(k):
        return ""
    if k < 3.0:
        return "severe_hypo"
    elif k < 3.5:
        return "moderate_hypo"
    elif k <= 5.0:
        return "normal"
    elif k <= 5.5:
        return "mild_hyper"
    else:
        return "severe_hyper"


# ------------------------------ Core Enhancer ------------------------------

@dataclass
class VitalV2DataEnhancer:
    schema_path: str = "feature_schema.json"
    enhancement_log: List[str] = field(default_factory=list)
    feature_cols: Optional[List[str]] = None
    feature_medians: Optional[Dict[str, float]] = None

    # ------------------------ Load + Schema Enforcement ------------------------

    def load_prepared_dataset(
        self,
        csv_path: str = "vital_prepared_dataset.csv",
        enforce_schema: bool = True
    ) -> pd.DataFrame:
        """Load Step-1 output and (optionally) enforce schema to prevent drift."""
        self._log(f"Loading prepared dataset: {csv_path}")
        df = pd.read_csv(csv_path)

        if enforce_schema:
            self._load_schema_if_available()
            if self.feature_cols:
                # Ensure presence & order
                for col in self.feature_cols:
                    if col not in df.columns:
                        df[col] = np.nan
                # Fill straggler NaNs using schema medians (if provided)
                if self.feature_medians:
                    med = {k: v for k, v in self.feature_medians.items() if k in df.columns}
                    if med:
                        df[list(med.keys())] = df[list(med.keys())].fillna(med)
                # Keep only schema features + essential targets/ids if present
                keep = set(self.feature_cols) | {
                    "caseid", "potassium", "k_status", "signal_quality",
                    "segment_type", "segment_duration", "lab_time", "time_to_lab"
                }
                ordered_cols = [c for c in self.feature_cols if c in df.columns]
                remainder = [c for c in df.columns if c not in self.feature_cols and c in keep]
                df = df[ordered_cols + remainder].copy()
                self._log(f"Schema enforced with {len(ordered_cols)} features.")

        # DO NOT re-filter by signal quality here — Step-1 already did that.
        return df

    def _load_schema_if_available(self) -> None:
        """Load feature schema (feature order + medians) if present."""
        try:
            with open(self.schema_path, "r") as f:
                schema = json.load(f)
            self.feature_cols = schema.get("feature_cols")
            self.feature_medians = schema.get("feature_medians", {})
            self._log(f"Loaded schema: {len(self.feature_cols or [])} features, "
                      f"{len(self.feature_medians or {})} medians.")
        except FileNotFoundError:
            self._log("Schema file not found — proceeding without strict schema.")
        except Exception as e:
            warnings.warn(f"Failed to read schema: {e}", RuntimeWarning)

    # ----------------------------- Target Creation -----------------------------

    def add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create discrete severity labels from 'potassium' without modifying potassium."""
        if "potassium" not in df.columns:
            raise ValueError("Column 'potassium' is required to create targets.")

        self._log("Creating 'k_clinical_severity' from 'potassium'...")
        df = df.copy()
        df["k_clinical_severity"] = df["potassium"].apply(potassium_to_severity)
        return df

    # --------------------------- Feature Matrix Builder ------------------------

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        exclude_cols: Optional[set] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select model features (no scaling here)."""
        exclude = set(exclude_cols) if exclude_cols is not None else set(EXCLUDE_COLS_DEFAULT)

        # If a schema is present, prefer it strictly
        if self.feature_cols:
            feat_cols = [c for c in self.feature_cols if c in df.columns]
        else:
            feat_cols = [c for c in df.columns if c not in exclude]

        X = df[feat_cols].copy()

        # Sanitize any residual inf/NaN with medians (schema medians preferred)
        num_cols = X.select_dtypes(include=[np.number]).columns
        medians = {}
        if self.feature_medians:
            medians = {k: v for k, v in self.feature_medians.items() if k in num_cols}
        # Fall back to dataset medians for any numeric columns missing in schema
        if set(num_cols) - set(medians.keys()):
            fallback = X[num_cols].median(numeric_only=True).to_dict()
            medians.update({k: v for k, v in fallback.items() if k not in medians})

        if len(num_cols):
            X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
            X[num_cols] = X[num_cols].fillna(medians)

        self._log(f"Built feature matrix with {X.shape[1]} features and {X.shape[0]} rows.")
        return X, feat_cols

    # ---------------------- Optional Classification Resampling ----------------------

    def resample_classification_only(
        self,
        X: pd.DataFrame,
        y_cls: pd.Series,
        k_neighbors: int = 3,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to the classification target ONLY.
        The regression target (potassium) must remain on the ORIGINAL dataset.

        Returns a NEW X_cls_resampled, y_cls_resampled for classification models.
        """
        if not _HAS_IMBLEARN:
            self._log("imblearn not available — skipping resampling.")
            return X.copy(), y_cls.copy()

        # Encode classes for SMOTE if necessary
        y = y_cls.astype("category")
        if y.cat.categories.size <= 1:
            self._log("Only one class present — skipping SMOTE.")
            return X.copy(), y_cls.copy()

        smote = SMOTE(k_neighbors=min(k_neighbors, y.cat.categories.size - 1),
                      random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)
        self._log(f"SMOTE done: {len(X)} → {len(X_res)} samples.")
        return X_res, pd.Series(y_res, name=y_cls.name)

    # ------------------------------- Orchestration -------------------------------

    def run_complete_enhancement(
        self,
        csv_path: str = "vital_prepared_dataset.csv",
        schema_path: str = "feature_schema.json",
        do_resample_classification: bool = False,
        save_outputs: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict]:
        """
        Main entry:
          - Load & schema-lock prepared dataset
          - Create targets
          - Build feature matrix (unscaled; no leakage)
          - (Optional) create a separate, SMOTE-resampled classification set and save it
          - Return unified X, y_dict (original shapes), and dataset_info
        """
        self.schema_path = schema_path  # update if a custom path was passed
        df = self.load_prepared_dataset(csv_path, enforce_schema=True)
        df = self.add_targets(df)

        # Build features from df (no scaling)
        X, feat_cols = self.build_feature_matrix(df)

        # Targets
        y_dict: Dict[str, pd.Series] = {}
        if "k_clinical_severity" in df.columns:
            y_dict["k_clinical_severity"] = df["k_clinical_severity"].copy()
        if "potassium" in df.columns:
            y_dict["potassium"] = df["potassium"].copy()
        if "high_risk" in df.columns:
            y_dict["high_risk"] = df["high_risk"].copy()
        if "arrhythmia_risk" in df.columns:
            y_dict["arrhythmia_risk"] = df["arrhythmia_risk"].copy()

        # Optional: create a separate resampled dataset ONLY for classification
        if do_resample_classification and "k_clinical_severity" in y_dict:
            X_cls_res, y_cls_res = self.resample_classification_only(
                X=X,
                y_cls=y_dict["k_clinical_severity"]
            )
            if save_outputs:
                X_cls_res.to_csv("vital_cls_resampled_features.csv", index=False)
                y_cls_res.to_frame().to_csv("vital_cls_resampled_labels.csv", index=False)
                self._log("Saved classification-only resampled datasets: "
                          "vital_cls_resampled_features.csv, vital_cls_resampled_labels.csv")

        # Persist unified (original, not-resampled) dataset if requested
        if save_outputs:
            X.to_csv("vital_enhanced_features.csv", index=False)
            pd.DataFrame(y_dict).to_csv("vital_enhanced_targets.csv", index=False)
            meta = {
                "n_samples": len(X),
                "n_features": len(feat_cols),
                "feature_columns": feat_cols,
                "target_columns": list(y_dict.keys()),
                "class_distribution": (
                    y_dict["k_clinical_severity"].value_counts().to_dict()
                    if "k_clinical_severity" in y_dict else {}
                ),
                "notes": [
                    "No scaling applied here — scale inside CV/pipeline.",
                    "Regression target (potassium) remains original (no resampling).",
                    "If requested, a separate classification-only SMOTE set is saved."
                ]
            }
            with open("vital_enhanced_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            self._log("Saved vital_enhanced_features.csv, vital_enhanced_targets.csv, vital_enhanced_meta.json")

        dataset_info = {
            "n_samples": len(X),
            "n_features": len(feat_cols),
            "feature_columns": feat_cols,
            "target_columns": list(y_dict.keys()),
            "enhancement_log": self.enhancement_log
        }
        return X, y_dict, dataset_info

    # ------------------------------- Utilities -----------------------------------

    def _log(self, msg: str) -> None:
        print(msg)
        self.enhancement_log.append(msg)


# --------------------------------- CLI usage ---------------------------------

def main():
    enhancer = VitalV2DataEnhancer()
    X, y_dict, info = enhancer.run_complete_enhancement(
        csv_path="vital_prepared_dataset.csv",
        schema_path="feature_schema.json",
        do_resample_classification=False,   # set True to also emit a separate SMOTE set for classification
        save_outputs=True
    )

    # Pretty print summary
    print("\n=== ENHANCEMENT SUMMARY ===")
    print(f"Samples: {info['n_samples']} | Features: {info['n_features']}")
    print(f"Targets: {info['target_columns']}")
    if "k_clinical_severity" in y_dict:
        print("Class distribution:", y_dict["k_clinical_severity"].value_counts().to_dict())
    print("\nLog:")
    for line in info["enhancement_log"]:
        print(" -", line)


if __name__ == "__main__":
    main()
