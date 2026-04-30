"""Model-evaluation pipeline nodes - predict and score with XGBoost model."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
import shap
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)
CLASS_NAMES = ["low risk", "mid risk", "high risk"]

def make_predictions(
        model: XGBClassifier,
        X_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run inference. Returns softmax probabilities + predicted class + label."""

    proba = model.predict_proba(X_test.values)
    preds = np.argmax(proba, axis=1)

    result = pd.DataFrame(
        proba,
        columns=[f"prob_{c.replace(' ', '_')}" for c in CLASS_NAMES],
    )

    result["predicted_class"] = preds
    result["predicted_label"] = [CLASS_NAMES[p] for p in preds]
    logger.info("Prediction distribution:\n%s", result["predicted_label"].value_counts())
    return result


def evaluate_model(
        predictions: pd.DataFrame,
        y_test: pd.DataFrame,
) -> Dict[str, float]:
    """
    Full evaluation suite:
    accuracy, macro P/R/F1, weightd F1, per-class F1,
    Cohen's Kappa, Matthws CC, macro ROC-AUC.
    """

    y_true = y_test.values.ravel()
    y_pred = predictions["predicted_class"].values
    proba = predictions [
        [f"prob_{c.replace(' ', '_')}" for c in CLASS_NAMES]
    ].values

    acc         = accuracy_score(y_true, y_pred)
    macro_p     = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    macro_r     = recall_score(y_true, y_pred,    average="macro",    zero_division=0)
    macro_f1    = f1_score(y_true, y_pred,        average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred,        average="weighted", zero_division=0)
    kappa       = cohen_kappa_score(y_true, y_pred)
    mcc         = matthews_corrcoef(y_true, y_pred)
    auc_macro   = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    logger.info("\n%s", classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4
    ))

    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_true, y_pred))

    metrics = {
        "accuracy":        round(float(acc),             4),
        "macro_precision": round(float(macro_p),         4),
        "macro_recall":    round(float(macro_r),         4),
        "macro_f1":        round(float(macro_f1),        4),
        "weighted_f1":     round(float(weighted_f1),     4),
        "cohen_kappa":     round(float(kappa),           4),
        "matthews_cc":     round(float(mcc),             4),
        "roc_auc_macro":   round(float(auc_macro),       4),
        "f1_low_risk":     round(float(per_class_f1[0]), 4),
        "f1_mid_risk":     round(float(per_class_f1[1]), 4),
        "f1_high_risk":    round(float(per_class_f1[2]), 4),
    }

    logger.info("Metrics:\%s", json.dumps(metrics, indent=2))
    return metrics

import shap

def explain_model(model, X_test: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    ## shap_values shape: (n_smaples, n_features, n_classes)
    ## Save mean absolute SHAP per feature per class
    feature_importance = pd.DataFrame(
        np.abs(shap_values).mean(axis=0),
        index=X_test.columns,
        columns = ["low_risk", "mid_risk", "high_risk"]
    )

    return feature_importance

def apply_decision_threshold(
        predictions: pd.DataFrame,
        high_risk_threshold: float,
        mid_risk_threshold: float,
) -> pd.DataFrame:
    """
    Override predicted class when model confidence is below threshold.
    Low-confidence high-risk -> escalate to clinician review.
    """
    predictions = predictions.copy()
    predictions["confidence"] = predictions[
        [f"prob_{c.replace(' ','_')}" for c in ["low risk", "mid risk", "high risk"]]].max(axis=1)
    
    predictions["clinical_action"] = predictions.apply(
        lambda r: "REFER IMMEDIATELY"   if r["predicted_label"] == "high risk"
                                            and r["confidence"] >= high_risk_threshold
        else "CLINICIAN REVIEW"          if r["predicted_label"] == "high risk"
                                            and r["confidence"] < high_risk_threshold
        else "MONITOR CLOSELY"           if r["predicted_label"] == "mid risk"
                                            and r["confidence"] >= mid_risk_threshold
        else "ROUTINE CARE",
        axis=1,
    )

    return predictions

def evaluate_fairness(
    predictions: pd.DataFrame,
    y_test: pd.DataFrame,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Compute accuracy and F1 per age group to detect bias."""
    results = []
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)

    age_bins = pd.cut(X_test["Age"], bins=[10, 20, 30, 40, 70],
                      labels=["10-20", "21-30", "31-40", "41+"])

    for group in age_bins.unique():
        idx = age_bins == group
        if idx.sum() < 5:
            continue
        acc = accuracy_score(y_test[idx], predictions["predicted_class"][idx])
        f1  = f1_score(y_test[idx], predictions["predicted_class"][idx],
                       average="macro", zero_division=0)
        results.append({"age_group": group, "accuracy": round(acc, 4), "macro_f1": round(f1, 4), "n_samples": int(idx.sum())})

    return pd.DataFrame(results)

# ── Explainability ────────────────────────────────────────────────────────────

def explain_model(model, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values per feature and class.

    Output: rows = features, columns = [low_risk, mid_risk, high_risk].
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For multi-class XGBoost, shap_values is usually a list of arrays
    if isinstance(shap_values, list):
        # list length = n_classes, each (n_samples, n_features)
        shap_arr = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values], axis=1)
    else:
        # fallback: (n_samples, n_features, n_classes)
        shap_arr = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.DataFrame(
        shap_arr,
        index=X_test.columns,
        columns=["low_risk", "mid_risk", "high_risk"],
    )
    return feature_importance

# ── Clinical decision thresholds ──────────────────────────────────────────────

def apply_decision_threshold(
    predictions: pd.DataFrame,
    high_risk_threshold: float,
    mid_risk_threshold: float,
) -> pd.DataFrame:
    """
    Add 'confidence' and 'clinical_action' based on risk thresholds.

    - High risk + high confidence -> REFER IMMEDIATELY
    - High risk + low confidence  -> CLINICIAN REVIEW
    - Mid risk + high confidence  -> MONITOR CLOSELY
    - Otherwise                   -> ROUTINE CARE
    """
    predictions = predictions.copy()
    prob_cols = [
        f"prob_{c.replace(' ', '_')}"
        for c in ["low risk", "mid risk", "high risk"]
    ]
    predictions["confidence"] = predictions[prob_cols].max(axis=1)

    def _decide(row):
        if row["predicted_label"] == "high risk":
            return (
                "REFER IMMEDIATELY"
                if row["confidence"] >= high_risk_threshold
                else "CLINICIAN REVIEW"
            )
        if row["predicted_label"] == "mid risk":
            return (
                "MONITOR CLOSELY"
                if row["confidence"] >= mid_risk_threshold
                else "ROUTINE CARE"
            )
        return "ROUTINE CARE"

    predictions["clinical_action"] = predictions.apply(_decide, axis=1)
    return predictions

# ── Fairness by age ───────────────────────────────────────────────────────────

def evaluate_fairness(
    predictions: pd.DataFrame,
    y_test: pd.DataFrame,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute accuracy and macro F1 by age group to detect performance gaps.
    """
    results = []

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)

    age_bins = pd.cut(
        X_test["Age"],
        bins=[10, 20, 30, 40, 70],
        labels=["10-20", "21-30", "31-40", "41+"],
    )

    for group in age_bins.unique():
        idx = age_bins == group
        if idx.sum() < 5:
            continue  # too few samples, skip

        acc = accuracy_score(y_test[idx], predictions["predicted_class"][idx])
        f1 = f1_score(
            y_test[idx],
            predictions["predicted_class"][idx],
            average="macro",
            zero_division=0,
        )
        results.append(
            {
                "age_group": str(group),
                "n_samples": int(idx.sum()),
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(f1), 4),
            }
        )

    return pd.DataFrame(results)