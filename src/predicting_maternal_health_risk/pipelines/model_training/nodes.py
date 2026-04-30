"""Model training pipeline nodes."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> Tuple[XGBClassifier, Dict[str, float]]:
    """
    Train the XGBoost classifier on maternal health data.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Trained model and training metrics.
    """
    # Fixed (or later: parameterised) model hyperparameters
    model_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    # Simple class-balance handling (you can refine later)
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train.values.ravel(),
    )
    logger.info("Using balanced sample weights")

    model = XGBClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        learning_rate=model_params["learning_rate"],
        subsample=model_params["subsample"],
        colsample_bytree=model_params["colsample_bytree"],
        reg_alpha=model_params["reg_alpha"],
        reg_lambda=model_params["reg_lambda"],
        objective="multi:softprob",
        num_class=3,
        eval_metric=["mlogloss", "merror"],
        n_jobs=-1,
        random_state=model_params["random_state"],
        verbosity=1,
        use_label_encoder=False,
    )

    eval_set = [(X_train.values, y_train.values.ravel())]

    model.fit(
        X_train.values,
        y_train.values.ravel(),
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=50,
    )

    # Training metrics
    results = model.evals_result()
    final_idx = -1
    train_metrics = {
        "final_train_logloss": round(results["validation_0"]["mlogloss"][final_idx], 4),
        "final_train_error": round(results["validation_0"]["merror"][final_idx], 4),
        "n_estimators": float(
            getattr(model, "best_iteration", model_params["n_estimators"] - 1) + 1
        ),
    }

    logger.info("Model training completed with metrics: %s", train_metrics)
    return model, train_metrics