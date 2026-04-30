"""Model-Evaluation pipeline definition"""

from kedro.pipeline import node, pipeline
from .nodes import (
    evaluate_model,
    make_predictions,
    explain_model,
    apply_decision_threshold,
    evaluate_fairness,
)

def create_pipeline(**kwargs):
    return pipeline([
        node(
            func=make_predictions,
            inputs=["xgb_model", "X_test"],
            outputs="predictions",
            name="predict_node",
            tags=["model_evaluation"],
        ),
        node(
            func=apply_decision_threshold,
            inputs=[
                "predictions",
                "params:clinical_thresholds.high_risk_threshold",
                "params:clinical_thresholds.mid_risk_threshold",
            ],
            outputs="clinical_predictions",
            name="decision_node",
            tags=["model_evaluation"],
        ),
        node(
            func=evaluate_model,
            inputs=["clinical_predictions", "y_test"],
            outputs="metrics",
            name="evaluate_node",
            tags=["model_evaluation"],
        ),
        node(
            func=explain_model,
            inputs=["xgb_model", "X_test"],
            outputs="shap_importance",
            name="explain_node",
            tags=["model_evaluation"],
        ),
        node(
            func=evaluate_fairness,
            inputs=["clinical_predictions", "y_test", "X_test"],
            outputs="fairness_by_age",
            name="fairness_node",
            tags=["model_evaluation"],
        ),
    ])