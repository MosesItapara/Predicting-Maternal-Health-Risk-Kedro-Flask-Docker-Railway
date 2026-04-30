"""Model training pipeline definition."""

from kedro.pipeline import pipeline, node
from .nodes import train_model

def create_pipeline(**kwargs):
    """Create the model training pipeline."""
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs=["xgb_model", "training_metrics"],
                name="train_model_node",
                tags=["model_training"],
            ),
        ]
    )