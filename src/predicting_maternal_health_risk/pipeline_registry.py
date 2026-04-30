"""Project pipelines."""

from kedro.pipeline import Pipeline

from predicting_maternal_health_risk.pipelines.data_processing import (
    create_pipeline as dp,
)
from predicting_maternal_health_risk.pipelines.model_training import (
    create_pipeline as mt,
)
from predicting_maternal_health_risk.pipelines.model_evaluation import (
    create_pipeline as me,
)

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    data_processing_pipeline   = dp()
    model_training_pipeline    = mt()
    model_evaluation_pipeline  = me()

    # Clinical = training + full evaluation stack
    clinical_pipeline = model_training_pipeline + model_evaluation_pipeline

    return {
        "__default__": (
            data_processing_pipeline
            + model_training_pipeline
            + model_evaluation_pipeline
        ),
        "dp":          data_processing_pipeline,
        "mt":          model_training_pipeline,
        "me":          model_evaluation_pipeline,
        "train_eval":  model_training_pipeline + model_evaluation_pipeline,
        "clinical": clinical_pipeline,
    }
