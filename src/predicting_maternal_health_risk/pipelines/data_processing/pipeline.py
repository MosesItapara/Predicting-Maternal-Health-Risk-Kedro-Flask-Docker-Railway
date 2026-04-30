"""Data-processing pipeline definition."""

from kedro.pipeline import node, pipeline
from .nodes import clean_data, encode_target, split_and_scale


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=clean_data,
                inputs="raw_data",
                outputs="cleaned_data",
                name="clean_node",
            ),
            node(
                func=encode_target,
                inputs=["cleaned_data", "params:risk_mapping"],
                outputs="encoded_data",
                name="encode_node",
            ),
            node(
                func=split_and_scale,
                inputs=[
                    "encoded_data",
                    "params:target_column",
                    "params:test_size",
                    "params:random_state",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test", "scaler"],
                name="split_node",
                tags=["data_processing"],
            ),
        ]
    )