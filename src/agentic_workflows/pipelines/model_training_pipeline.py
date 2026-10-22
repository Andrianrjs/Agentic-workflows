from kfp import dsl

from agentic_workflows.pipeline_components.data_transformation_component import (
    data_transformation_component,
)


@dsl.pipeline(name="yoda-model-training-pipeline")
def model_training_pipeline(raw_data_uri: str, test_size: float = 0.2):
    data_transformation_component(raw_data_uri=raw_data_uri, test_size=test_size)