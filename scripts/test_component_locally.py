from agentic_workflows.constants import RAW_DATA_URI
from agentic_workflows.pipeline_components.data_transformation_component import (
    data_transformation_component,
)

data_transformation_component.python_func(
    raw_data_uri=RAW_DATA_URI,
    train_dataset_path="train.csv",
    test_dataset_path="test.csv",
)