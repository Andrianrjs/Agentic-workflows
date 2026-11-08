from agentic_workflows.pipeline_components.inference_component import (
    inference_component,
)

from kfp import dsl

from agentic_workflows.pipeline_components.data_transformation_component import (
    data_transformation_component,
)
from agentic_workflows.pipeline_components.fine_tuning_component import (
    fine_tuning_component,
)
from agentic_workflows.pipeline_components.evaluation_component import (
    evaluation_component,
)

@dsl.pipeline(name="rhyming-model-training-pipeline")
def model_training_pipeline(raw_data_uri: str, test_size: float = 0.2):
    data_task = data_transformation_component(raw_data_uri=raw_data_uri, test_size=test_size)

    fine_tuning_task = fine_tuning_component(
        train_dataset=data_task.outputs["train_dataset_path"],
    )
    fine_tuning_task.set_accelerator_type("NVIDIA_TESLA_T4")
    fine_tuning_task.set_accelerator_limit(1)
    fine_tuning_task.set_cpu_limit("16")
    fine_tuning_task.set_memory_limit("50G")

    inference_task = inference_component(
    model=fine_tuning_task.outputs["model"],
    test_dataset=data_task.outputs["test_dataset_path"],
    )
    inference_task.set_accelerator_type("NVIDIA_TESLA_T4")
    inference_task.set_accelerator_limit(1)
    inference_task.set_cpu_limit("8")
    inference_task.set_memory_limit("30G")
    
    evaluation_component(predictions=inference_task.outputs["predictions"])
