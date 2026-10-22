from google.cloud import aiplatform
from kfp import compiler

from agentic_workflows.constants import (
    BUCKET_NAME,
    PIPELINE_ROOT,
    PROJECT_ID,
    RAW_DATA_URI,
    REGION,
)
from agentic_workflows.pipelines.model_training_pipeline import model_training_pipeline

PIPELINE_FILE = "model_training_pipeline.json"

compiler.Compiler().compile(model_training_pipeline, PIPELINE_FILE)

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")

job = aiplatform.PipelineJob(
    display_name="yoda-model-training-pipeline",
    template_path=PIPELINE_FILE,
    pipeline_root=PIPELINE_ROOT,
    parameter_values={"raw_data_uri": RAW_DATA_URI},
    enable_caching=False,
)
job.submit()