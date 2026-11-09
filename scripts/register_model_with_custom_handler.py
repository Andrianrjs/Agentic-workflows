import argparse
from pathlib import Path

from google.cloud import aiplatform, storage

from agentic_workflows.constants import PROJECT_ID, REGION

HANDLER_PATH = Path(__file__).resolve().parents[1] / "src" / "agentic_workflows" / "handler.py"
SERVING_IMAGE_URI = (
    "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/"
    "huggingface-pytorch-inference-cu121.2-3.transformers.4-46.ubuntu2204.py311"
)


def upload_handler(model_uri: str) -> None:
    """Copy handler.py into the model artifact folder in GCS."""
    bucket_name, _, prefix = model_uri.removeprefix("gs://").partition("/")
    blob = storage.Client(project=PROJECT_ID).bucket(bucket_name).blob(f"{prefix.rstrip('/')}/handler.py")
    blob.upload_from_filename(str(HANDLER_PATH))
    print(f"Uploaded handler.py to {model_uri.rstrip('/')}/handler.py")


def register_model(model_uri: str, model_name: str, parent_model: str | None = None) -> aiplatform.Model:
    """Register the model artifacts in Vertex AI Model Registry."""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    return aiplatform.Model.upload(
        display_name=model_name,
        description="Phi-3-mini fine-tuned with LoRA to answer questions with 4-line AABB rhyming poems.",
        artifact_uri=model_uri,
        serving_container_image_uri=SERVING_IMAGE_URI,
        serving_container_ports=[8080],
        parent_model=parent_model,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True, help="GCS folder of the model artifact")
    parser.add_argument("--model-name", required=True, help="Display name in Model Registry")
    parser.add_argument("--parent-model", default=None, help="Resource name of an existing model, to add a new version")
    args = parser.parse_args()

    upload_handler(args.model_uri)
    model = register_model(args.model_uri, args.model_name, args.parent_model)
    print(f"Registered: {model.resource_name} (version {model.version_id})")
