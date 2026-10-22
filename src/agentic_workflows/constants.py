import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "europe-west2")
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")

RAW_DATA_URI = f"gs://{BUCKET_NAME}/yoda_sentences.csv"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"