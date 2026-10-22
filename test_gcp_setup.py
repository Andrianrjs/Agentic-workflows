import os
from dotenv import load_dotenv
from google.cloud import aiplatform, storage

# Charger les variables du fichier .env
load_dotenv()

PROJECT_ID = os.getenv("agentics-workflows-510809")
REGION = os.getenv("europe-west1")
BUCKET_NAME = os.getenv("clem-andrian-llmops")

print("--- VERIFICATION DE LA CONFIGURATION GCP ---")
print(f"Projet GCP : {PROJECT_ID}")
print(f"Region     : {REGION}")
print(f"Bucket GCS : {BUCKET_NAME}\n")

# 1. Test d'initialisation du client Vertex AI
try:
    aiplatform.init(project=PROJECT_ID, location=REGION)
    print("Client Vertex AI initialise avec succes !")
except Exception as e:
    print(f"Erreur Vertex AI : {e}")

# 2. Test d'acces au bucket Google Cloud Storage
try:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs())
    print(f"Bucket GCS '{BUCKET_NAME}' accessible avec succes !")
    print("Fichiers trouves dans le bucket :")
    for blob in blobs:
        print(f"  - {blob.name}")
except Exception as e:
    print(f"Erreur acces bucket GCS : {e}")