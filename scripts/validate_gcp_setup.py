from google.cloud import aiplatform, storage

from agentic_workflows.constants import BUCKET_NAME, PROJECT_ID, REGION

print("--- VERIFICATION DE LA CONFIGURATION GCP ---")
print(f"Projet GCP : {PROJECT_ID}")
print(f"Region     : {REGION}")
print(f"Bucket GCS : {BUCKET_NAME}")
print()

try:
    aiplatform.init(project=PROJECT_ID, location=REGION)
    print("Client Vertex AI initialise avec succes !")
except Exception as e:
    print(f"Erreur Vertex AI : {e}")

try:
    client = storage.Client(project=PROJECT_ID)
    blobs = list(client.list_blobs(BUCKET_NAME, max_results=10))
    print(f"Acces au bucket OK, {len(blobs)} fichier(s) affiche(s) :")
    for blob in blobs:
        print(f"  - {blob.name}")
except Exception as e:
    print(f"Erreur acces bucket GCS : {e}")