# Agentic Workflows

Team repository for the **Agentic Workflows / LLMOps** course (Albert School, MSc2).

## Prerequisites

- [Git](https://github.com/git-guides/install-git)
- [Docker](https://docs.docker.com/desktop/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)

## Installation

```bash
git clone https://github.com/Andrianrjs/Agentic-workflows.git
cd Agentic-workflows
uv sync
```

`uv sync` creates the `.venv` folder (Python 3.11) with the dependencies:
`google-cloud-aiplatform`, `google-cloud-bigquery`, `google-cloud-storage`, `python-dotenv`.

Then configure your IDE to use `.venv/bin/python` (`which python` should point to it).

## GCP setup

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud auth application-default login
```


```bash
cp .env.example .env
```

| Variable          | Description                          |
|-------------------|--------------------------------------|
| `GCP_PROJECT_ID`  | agentics-workflows-510809                 |
| `GCP_REGION`      | GCP region (e.g. `europe-west2`)     |
| `GCP_BUCKET_NAME` | GCS bucket (`clem-andrian-llmops`)   |

The `.env` file is ignored by git: never commit it.

## Check the setup

```bash
uv run python test_gcp_setup.py
```

The script initializes Vertex AI and lists the contents of the GCS bucket.

## Structure

```
├── src/agentic_workflows/   # Project code
├── test_gcp_setup.py        # GCP connection test
├── session*_practice.md     # Practice instructions
├── pyproject.toml / uv.lock # Dependencies
└── .env.example             # Environment variables template
```
