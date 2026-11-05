# Agentic Workflows

Team repository for the **Agentic Workflows / LLMOps** course (Albert School, MSc2).

## Prerequisites

- [Git](https://github.com/git-guides/install-git)
- [Docker](https://docs.docker.com/desktop/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)

## Installation

```
git clone https://github.com/Andrianrjs/Agentic-workflows.git
cd Agentic-workflows
uv sync
```

`uv sync` creates the `.venv` folder (Python 3.11) with the project dependencies.

Then configure your IDE to use `.venv/bin/python` (`which python` should point to it).

## GCP setup

```
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud auth application-default login
```

```
cp .env.example .env
```

| Variable | Description |
| --- | --- |
| `GCP_PROJECT_ID` | agentics-workflows-510809 |
| `GCP_REGION` | GCP region (`europe-west2`) |
| `GCP_BUCKET_NAME` | GCS bucket (`clem-andrian-llmops`) |

The `.env` file is ignored by git: never commit it.

## Check the setup

```
uv run python scripts/validate_gcp_setup.py
```

The script initializes Vertex AI and lists the contents of the GCS bucket.

## Run the pipeline

Test the data transformation component locally:

```
uv run python scripts/test_component_locally.py
```

Compile and submit the pipeline to Vertex AI:

```
uv run python scripts/pipeline_runner.py
```

## Structure

```
├── src/agentic_workflows/
│   ├── constants.py              # Settings loaded from .env
│   ├── pipeline_components/      # Pipeline steps (KFP components)
│   └── pipelines/                # Pipeline definitions
├── scripts/
│   ├── validate_gcp_setup.py     # GCP connection test
│   ├── test_component_locally.py # Run a component locally
│   └── pipeline_runner.py        # Compile and submit to Vertex AI
├── session*_practice.md          # Practice instructions
├── pyproject.toml / uv.lock      # Dependencies
└── .env.example                  # Environment variables template
```