# Agentic Workflows

Team repository for the **Agentic Workflows / LLMOps** course (Albert School, MSc2).

Fine-tuning Phi-3-mini (LoRA) on Vertex AI Pipelines to answer questions with a 4-line AABB rhyming poem. The pipeline was first built on the course's Yoda example, then reused on our own dataset (`rhyming_qa_700.csv`).


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


## Results

| Run | Data | Epochs | train_loss | eval_loss | ROUGE-L | BLEU | Exact match | Rhyme score |
|---|---|---|---|---|---|---|---|---|
| 2026-11-05 18:02 | Yoda (course example) | 10 | 0.90 | 2.90 | 0.82 | 0.67 | 0.22 | — |
| 2026-11-08 19:15 | Rhyming Q&A, 700 pairs | 5 | 0.79 | 1.32 | 0.31 | 0.11 | 0.00 | 0.62 |

- **Yoda run:** clear overfitting (eval_loss went back up after epoch ~3, see TensorBoard). Fixed by keeping the best checkpoint (`load_best_model_at_end`) and reducing to 5 epochs.
- **Rhyming run:** the rhyme score went from 0.25 (Colab test, 100 pairs) to 0.62 with 700 pairs, so more data clearly helped.
- ROUGE/BLEU/exact match are low on purpose: the model writes its own poem instead of copying the reference, so word overlap is not a good measure for this task.
- Not measured yet: whether the poems are factually correct (planned with an LLM judge in session 5).

### Sample outputs (rhyming run, 2026-11-08)

Three poems from the test set, read by hand to check rhymes and facts.

**"What did Abraham Lincoln do?"** · rhyme 1.0 · ROUGE-L 0.64

> Abraham Lincoln was the sixteenth president of the United States,
> Who led the country through the Civil War's fates;
> He freed the slaves with the Emancipation Proclamation's might,
> And gave the Gettysburg Address in a short, powerful flight.

Facts are correct. Some words are only there to rhyme ("the Civil War's fates").

**"Is Bitcoin real money?"** · rhyme 0.5 · ROUGE-L 0.25

> Bitcoin is a digital currency without a bank,
> People can send it without a paper or a rank;
> It is not controlled by any government's hand,
> But its value can change, as the market demands.

Facts are correct, but line 2 is filler and the poem doesn't clearly answer the question. "hand / demands" rhymes to the ear but scores 0 because of the final "s": the rhyme score is strict.

**"Why do museums exist?"** · rhyme 1.0 · ROUGE-L 0.22

> Museums collect and display objects of art and history,
> From ancient statues to paintings of great mystery;
> They help us learn about people and places far away,
> And protect them so future generations can see them one day.

The best of the three: correct rhymes, and it actually answers "why". Yet it has the lowest ROUGE-L, which shows that word overlap with the reference is not a good measure of quality here.

**Takeaways:** all three are factually correct; the main weakness is filler words chosen for the rhyme; the rhyme score slightly underestimates the model.


## Model Registry

| Model | Version | Resource name | Trained by run |
|---|---|---|---|
| rhyming-phi3 | 1 | `projects/123905721265/locations/europe-west2/models/2406050299945943040` | rhyming-model-training-pipeline-20261108191541 |
