# Session 2 - Practice

Today's goal is to build and run your first Kubeflow Pipeline!

It will contain a single step that:
- Reads the raw Yoda sentences dataset from GCS
- Formats the dataset to the expected input format for Phi-3 fine-tuning by applying the chat template
- Split the dataset into a training and a test set
- Write the resulting datasets to GCS

To do this, let's:
1. Create a Python function that implements the previous steps (and run it locally)
2. Turn this function into a Kubeflow Pipeline component, and add it to a Kubeflow Pipeline
3. Create and launch a script that compiles and submits the pipeline to Vertex AI

Here's the suggested structure of your repository:
```
├── README.md
├── pyproject.toml
├── .venv/  # created by uv
├── .gitignore
├── .env.example
├── .env  # not committed to git
├── src/
│   ├── constants.py  # constants, loads from environment variables
│   ├── pipeline_components/
│   │   └── data_transformation_component.py  # the Kubeflow Pipeline component
│   ├── pipelines/
│   │   └── model_training_pipeline.py  # the Kubeflow Pipeline
├── scripts/
│   ├── validate_gcp_setup.py  # from session 1
│   └── pipeline_runner.py  # compiles and submits the pipeline job
```

Good luck!

## 1. Data processing function

Both data processing and fine-tuning of Phi-3 are heavily inspired by this [Hugging Face article](https://huggingface.co/blog/dvgodoy/fine-tuning-llm-hugging-face).

1. Load the dataset from GCS using pandas.
    - Pandas can read CSV files directly from GCS using a `gs://` URI.
2. Convert the pandas DataFrame to a Hugging Face `datasets.Dataset` object using `datasets.Dataset.from_pandas()`.
3. Based on the article contents, format the dataset to the expected input format for `SFTTrainer` for Phi-3 fine-tuning by converting it to a conversational format, for example:
```
[
    {
        'role': 'user',
        'content': 'The birch canoe slid on the smooth planks.'
    },
    {
        'role': 'assistant',
        'content': 'On the smooth planks, the birch canoe slid. Yes, hrrrm.'
    }
]
```
4. Split the dataset into a training and a test set using `train_test_split()` class method of a Hugging Face `datasets.Dataset` object.
5. Write the resulting datasets to CSV files locally at first to check the result.
6. [Bonus] Add a unit test!

## 2. Creating a pipeline

1. Wrap the previous code in a Python function, and declare it as a component using the `@component` decorator from `kfp.dsl`.
    - Use the `OutputPath` type hint to declare the output datasets.
    - Are there any other inputs to the function you should add?
2. Specify the base image of the component and the packages to install using the `@component` decorator arguments.
    - You can take a look at the base images maintained by Python [here](https://hub.docker.com/_/python). These images are made for Python workflows!
3. Make sure to add relevant logging to the component for easy debugging.
4. Create a Kubeflow Pipeline that contains the single data processing component.
    - Does it take any input parameters?


## 3. Creating a pipeline runner script
1. Create a script that compiles and submits the pipeline to Vertex AI using the Vertex AI `aiplatform.PipelineJob` class.
    - Make sure you are using the right project and region (europe-west2).
    - Make sure to set the `pipeline_root` argument to a GCS path in your bucket.
2. Follow the link in the output of the script to see the pipeline running in the GCP console. 
    - Can you see the artifacts produced by the pipeline step?
    - If it fails, check the logs in the GCP console to debug the issue.
3. Once the pipeline has run successfully, check that the output datasets are in the expected format in your GCS bucket.


