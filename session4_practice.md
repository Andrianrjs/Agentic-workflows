# Session 4 - Practice

Today's goal is to register a fine-tuned model that was created from a run of the previous session's Kubeflow Pipeline, deploy it to Vertex AI for online predictions, and create a simple web app to interact with it.

As mentioned in the theory, since we'll be using a [pre-built Hugging Face Docker container for Vertex AI](https://github.com/huggingface/Google-Cloud-Containers), we need to create a custom `EndpointHandler` object that loads the model performs inference in accordance with the API constraints of Vertex AI. 

So, if we break down the main steps:
1. Create a custom `EndpointHandler` class that loads the fine-tuned model and performs inference, and test it locally.
2. Create a script that, for a given Model Artifact URI in GCS, uploads the `EndpointHandler` script in a `handler.py` file (expected by the pre-built container), and registers the model to Vertex AI Model Registry.
3. Deploy the registered model to a Vertex AI Endpoint (using the GCP console directly for simplicity).
4. Create a simple web app using [Chainlit](https://chainlit.io/) that calls through HTTP the deployed model endpoint for predictions.


The suggested structure of your repository is, at the end of the session:

```
├── README.md
├── pyproject.toml
├── .venv/ 
├── .gitignore
├── .env.example
├── .env 
├── src/
│   ├── app/
│   │   └── main.py  # the Chainlit web app file
│   ├── constants.py  
│   ├── handler.py  # file with the custom EndpointHandler
│   ├── pipeline_components/
│   │   ├── fine_tuning_component.py 
│   │   ├── inference_component.py 
│   │   ├── evaluation_component.py  
│   │   └── data_transformation_component.py  
│   ├── pipelines/
│   │   └── model_training_pipeline.py
├── scripts/
│   ├── validate_gcp_setup.py 
│   ├── register_model_with_custom_handler.py  # a new script!
│   └── pipeline_runner.py  
```

# 1. Creating the custom EndpointHandler

In this section, the goal is to write the `EndpointHandler` class that will be used by the pre-built Hugging Face container to load the fine-tuned model and perform inference.

To be sure that it works, we'll test it locally first: this means that we'll set some values that are compatible with your local environment, that will need to be changed in the next section!

1. Create a new file `src/handler.py`, and define an `EndpointHandler` class with an `__init__` that accepts a `model_dir` parameter as input. 
    ```python
    class EndpointHandler:
        """Handler for processing inference requests using a Hugging Face model."""

        def __init__(self, model_dir: str = MODEL_DIR) -> None:
            """Load tokenizer and model from the specified directory."""
            # TODO: Load the tokenizer and model from the model_dir
            pass
    ```
    Complete the method by assigning to the `EndpointHandler` instance a tokenizer (loaded with `AutoTokenizer.from_pretrained`) and a model (loaded with `AutoModelForCausalLM.from_pretrained`).
    - Make sure to set the `device_map` parameter to the appropriate value given your local set up.
    - Download the model artifacts from GCS to a local directory, and pass that local directory path as `model_dir`.
    - Don't forget to set your model to `eval()` mode after loading it!
2. In a notebook, test that you can correctly instantiate the `EndpointHandler` class.
3. Create a `__call__` method that accepts a dictionary corresponding to the Vertex AI prediction request format, and that ouputs a dictionary corresponding to the Vertex AI prediction response format.
As a reminder, the input format looks like:
    ```json
    // Input format
    {
    "instances": [
        {
        // Should contain the user's message, already with chat template applied.
        // You can structure this as you want!
        }
    ], 
    "parameters": {
        // Should contain generation parameters (topk, temperature, ...)
        // You can structure this as you want!
    }
    }
    ```
    And the output format looks like:
    ```json
    // Output format
    {
    "predictions": [
        {
        // Should contain the model's response
        // You can also structure this as you want!
        }
    ]
    }
    ```
    This method should be similar to the `inference_component.py` code, it should:
    - Tokenize the input message(s) from the `instances` list.
    - Generate the output(s) using the model's `generate` method, passing the generation parameters from the `parameters` dictionary.
    - Decode the generated output(s) back to text.
    - Structure the output in the expected format and return it.
4. Test the `__call__` method in a notebook by creating a sample input dictionary and calling the method with it. Is the format correct?

# 2. Creating the model registration script

In this section, the goal is to create a `script/register_model_with_custom_handler.py` script that:
1. Uploads the previous `handler.py` file to the GCS bucket that contains the model artifacts logged by the Kubeflow Pipeline.
2. Registers the fine-tuned model to Vertex AI Model Registry.

## Uploading the handler.py file to GCS

1. In a notebook, test that given a model artifact URI in GCS (for example, `gs://your-bucket/your-vertexai-pipeline-root/path/to/model/`), you can upload the `handler.py` file stored locally to that location.

## Registering the model to Vertex AI Model Registry

1. In the same notebook, test that you can register a model to Vertex AI Model Registry using the `aiplatform.Model.upload` method.
- Set the `artifact_uri` parameter to the model artifact directory in GCS (where you just uploaded the `handler.py` file).
- Set as `serving_container_image_uri` parameter the URI of the pre-built Hugging Face container: `us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cu121.2-3.transformers.4-46.ubuntu2204.py311`.
- Open the container p ort 8080 by passing the following parameter: `serving_container_ports=[8080]`.
    - This is the port that Vertex AI will use to send prediction requests to the container, which is expected by the pre-built container!
- Set other parameters as needed (model name, description, etc.).
2. Can you see the registered model in the Vertex AI console? Can you find the lineage (which pipeline run was used to produce the registered model)?
3. Try registering a second version of the same model by passing the `parent_model` parameter to the `aiplatform.Model.upload` method. Can you see the new version in the console?

## Wrapping both steps in a script
To make sure that we always register the model with the correct handler, we can use a script to automate both previous steps.

1. Create a new script `scripts/register_model_with_custom_handler.py` that accepts at least as input parameters the model URI, and the model name to register.
2. In the script, implement the two previous steps.
3. Test the script from the terminal, can you register a model with it, and does that model have the `handler.py` file?

# 3. Deploying the model to a Vertex AI Endpoint

## Creating the Endpoint

> ⚠️ Deploying a model to an Endpoint in Vertex AI takes between 15-30 minutes! You can start this step while working on the Chainlit app in the next section.

1. In the Vertex AI Model Registry console, go to the three dots at the end of your model and select "Deploy to Endpoint".
2. Create a new Endpoint, and:
- Set the minimum and maximum number of nodes (1 and 1 are fine for testing, no autoscaling).
- Select the machine type (I recommend n1-standard-8 so that the model loading time is reasonable).
- Add the T4 GPU accelerator authorized by your quota increase request.
- No need for additional configuration, or the activation of Model Monitoring for this practice!
3. The deployment process should start, you can take a regular look at the logs (click the three dots on the Endpoint page, then "View logs") to see the progress.

## Testing the endpoint

If you don't have an API testing tool like Postman installed, you can use `curl` from the terminal to test the endpoint once the deployment is complete.

1. Create a sample input JSON file to send to your API!
2. Add the following variables to your `.env` file (and update your `.env.example` accordingly):
    ```env
    GCP_PROJECT_NUMBER=your-project-number
    GCP_ENDPOINT_ID=your-endpoint-id
    INPUT_DATA_FILE=path/to/your/input/data.json
    ```
    - Your project number can be found in the GCP project settings page.
    - Your endpoint ID can be found in the Endpoint details page in Vertex AI console.
3. Once the deployment is complete, you can test it with the following `curl` command:
```bash
curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
"https://europe-west2-aiplatform.googleapis.com/v1/projects/${GCP_VERTEX_PROJECT_ID}/locations/europe-west2/endpoints/${GCP_ENDPOINT_ID}:predict" \
-d "@${INPUT_DATA_FILE}"
```
4. Do you get a valid response from the model? Does it sound like Yoda?

> ### ⚠️ If you have successfully created an endpoint, be sure to delete it at the end of your practice to avoid unnecessary costs!

# 4. Creating a simple Chainlit web app

In this section, the goal is to create a simple web app using [Chainlit](https://chainlit.io/) that interacts with the deployed model endpoint for predictions.

## Hello world application

1. In the `src/app/main.py` file, create a Chainlit app that, on a user message, simply sends back the user's message.
2. Run the application locally - does it work?

## Posting to the API from Python

By default, the endpoint is protected - you need to authenticate to be able to get your prediction!

To simplify, we can fetch from Python an access token linked to your user account, and then pass it to the `requests` library to authenticate.

> Note that in production, this wouldn't be possible: you don't have a user to click "Allow" on the OAuth2 consent screen! You would need to use a service account with the appropriate permissions instead.

1. In a notebook, test that you can get access a token by calling the `gcloud auth print-access-token` command from Python (using the `subprocess` library).
    ```python
    access_token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()
    ```
    - Under the hood, `gcloud` is using your environment variables to find the correct project and region.
2. Build the endpoint URL from your constants / environment variables.
    - See the previous section's `curl` command for reference.
3. Using the `requests` library, send a POST request to the endpoint URL with:
    - An `Authorization` header with the value `f"Bearer {access_token}"` and a `Content-Type` header with the value `application/json`.
    - A dictionary body with a sample input message (similar to the one you used to test the endpoint with `curl`).
4. Get the response status code and JSON payload. Is it in the correct format?

## Wrapping it all together

1. Create a function that, given a Chainlit `cl.Message` object from the user, applies the chat template to it, and sends a prediction request to the deployed endpoint. It should return the model's response as a string.
    - You can re-use the model's response extraction code from the `inference_component.py` file!
    - What inference parameters are you going to set? 
2. In the function wrapped by the `@cl.on_message` decorator, call the previous function to get the model's response, and send it back to the user using `cl.Message` (instead of just echoing the user's message).
3. Run the Chainlit app locally, can you chat with your fine-tuned model through the web interface?