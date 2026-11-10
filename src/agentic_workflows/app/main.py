import subprocess

import chainlit as cl
import requests
from langfuse import Langfuse, observe

from agentic_workflows.constants import ENDPOINT_ID, PROJECT_NUMBER, REGION

ENDPOINT_URL = (
    f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}"
    f"/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
)
GENERATION_PARAMETERS = {"max_new_tokens": 96, "do_sample": False}

# Langfuse reads the LANGFUSE_* variables loaded from .env by constants.py.
# Chainlit's own OpenTelemetry traces are blocked so that only our traces are logged.
langfuse = Langfuse(blocked_instrumentation_scopes=["chainlit"])


def build_prompt(question: str) -> str:
    """Apply the Phi-3 chat template to the user's question."""
    return f"<|user|>\n{question}<|end|>\n<|assistant|>\n"


def get_access_token() -> str:
    """Get a short-lived access token from the logged-in gcloud account."""
    return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()


@observe(name="rhyming-poem")
def get_poem(message: cl.Message) -> str:
    """Send the user's question to the endpoint and return the generated poem."""
    langfuse.update_current_span(input=message.content)

    payload = {
        "instances": [{"prompt": build_prompt(message.content)}],
        "parameters": GENERATION_PARAMETERS,
    }
    headers = {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json",
    }

    with langfuse.start_as_current_generation(name="vertex-endpoint-call") as gen:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        raw_response = response.json()
        gen.update(
            input=payload,
            output=raw_response,
            model=raw_response.get("modelDisplayName"),
            version=raw_response.get("modelVersionId"),
            metadata={**GENERATION_PARAMETERS, "endpoint_id": ENDPOINT_ID},
        )

    poem = raw_response["predictions"][0]["response"]
    langfuse.update_current_span(output=poem)
    # The LLM-as-a-judge evaluators read {{input}} and {{output}} from the trace.
    langfuse.update_current_trace(input=message.content, output=poem)
    return poem


@cl.on_message
async def main(message: cl.Message) -> None:
    """Answer each user message with a rhyming poem from the deployed model."""
    poem = await cl.make_async(get_poem)(message)
    await cl.Message(content=poem.replace("\n", "  \n")).send()