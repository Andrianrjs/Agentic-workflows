import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "model"


class EndpointHandler:
    """Handler for processing inference requests using a Hugging Face model."""

    def __init__(self, model_dir: str = MODEL_DIR) -> None:
        """Load tokenizer and model from the specified directory."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

    def __call__(self, data: dict) -> dict:
        """Generate a poem for each prompt in a Vertex AI prediction request."""
        parameters = {"max_new_tokens": 96, "do_sample": False, **data.get("parameters", {})}
        predictions = []
        for instance in data["instances"]:
            inputs = self.tokenizer(instance["prompt"], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **parameters)
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            predictions.append({"response": response})
        return {"predictions": predictions}
