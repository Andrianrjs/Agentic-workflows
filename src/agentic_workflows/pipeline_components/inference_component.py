from kfp.dsl import Dataset, Input, Model, Output, component


@component(
    base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel",
    packages_to_install=[
        "transformers==4.46.3",
        "peft==0.13.2",
        "accelerate==1.1.1",
        "pandas==2.2.3",
    ],
)
def inference_component(
    model: Input[Model],
    test_dataset: Input[Dataset],
    predictions: Output[Dataset],
    max_new_tokens: int = 96,
):
    import json
    import logging
    import re

    import pandas as pd
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def build_prompt(tokenizer, sentence):
        messages = [{"role": "user", "content": sentence}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_response(model, tokenizer, prompt, **kwargs):
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, eos_token_id=tokenizer.eos_token_id, **kwargs)
        return tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

    def extract_response(model_output):
        match = re.search(
            r"<\|assistant\|>(.*?)(?:<\|end\|>|<\|endoftext\|>|$)", model_output, re.DOTALL
        )
        return match.group(1).strip() if match else model_output.strip()

    # The LoRA adapter folder knows its base model: peft loads Phi-3 then adds the adapter
    tokenizer = AutoTokenizer.from_pretrained(model.path)
    fine_tuned_model = AutoPeftModelForCausalLM.from_pretrained(
        model.path, device_map={"": 0}, torch_dtype=torch.float16
    )
    fine_tuned_model.eval()
    logger.info("Loaded fine-tuned model from %s", model.path)

    test_df = pd.read_csv(test_dataset.path)
    rows = []
    for i, messages_json in enumerate(test_df["messages"]):
        messages = json.loads(messages_json)
        sentence, reference = messages[0]["content"], messages[1]["content"]
        raw_output = generate_response(
            fine_tuned_model, tokenizer, build_prompt(tokenizer, sentence),
            max_new_tokens=max_new_tokens, do_sample=False,
        )
        rows.append({
            "user_input": sentence,
            "reference": reference,
            "extracted_response": extract_response(raw_output),
        })
        if i < 3:
            logger.info("Example %d: %s -> %s", i, sentence, rows[-1]["extracted_response"])

    pd.DataFrame(rows).to_csv(predictions.path, index=False)
    logger.info("Wrote %d predictions to %s", len(rows), predictions.path)