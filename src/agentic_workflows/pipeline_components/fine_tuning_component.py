from kfp.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
    base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel",
    packages_to_install=[
        "transformers==4.46.3",  # from uv pip list
        "peft==0.13.2",
        "trl==0.12.2",  # from uv pip list
        "accelerate==1.15.0",  # from uv pip list
        "datasets==3.6.0",  # from uv pip list
        "pandas==2.2.3",
        "bitsandbytes==0.47.0",
        "tensorboard",
    ],
)
def fine_tuning_component(
    train_dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    model_id: str = "microsoft/Phi-3-mini-4k-instruct",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    num_train_epochs: int = 10,
    learning_rate: float = 3e-4,
    per_device_train_batch_size: int = 16,
    max_seq_length: int = 128,
    validation_size: float = 0.1,
    seed: int = 42,
):
    import json
    import logging

    import pandas as pd
    import torch
    from datasets import Dataset as HFDataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    hyperparams = {
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "max_seq_length": max_seq_length,
    }
    logger.info("Hyperparameters: %s", hyperparams)

    # Data: on Vertex AI, train_dataset.path is a local path to the GCS file
    df = pd.read_csv(train_dataset.path)
    conversations = [json.loads(m) for m in df["messages"]]
    dataset = HFDataset.from_dict({"messages": conversations})
    splits = dataset.train_test_split(test_size=validation_size, seed=seed)
    logger.info("Train: %d rows, validation: %d rows", len(splits["train"]), len(splits["test"]))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "right"

    # 4-bit quantization so that Phi-3 fits on a single T4 GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map={"": 0}, quantization_config=bnb_config
    )
    logger.info("Loaded %s, memory footprint: %.0f MB", model_id, base_model.get_memory_footprint() / 1e6)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
    )

    sft_config = SFTConfig(
        output_dir="/tmp/checkpoints",
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        max_seq_length=max_seq_length,
        packing=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=["tensorboard"],
        logging_dir=metrics.path,  # TensorBoard logs saved with the Metrics artifact
        seed=seed,
    )

    trainer = SFTTrainer(
        model=base_model,
        args=sft_config,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.model.print_trainable_parameters()

    train_result = trainer.train()
    eval_result = trainer.evaluate()
    logger.info("Train loss: %.4f, eval loss: %.4f", train_result.training_loss, eval_result["eval_loss"])

    # Saves the LoRA adapter (about 50 MB) and the tokenizer
    trainer.save_model(model.path)
    tokenizer.save_pretrained(model.path)
    model.metadata.update({"base_model": model_id, **hyperparams})

    metrics.log_metric("train_loss", train_result.training_loss)
    metrics.log_metric("eval_loss", eval_result["eval_loss"])
    for name, value in hyperparams.items():
        metrics.log_metric(name, value)