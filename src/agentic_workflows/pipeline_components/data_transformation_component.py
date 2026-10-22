from kfp.dsl import component, OutputPath


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "datasets", "gcsfs"],
)
def data_transformation_component(
    raw_data_uri: str,
    train_dataset_path: OutputPath("Dataset"), # type: ignore
    test_dataset_path: OutputPath("Dataset"), # type: ignore
    test_size: float = 0.2,
    seed: int = 42,
):
    import json
    import logging

    import pandas as pd
    from datasets import Dataset

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    df = pd.read_csv(raw_data_uri)
    logger.info("Loaded %d rows with columns %s", len(df), list(df.columns))

    dataset = Dataset.from_pandas(df)

    def to_messages(example):
        return {
            "messages": [
                {"role": "user", "content": example["sentence"]},
                {"role": "assistant", "content": example["translation_extra"]},
            ]
        }

    dataset = dataset.map(to_messages, remove_columns=dataset.column_names)
    splits = dataset.train_test_split(test_size=test_size, seed=seed)
    logger.info("Train: %d rows, test: %d rows", len(splits["train"]), len(splits["test"]))

    for split, path in [("train", train_dataset_path), ("test", test_dataset_path)]:
        # A CSV can't store a list of dicts, so each conversation is saved as JSON text
        rows = [json.dumps(m) for m in splits[split]["messages"]]
        pd.DataFrame({"messages": rows}).to_csv(path, index=False)
        logger.info("Wrote %s split to %s", split, path)
