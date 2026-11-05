from kfp.dsl import Dataset, Input, Metrics, Output, component


@component(
    base_image="cicirello/pyaction:3.11",
    packages_to_install=[
        "ragas==0.2.15",  # from uv pip list
        "langchain-community==0.3.31",  # from uv pip list
        "datasets==3.6.0",  # from uv pip list
        "pandas==2.2.3",
        "rouge-score==0.1.2",
        "sacrebleu==2.6.0",
        "rapidfuzz==3.14.6",
    ],
)
def evaluation_component(
    predictions: Input[Dataset],
    evaluation_results: Output[Dataset],
    metrics: Output[Metrics],
):
    import logging

    import pandas as pd
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, ExactMatch, NonLLMStringSimilarity, RougeScore

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    metric_definitions = [
        RougeScore(name="rouge_l", rouge_type="rougeL"),
        BleuScore(),
        NonLLMStringSimilarity(),
        ExactMatch(),
    ]

    def compute_metrics(user_input, response, reference):
        sample = SingleTurnSample(user_input=user_input, response=response, reference=reference)
        return {metric.name: metric.single_turn_score(sample) for metric in metric_definitions}

    predictions_df = pd.read_csv(predictions.path)
    # An empty model answer is read as NaN by pandas, which ragas cannot score
    predictions_df["extracted_response"] = predictions_df["extracted_response"].fillna("").astype(str)
    logger.info("Evaluating %d predictions", len(predictions_df))

    per_sample_metrics = pd.DataFrame([
        {
            **row,
            **compute_metrics(row["user_input"], row["extracted_response"], row["reference"]),
        }
        for row in predictions_df.to_dict("records")
    ])
    per_sample_metrics.to_csv(evaluation_results.path, index=False)

    for metric in metric_definitions:
        value = float(per_sample_metrics[metric.name].mean())
        metrics.log_metric(f"{metric.name}_mean", value)
        logger.info("%s_mean: %.4f", metric.name, value)