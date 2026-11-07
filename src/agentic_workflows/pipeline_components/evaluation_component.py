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
        "pronouncing==0.2.0",  # Pronunciation dictionary used by the rhyme metric
    ],
)
def evaluation_component(
    predictions: Input[Dataset],
    evaluation_results: Output[Dataset],
    metrics: Output[Metrics],
):
    import logging
    import re

    import pandas as pd
    import pronouncing
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

    def last_word(line):
        """Return the last word of a line, in lowercase, without punctuation."""
        words = re.findall(r"[a-z']+", line.lower())
        return words[-1] if words else ""

    def words_rhyme(first, second):
        """Check whether two words share the same ending sound."""
        if not first or not second or first == second:
            return False
        first_parts = {pronouncing.rhyming_part(p) for p in pronouncing.phones_for_word(first)}
        second_parts = {pronouncing.rhyming_part(p) for p in pronouncing.phones_for_word(second)}
        return bool(first_parts & second_parts)

    def rhyme_score(poem):
        """Return the share of correct AABB rhymes, or 0 if the poem does not have 4 lines."""
        lines = [line.strip() for line in str(poem).splitlines() if line.strip()]
        if len(lines) != 4:
            return 0.0
        words = [last_word(line) for line in lines]
        return (words_rhyme(words[0], words[1]) + words_rhyme(words[2], words[3])) / 2

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

    # Rhyme metric: ROUGE and BLEU cannot tell whether a poem rhymes
    per_sample_metrics["rhyme_score"] = per_sample_metrics["extracted_response"].apply(rhyme_score)
    rhyme_score_mean = float(per_sample_metrics["rhyme_score"].mean())
    metrics.log_metric("rhyme_score_mean", rhyme_score_mean)
    logger.info("rhyme_score_mean: %.4f", rhyme_score_mean)

    per_sample_metrics.to_csv(evaluation_results.path, index=False)

    for metric in metric_definitions:
        value = float(per_sample_metrics[metric.name].mean())
        metrics.log_metric(f"{metric.name}_mean", value)
        logger.info("%s_mean: %.4f", metric.name, value)