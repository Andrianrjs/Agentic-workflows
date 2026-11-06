"""Count how many tokens each question/poem pair uses with the Phi-3 tokenizer."""

import sys

import pandas as pd
from transformers import AutoTokenizer

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    full_lengths = []
    poem_lengths = []
    for row in df.to_dict("records"):
        messages = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["poem"]},
        ]
        # Full pair with the chat tags: this is what max_seq_length must cover
        full_lengths.append(len(tokenizer.apply_chat_template(messages, tokenize=True)))
        # Poem only: this is what max_new_tokens must cover
        poem_lengths.append(len(tokenizer(row["poem"], add_special_tokens=False)["input_ids"]))

    print(f"Examples: {len(df)}")
    print(f"Full pair (question + poem + tags): max = {max(full_lengths)}, mean = {sum(full_lengths) / len(full_lengths):.1f}")
    print(f"Poem only: max = {max(poem_lengths)}, mean = {sum(poem_lengths) / len(poem_lengths):.1f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: uv run python scripts/count_tokens.py <path_to_csv>")
    main(sys.argv[1])
