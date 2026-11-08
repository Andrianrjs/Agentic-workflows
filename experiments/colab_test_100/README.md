# Colab test run: 100 question/poem pairs

Fine-tuning test of Phi-3 on Google Colab, run on 2026-11-08 before the full Vertex AI run.
It uses the same settings as the pipeline on the Clémence branch.

## Setup

- Dataset: `rhyming_qa_v1.csv` (100 question/poem pairs, 4 lines each, AABB rhymes)
- Split: 72 training, 8 validation, 20 test examples (seed 42)
- Model: `microsoft/Phi-3-mini-4k-instruct`, 4-bit quantization, LoRA r=8, alpha=16
- Trainable parameters: 12,582,912 of 3,833,662,464 (0.33%)
- Training: 10 epochs, batch size 16, learning rate 3e-4, `max_seq_length=128`, `packing=False`
- Best model kept with `load_best_model_at_end=True` on `eval_loss`
- Hardware: Google Colab, Tesla T4
- Libraries: transformers 4.46.3, peft 0.13.2, trl 0.12.2, bitsandbytes 0.48.0 (needed for CUDA 13 on Colab)

## Results on the 20 test questions

| Metric | Before fine-tuning | After fine-tuning |
| --- | --- | --- |
| Answers with 4 lines | 0 / 20 (long prose, often cut at 96 tokens) | 20 / 20 |
| Mean rhyme score | 0.00 | 0.25 |
| Poems with both rhyme pairs | 0 | 2 |
| Poems with one rhyme pair | 0 | 6 |

The model learned the poem format. Rhymes are starting to appear but remain weak.
Some answers contain factual errors caused by the rhyme, for example the Mona Lisa
"came from France".

## Training and validation loss

| Epoch | Training loss | Validation loss |
| --- | --- | --- |
| 1 | 1.96 | 1.84 |
| 2 | 1.37 | 1.48 |
| 3 | 1.20 | 1.38 |
| 4 | 0.86 | 1.36 |
| 5 | 0.70 | 1.47 |
| 6 | 0.52 | 1.67 |
| 7 | 0.42 | 1.91 |
| 8 | 0.24 | 2.08 |
| 9 | 0.13 | 2.21 |
| 10 | 0.11 | 2.28 |

Validation loss is lowest at epoch 4, then rises: the model overfits after epoch 4.
The kept model is the epoch 4 checkpoint.

## Comparison with the 10-pair test

With 10 pairs (7 training examples, 10 weight updates), the model still answered in prose
and the rhyme score stayed at 0. With 100 pairs (72 training examples, 50 weight updates),
it learned the 4-line format. Data volume is the main factor.

## Next steps

- Build a dataset of about 700 pairs to improve rhymes.
- Consider reducing `num_train_epochs` from 10 to 5 on Vertex AI.
- Add an evaluation of factual correctness, for example with an LLM judge.

## Files

- `colab_fine_tuning.ipynb`: executed notebook with all outputs
- `colab_predictions.csv`: before/after answers, rhyme scores and reference poems