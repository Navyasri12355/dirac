"""ROUGE evaluation harness for sample Q&A pairs in eval/sample_qas.json."""

import json
from pathlib import Path

import evaluate

rouge = evaluate.load("rouge")
SAMPLE_FILE = Path("eval/sample_qas.json")

# 5 sample Q&A pairs used when eval/sample_qas.json is unavailable.
DEFAULT_SAMPLES = [
    {
        "question": "What is LoRA?",
        "reference": "LoRA is a parameter-efficient fine-tuning method "
        "that injects trainable low-rank matrices into each "
        "layer of a pre-trained transformer model.",
        "prediction": "LoRA adds small trainable matrices to frozen model "
        "weights, reducing memory and compute for fine-tuning.",
    },
    {
        "question": "What is RLHF?",
        "reference": "RLHF uses human feedback to train a reward model "
        "which then guides policy optimisation via PPO.",
        "prediction": "RLHF trains language models using reinforcement "
        "learning guided by human preference data.",
    },
    {
        "question": "Define chain-of-thought prompting.",
        "reference": "Chain-of-thought prompting elicits step-by-step "
        "reasoning from LLMs by including reasoning examples.",
        "prediction": "Chain-of-thought prompting improves reasoning by "
        "asking the model to show its intermediate steps.",
    },
    {
        "question": "What is RAG?",
        "reference": "RAG augments generation with retrieved documents "
        "to ground outputs in factual external knowledge.",
        "prediction": "RAG retrieves relevant documents and feeds them "
        "to the language model before generating an answer.",
    },
    {
        "question": "What is QLoRA?",
        "reference": "QLoRA combines 4-bit quantisation with LoRA to "
        "enable fine-tuning of large models on single GPUs.",
        "prediction": "QLoRA uses quantised weights with low-rank adapters "
        "to fine-tune large language models on consumer GPUs.",
    },
]


def run_eval(samples: list[dict[str, str]] | None = None) -> dict[str, float]:
    if samples is None:
        if SAMPLE_FILE.exists():
            samples = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
        else:
            samples = DEFAULT_SAMPLES

    if not samples:
        raise ValueError("No evaluation samples provided.")

    preds = [s["prediction"] for s in samples]
    refs = [s["reference"] for s in samples]

    raw_results = rouge.compute(predictions=preds, references=refs)
    if raw_results is None:
        raise RuntimeError("ROUGE computation failed to return a result.")

    results = {str(k): float(v) for k, v in raw_results.items()}

    print("\n-- ROUGE Scores ----------------------------")
    for metric, score in results.items():
        print(f" {metric:12s}: {score:.4f}")
    print("--------------------------------------------")

    return results


if __name__ == "__main__":
    run_eval()