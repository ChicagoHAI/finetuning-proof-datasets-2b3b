import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from sklearn.utils import resample
from torch.utils.data import random_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Trainer,
)


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def letter_for_index(idx: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return letters[idx]


def format_mmlu_question(question: str, options: List[str]) -> str:
    option_lines = [f"{letter_for_index(i)}. {opt}" for i, opt in enumerate(options)]
    return (
        f"Question: {question}\nOptions:\n"
        + "\n".join(option_lines)
        + "\nAnswer with the option letter."
    )


def build_fewshot_prefix(samples: List[Dict]) -> str:
    blocks = []
    for ex in samples:
        prompt = format_mmlu_question(ex["question"], ex["options"])
        answer_letter = letter_for_index(ex["answer_index"])
        blocks.append(f"{prompt}\nAnswer: {answer_letter}")
    return "\n\n".join(blocks) + "\n\n"


def greedy_predict(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 4,
    device: str = "cpu",
) -> List[str]:
    tokenized = tokenizer(
        prompts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    outputs = model.generate(**tokenized, max_new_tokens=max_new_tokens)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def accuracy_from_predictions(labels: List[str], preds: List[str]) -> float:
    clean_preds = []
    for p in preds:
        # Extract first uppercase letter guess; fallback to full string.
        for ch in p:
            if ch.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                clean_preds.append(ch.upper())
                break
        else:
            clean_preds.append(p.strip().upper())
    match = [int(p == l) for p, l in zip(clean_preds, labels)]
    return sum(match) / len(match)


def bootstrap_ci(labels: List[str], preds: List[str], n_boot: int = 500) -> Tuple[float, Tuple[float, float]]:
    clean_preds = []
    for p in preds:
        for ch in p:
            if ch.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                clean_preds.append(ch.upper())
                break
        else:
            clean_preds.append(p.strip().upper())
    accs = []
    for _ in range(n_boot):
        res_labels, res_preds = resample(labels, clean_preds, replace=True)
        accs.append(sum(int(a == b) for a, b in zip(res_labels, res_preds)) / len(res_labels))
    accs.sort()
    return (
        sum(int(a == b) for a, b in zip(labels, clean_preds)) / len(labels),
        (accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot)]),
    )


@dataclass
class EvalResult:
    name: str
    accuracy: float
    ci: Tuple[float, float]
    n: int
    details_path: str


def evaluate_mmlu_subset(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    subset_size: int = 200,
    fewshot_examples: List[Dict] | None = None,
    seed: int = 42,
    tag: str = "zero_shot",
    device: str = "cpu",
) -> EvalResult:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=subset_size, replace=False)
    subset = dataset.select(indices)
    fewshot_prefix = build_fewshot_prefix(fewshot_examples) if fewshot_examples else ""
    prompts = [fewshot_prefix + format_mmlu_question(ex["question"], ex["options"]) for ex in subset]
    labels = [letter_for_index(ex["answer_index"]) for ex in subset]
    preds = greedy_predict(model, tokenizer, prompts, device=device)
    acc, ci = bootstrap_ci(labels, preds)
    details_file = RESULTS_DIR / f"mmlu_preds_{tag}.jsonl"
    with details_file.open("w") as f:
        for ex, pred in zip(subset, preds):
            f.write(json.dumps({"id": ex["question_id"], "label": letter_for_index(ex["answer_index"]), "pred": pred}) + "\n")
    return EvalResult(tag, acc, ci, len(subset), str(details_file))


def prepare_mmlu_training_splits(dataset: Dataset, train_size: int = 400, eval_size: int = 200, seed: int = 42):
    # Use a deterministic split on a subset of the test set (acknowledging this is a probe, not official training).
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=train_size + eval_size, replace=False)
    subset = dataset.select(indices)
    train_subset, eval_subset = random_split(
        subset, [train_size, eval_size], generator=torch.Generator().manual_seed(seed)
    )
    return Dataset.from_list(list(train_subset)), Dataset.from_list(list(eval_subset))


def tokenize_examples(tokenizer: AutoTokenizer, max_input: int = 512, max_target: int = 8):
    def _tok(batch):
        prompts = [format_mmlu_question(q, o) for q, o in zip(batch["question"], batch["options"])]
        targets = [letter_for_index(idx) for idx in batch["answer_index"]]
        model_inputs = tokenizer(
            prompts,
            max_length=max_input,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            targets,
            max_length=max_target,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _tok


def finetune_mmlu(
    model_name: str,
    train_ds: Dataset,
    eval_ds: Dataset,
    output_dir: Path,
    seed: int = 42,
    epochs: float = 3.0,
    lr: float = 5e-4,
    batch_size: int = 8,
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, Dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenized_train = train_ds.map(tokenize_examples(tokenizer), batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(tokenize_examples(tokenizer), batched=True, remove_columns=eval_ds.column_names)

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=10,
        seed=seed,
        report_to="none",
        predict_with_generate=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")
    with (output_dir / "eval_metrics.json").open("w") as f:
        json.dump(eval_metrics, f, indent=2)
    return model, tokenizer, eval_metrics


def evaluate_bbh_tasks(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    tasks: List[str],
    root: Path,
    per_task_limit: int = 50,
    device: str = "cpu",
) -> Dict:
    summaries = {}
    for task in tasks:
        ds = load_from_disk(root / task)["test"]
        subset = ds.select(range(min(per_task_limit, len(ds))))
        prompts = [f"{row['input']}\nAnswer:" for row in subset]
        labels = [row["target"] if "target" in row else row["targets"][0] for row in subset]
        preds = greedy_predict(model, tokenizer, prompts, device=device, max_new_tokens=8)
        matches = [int(p.strip().lower() == l.strip().lower()) for p, l in zip(preds, labels)]
        acc = sum(matches) / len(matches)
        summaries[task] = {
            "accuracy": acc,
            "n": len(subset),
            "details": [{"label": l, "pred": p} for l, p in zip(labels, preds)],
        }
        details_path = RESULTS_DIR / f"bbh_{task}_preds.json"
        with details_path.open("w") as f:
            json.dump(summaries[task]["details"], f, indent=2)
        summaries[task]["details_path"] = str(details_path)
    return summaries


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(device)

    mmlu = load_from_disk("datasets/mmlu_pro")
    test_ds = mmlu["test"]

    # Build train/eval subsets for fine-tuning probe.
    train_ds, eval_ds = prepare_mmlu_training_splits(test_ds, train_size=400, eval_size=200, seed=42)

    # Few-shot exemplars drawn from training subset for prompting baseline.
    fewshot_examples = train_ds.shuffle(seed=42).select(range(3))

    # Baseline evaluations.
    zero_res = evaluate_mmlu_subset(
        base_model, tokenizer, eval_ds, subset_size=len(eval_ds), tag="zero_shot", device=device
    )
    fewshot_res = evaluate_mmlu_subset(
        base_model,
        tokenizer,
        eval_ds,
        subset_size=len(eval_ds),
        fewshot_examples=fewshot_examples,
        tag="few_shot",
        device=device,
    )

    # Fine-tuning.
    finetune_dir = RESULTS_DIR / "finetuned_mmlu_t5_small"
    finetune_dir.mkdir(parents=True, exist_ok=True)
    ft_model, ft_tokenizer, ft_eval_metrics = finetune_mmlu(
        base_model_name, train_ds, eval_ds, finetune_dir, seed=42, epochs=3, lr=5e-4, batch_size=8
    )
    ft_model = ft_model.to(device)
    finetune_res = evaluate_mmlu_subset(
        ft_model, ft_tokenizer, eval_ds, subset_size=len(eval_ds), tag="finetuned", device=device
    )

    # BBH evaluation (zero-shot only for time).
    bbh_tasks = ["boolean_expressions", "causal_judgement", "logical_deduction_three_objects"]
    bbh_root = Path("datasets/bbh")
    bbh_results = evaluate_bbh_tasks(ft_model, ft_tokenizer, bbh_tasks, bbh_root, per_task_limit=50, device=device)

    summary = {
        "mmlu_pro": {
            "zero_shot": vars(zero_res),
            "few_shot": vars(fewshot_res),
            "finetuned": vars(finetune_res),
            "finetune_eval_metrics": ft_eval_metrics,
        },
        "bbh": bbh_results,
        "config": {
            "model": base_model_name,
            "device": device,
            "train_size": len(train_ds),
            "eval_size": len(eval_ds),
            "seed": 42,
        },
    }
    with (RESULTS_DIR / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("Saved results to", RESULTS_DIR / "metrics.json")
    print(json.dumps(summary["mmlu_pro"], indent=2))


if __name__ == "__main__":
    main()
