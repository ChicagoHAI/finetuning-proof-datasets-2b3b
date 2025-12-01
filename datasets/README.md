# Downloaded Datasets

Data files are stored locally under `datasets/` but excluded from git (.gitignore). Use the instructions below to re-download.

## Dataset 1: MMLU-Pro
- **Source**: HuggingFace `TIGER-Lab/MMLU-Pro` (arXiv:2406.01574)
- **Size**: 12,032 test, 70 validation examples; multiple-choice QA with chain-of-thought references
- **Format**: HuggingFace dataset (columns: `question`, `options`, `answer`, `answer_index`, `cot_content`, `category`, `src`)
- **Task**: Multi-task knowledge and reasoning; designed to reduce data contamination and shortcut finetuning
- **Splits**: `test`, `validation`
- **License**: MIT (per dataset card)
- **Location**: `datasets/mmlu_pro`

### Download Instructions
**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("TIGER-Lab/MMLU-Pro")
dataset.save_to_disk("datasets/mmlu_pro")
```

### Loading the Dataset
```python
from datasets import load_from_disk
mmlu_pro = load_from_disk("datasets/mmlu_pro")
sample = mmlu_pro["test"][0]
```

### Sample Data
See `datasets/mmlu_pro/samples/sample.json` for the first 5 records.

### Notes
- Contains chain-of-thought rationales; can probe whether finetuning leverages reasoning vs memorization.
- Test split size allows robust measurement without heavy storage (~6 MB total with BBH).

## Dataset 2: BIG-Bench Hard (BBH)
- **Source**: HuggingFace `lukaemon/bbh` (tasks from arXiv:2210.09261)
- **Size**: 27 task configs; most tasks have ~250 examples (total ~6.7k)
- **Format**: HuggingFace dataset per-task; typically `inputs` and `targets`/`label` fields depending on task
- **Task**: Adversarial reasoning tasks spanning logic, math, and language; intended to be resistant to superficial finetuning
- **Splits**: `test` only
- **License**: Apache-2.0 (per dataset card)
- **Location**: `datasets/bbh/<task_name>` with aggregated samples in `datasets/bbh/samples/sample.json`

### Download Instructions
**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
tasks = [
    # abbreviated list; see the dataset card for all tasks
    "boolean_expressions", "causal_judgement", "date_understanding",
    "formal_fallacies", "geometric_shapes", "logical_deduction_three_objects"
]
for task in tasks:
    ds = load_dataset("lukaemon/bbh", task)
    ds.save_to_disk(f"datasets/bbh/{task}")
```

### Loading the Dataset
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/bbh/formal_fallacies")
sample = ds["test"][0]
```

### Sample Data
See `datasets/bbh/samples/sample.json` for 2 examples per task.

### Notes
- Tasks are small but adversarial; good for measuring whether fine-tuning transfers reasoning or merely memorizes.
- Per-task layout keeps storage manageable and allows selective evaluation.
