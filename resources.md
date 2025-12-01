# Resources Catalog

## Summary
Catalog of papers, datasets, and code collected for studying whether certain benchmarks resist fine-tuning gains.

## Papers
Total papers downloaded: 5

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Measuring Massive Multitask Language Understanding | Hendrycks et al. | 2020 | `papers/2009.03300_mmlu.pdf` | Original MMLU benchmark. |
| MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark | Liu, Zhou, et al. | 2024 | `papers/2406.01574_mmlu_pro.pdf` | Hardened MMLU variant targeting contamination and robustness. |
| Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them | Suzgun et al. | 2022 | `papers/2210.09261_bigbench_hard.pdf` | Defines BBH adversarial tasks. |
| TruthfulQA: Measuring How Models Mimic Human Falsehoods | Lin et al. | 2021 | `papers/2109.07958_truthfulqa.pdf` | Truthfulness benchmark to test robustness to misleading cues. |
| Think You Have Solved Question Answering? Try ARC | Clark et al. | 2018 | `papers/1803.05457_arc.pdf` | Science reasoning QA; challenging for shallow finetuning. |

See `papers/README.md` for details.

## Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| MMLU-Pro | HuggingFace `TIGER-Lab/MMLU-Pro` | 12,032 test / 70 val | Multi-task MCQ + CoT | `datasets/mmlu_pro` | Contamination-reduced, robust to finetuning. |
| BIG-Bench Hard (BBH) | HuggingFace `lukaemon/bbh` | 27 tasks, ~250 ex/task | Adversarial reasoning | `datasets/bbh/<task>` | Small but challenging tasks; selective loading. |

See `datasets/README.md` for download instructions and samples.

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness | Standard eval suite covering MMLU/BBH/TruthfulQA/ARC | `code/lm-evaluation-harness` | Supports HF/OpenAI/vLLM backends. |
| mmlu-pro | https://github.com/TIGER-AI-Lab/MMLU-Pro | Official scripts/configs for MMLU-Pro | `code/mmlu-pro` | Includes task configs and contamination notes. |

## Resource Gathering Notes

### Search Strategy
- Focused on benchmarks explicitly discussing contamination and robustness (MMLU-Pro) and adversarial reasoning (BBH).
- Used arXiv API queries for titles/keywords and cloned official repos for reproducibility.

### Selection Criteria
- Benchmarks cited as challenging or contamination-reduced.
- Availability of open data/code for immediate experimentation.
- Coverage of reasoning, knowledge breadth, and truthfulness axes.

### Challenges Encountered
- BBH HuggingFace dataset requires per-task configs; scripted batch download per task.
- Limited number of explicitly “fine-tuning-proof” datasets; leaned on contamination-reduced or adversarial benchmarks.

### Gaps and Workarounds
- No single canonical finetuning-proof dataset; combined MMLU-Pro (harder/cleaner) with BBH (adversarial) to approximate.
- Additional truthfulness/safety datasets (e.g., TruthfulQA) remain to be added if needed; not downloaded to keep footprint small.

## Recommendations for Experiment Design
1. **Primary datasets**: Evaluate on MMLU-Pro and BBH; optionally layer TruthfulQA/ARC for truthfulness and science reasoning.
2. **Baseline methods**: Zero-/few-shot prompting (with/without CoT) and light supervised fine-tuning on disjoint corpora to measure gains vs prompting.
3. **Evaluation metrics**: Accuracy/EM per task, relative improvement over base prompts, contamination checks, and calibration/error analysis for CoT.
4. **Code reuse**: Use `lm-evaluation-harness` for unified evaluation; reference `mmlu-pro` configs for contamination-aware setup.
