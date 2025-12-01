# Literature Review

## Research Area Overview
Evaluating whether some benchmarks are resistant to performance gains from fine-tuning hinges on (1) dataset construction that minimizes train/test leakage and shortcut cues, and (2) tasks that require transferable reasoning rather than memorization. Recent work revisits classic knowledge benchmarks (MMLU) to harden them against contamination (MMLU-Pro) and proposes adversarial reasoning suites (BBH) and truthfulness evaluations (TruthfulQA) that stress generalization. These resources collectively probe whether fine-tuning yields genuine capability gains or just captures narrow distributions.

## Key Papers

### Measuring Massive Multitask Language Understanding (arXiv:2009.03300)
- **Authors**: Hendrycks et al.
- **Year**: 2020
- **Source**: arXiv
- **Key Contribution**: Introduces MMLU with 57 subjects to evaluate broad knowledge and reasoning.
- **Methodology**: Multiple-choice QA; evaluates zero-shot/few-shot prompting against supervised baselines.
- **Datasets Used**: Curated MMLU benchmark.
- **Results**: Demonstrated sizable gaps between LMs and human performance; modest gains from fine-tuning.
- **Code Available**: In evaluation harnesses (e.g., lm-evaluation-harness).
- **Relevance**: Baseline dataset to contrast with newer “proof” variants; known to be partially contaminated in some pretraining corpora.

### MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark (arXiv:2406.01574)
- **Authors**: Liu, Zhou, et al.
- **Year**: 2024
- **Source**: arXiv
- **Key Contribution**: Rebuilds MMLU with stricter quality control, adversarial filtering, and contamination reduction.
- **Methodology**: Reformulated questions, difficulty calibration, and chain-of-thought references; evaluates open- and closed-weight LMs.
- **Datasets Used**: New MMLU-Pro dataset; comparisons to original MMLU and other benchmarks.
- **Results**: Models show notably lower accuracy vs MMLU; fine-tuning provides smaller gains, supporting “finetuning-resistant” claims.
- **Code Available**: Yes (GitHub TIGER-AI-Lab/MMLU-Pro).
- **Relevance**: Direct candidate for a finetuning-proof evaluation set.

### Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them (arXiv:2210.09261)
- **Authors**: Suzgun, Scales, Schärli, et al.
- **Year**: 2022
- **Source**: arXiv
- **Key Contribution**: Defines BIG-Bench Hard (BBH), a subset of adversarial tasks requiring reasoning.
- **Methodology**: Evaluates LMs with and without chain-of-thought prompting; analyzes transfer and scaling effects.
- **Datasets Used**: 27 BBH tasks (~250 examples each).
- **Results**: Chain-of-thought boosts performance; fine-tuning alone often underperforms CoT prompting.
- **Code Available**: Tasks integrated into evaluation harnesses; open dataset on HF.
- **Relevance**: Stress-tests reasoning and reduces benefits of naïve fine-tuning.

### TruthfulQA: Measuring How Models Mimic Human Falsehoods (arXiv:2109.07958)
- **Authors**: Lin, Hilton, Evans
- **Year**: 2021
- **Source**: arXiv
- **Key Contribution**: Benchmark for factual robustness against common misconceptions.
- **Methodology**: Questions paired with correct and common false answers; evaluates truthfulness under different prompting.
- **Datasets Used**: TruthfulQA question set.
- **Results**: Models often repeat falsehoods; improvements depend on safety/robustness techniques more than fine-tuning scale.
- **Code Available**: Yes via evaluation harnesses.
- **Relevance**: Useful to test whether fine-tuning reduces hallucination rather than memorizing misleading patterns.

### Think You Have Solved Question Answering? Try ARC (arXiv:1803.05457)
- **Authors**: Clark et al.
- **Year**: 2018
- **Source**: arXiv
- **Key Contribution**: Science exam QA benchmark emphasizing multi-step reasoning over fact lookup.
- **Methodology**: Challenge and Easy sets; evaluates IR baselines, neural QA, and reasoning models.
- **Datasets Used**: ARC dataset.
- **Results**: Baselines underperform on Challenge split; fine-tuning helps modestly but not decisively without reasoning modules.
- **Code Available**: Yes via AI2 resources and evaluation harnesses.
- **Relevance**: Classic “hard” QA set—serves as a comparison point for newer finetuning-resistant benchmarks.

## Common Methodologies
- Contamination reduction (deduplication, source blacklisting) to limit pretraining overlap (MMLU-Pro).
- Adversarial/logic-focused task design to require reasoning (BBH, ARC).
- Chain-of-thought prompting as a baseline vs supervised fine-tuning (BBH).
- Multiple-choice QA for controlled evaluation (MMLU, MMLU-Pro, ARC).

## Standard Baselines
- Zero-/few-shot prompting with and without chain-of-thought.
- Supervised finetuning on related but non-identical corpora (e.g., MMLU train-like subsets).
- Retrieval-augmented inference for ARC/TruthfulQA.
- Scaling baselines: smaller vs larger models to study emergence and fine-tuning sensitivity.

## Evaluation Metrics
- Accuracy on multiple-choice QA (MMLU, MMLU-Pro, ARC).
- Exact match / string match (BBH tasks).
- Truthfulness score (TruthfulQA; proportion of factually correct answers).
- Secondary: calibration/error rates when CoT is used.

## Datasets in the Literature
- **MMLU**: Broad knowledge; known contamination risk.
- **MMLU-Pro**: Hardened against contamination; early evidence of weaker finetuning gains.
- **BBH**: Adversarial reasoning tasks; small but challenging.
- **TruthfulQA**: Truthfulness/hallucination focus; finetuning may not help unless targeted.
- **ARC-Challenge**: Classic reasoning QA; finetuning needs reasoning aids.

## Gaps and Opportunities
- Limited large-scale public benchmarks explicitly engineered to be “fine-tuning-proof”; most evidence is indirect (performance gaps, smaller fine-tuning deltas).
- Need standardized contamination audits and cross-model replication for MMLU-Pro and BBH.
- Few open datasets combine robustness to fine-tuning with safety/truthfulness objectives.

## Recommendations for Our Experiment
- **Recommended datasets**: MMLU-Pro (primary hardened benchmark); BBH (adversarial reasoning); optionally TruthfulQA or ARC-Challenge for truthfulness and science reasoning axes.
- **Recommended baselines**: Zero-/few-shot with and without chain-of-thought; small supervised fine-tuning on similar-but-disjoint data to test robustness; retrieval-augmented inference for ARC/TruthfulQA.
- **Recommended metrics**: Accuracy/EM per task; relative improvement over base model; contamination-aware reporting (e.g., overlap checks); calibration metrics if CoT used.
- **Methodological considerations**: Keep fine-tuning data distribution disjoint from evaluation tasks; report gains vs prompting-only baselines; evaluate across model scales to see if robustness persists; log per-task error modes to detect shortcut exploitation.
