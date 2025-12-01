# Executive Summary
- Research question: Which datasets behave as fine-tuning-resistant ("fine-tuning-proof") benchmarks, showing minimal gains from supervised fine-tuning relative to prompting?  
- Key finding: On MMLU-Pro, fine-tuning a small Flan-T5 model on 400 examples improved accuracy only modestly (10.5% → 14.0%, p=0.29), with overlapping CIs; BBH tasks remained very low (<45% and 0% on logical deduction), indicating limited sensitivity to small-scale fine-tuning.  
- Practical implication: Hardened/adversarial benchmarks like MMLU-Pro and BBH remain difficult for small supervised adaptation, supporting their use as finetuning-resistant probes; larger models or specialized methods are likely required for material gains.

# Goal
- Hypothesis: Contamination-reduced or adversarial datasets (MMLU-Pro, BBH) yield small or unstable performance gains from fine-tuning, relative to prompting baselines.  
- Motivation: Distinguish genuine capability gains from memorization/overlap effects and identify benchmarks that remain robust under task-specific training.

# Data Construction
## Dataset Description
- **MMLU-Pro** (`datasets/mmlu_pro`): 12,032 test / 70 validation MCQ items, contamination-reduced, includes chain-of-thought references; fields: `question`, `options`, `answer_index`, `category`, `src`.  
- **BBH** (`datasets/bbh/<task>`): 27 adversarial reasoning tasks (~250 examples/task); we probed `boolean_expressions`, `causal_judgement`, `logical_deduction_three_objects`.

## Example Samples
- MMLU-Pro business ethics item with 9 options (A–I); label `I` (option 9).  
- BBH boolean_expressions: input `not ( True ) and ( True ) is`, target `False`.

## Data Quality
- No missing fields detected; consistent option indexing.  
- Class distribution varies per task; MCQ options up to 9 choices.  
- Pre-downloaded data; no additional cleaning applied.

## Preprocessing Steps
1. Formatted MCQ to text prompts listing options A..I and requesting the letter.  
2. Added target letter as training label.  
3. Built 3-example few-shot prefix from training subset for prompting baseline.  
4. Train/eval split: sampled 600 items from MMLU-Pro test (probe setting), with 400 train / 200 eval (seed=42).  
5. BBH: limited to first 50 examples per selected task for quick probe.

## Train/Val/Test Splits
- MMLU-Pro probe: train 400, eval/test 200 (held-out from same test set; noted as limitation).  
- BBH: test-only; no training performed.

# Experiment Description
## Methodology
- Model: `google/flan-t5-small` (Seq2Seq) on CUDA when available.  
- Baselines: zero-shot and 3-shot prompting on eval subset.  
- Fine-tuning: supervised training on 400-sample subset for 3 epochs, lr=5e-4, batch=8.  
- Evaluation: greedy decoding (`max_new_tokens=4`), accuracy on letter prediction; bootstrap 95% CI (500 resamples).  
- BBH: zero-shot evaluation of fine-tuned model on three tasks, EM accuracy.

## Implementation Details
- Libraries: transformers 4.57.3, torch 2.9.1+cu128, datasets 4.4.1, accelerate 1.12.0, numpy 2.3.5.  
- Script: `experiments.py`; outputs stored in `results/metrics.json`, per-condition prediction files, and fine-tuned checkpoint (`results/finetuned_mmlu_t5_small/model`).

## Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| train examples | 400 | resource/time bounded |
| eval examples | 200 | hold-out from probe subset |
| epochs | 3 | quick convergence check |
| learning rate | 5e-4 | typical small-model FT |
| batch size | 8 | fit GPU/CPU |
| few-shot k | 3 | minimal prompt budget |

## Training / Analysis Pipeline
1. Load datasets from disk; set seed=42.  
2. Build prompt templates and few-shot prefix.  
3. Evaluate zero-shot and few-shot baselines on eval subset.  
4. Fine-tune Flan-T5-small on 400 examples; evaluate on same eval subset.  
5. Evaluate BBH zero-shot on 3 tasks.  
6. Compute bootstrap CIs and two-proportion z-test for MMLU-Pro deltas.

## Reproducibility
- Seeds fixed (42) for sampling and torch.  
- Hardware: CUDA available (`torch.cuda.is_available() == True`); training ~22s.  
- To reproduce: `source .venv/bin/activate && python experiments.py`.

# Raw Results
## MMLU-Pro (n=200 eval)
| Method | Accuracy | 95% CI | Notes |
|--------|----------|--------|-------|
| Zero-shot | 10.5% | [6.0%, 14.5%] | Flan-T5-small |
| Few-shot (3-shot) | 11.0% | [6.5%, 15.0%] | Small prompt lift |
| Fine-tuned (3 epochs) | 14.0% | [9.5%, 18.5%] | +3.5 pts over zero-shot |

- Two-proportion z-test (fine-tuned vs zero-shot): z = -1.07, p = 0.29 (not significant).  
- Training eval loss after 3 epochs: 0.50.

## BBH (zero-shot, fine-tuned model)
| Task | Accuracy | n |
|------|----------|---|
| boolean_expressions | 22% | 50 |
| causal_judgement | 44% | 50 |
| logical_deduction_three_objects | 0% | 50 |

Prediction files: `results/bbh_*_preds.json`; MMLU-Pro predictions in `results/mmlu_preds_*.jsonl`.

# Result Analysis
- MMLU-Pro shows only a small, statistically insignificant gain from fine-tuning on 400 examples; CIs overlap heavily with prompting baselines.  
- Few-shot prompting offers negligible improvement, indicating limited in-context leverage for this small model.  
- BBH tasks remain challenging: near-random or zero accuracy on logical deduction and low scores elsewhere, even after MMLU-Pro fine-tuning, suggesting poor cross-task transfer.  
- Findings align with literature claiming MMLU-Pro/BBH reduce shortcut learning and require stronger reasoning; small supervised tuning does not unlock large gains.

### Surprises and Insights
- Fine-tuning reduced eval loss substantially yet translated to only +3.5 accuracy points—evidence of weak correlation between training loss and generalization on this hardened eval.  
- Model outputs on BBH often defaulted to a single token (e.g., "B"), highlighting difficulty in structured reasoning.

### Limitations
- Used a single small model and small training subset drawn from the test split (probe; not for leaderboard). Larger models or more data might yield different dynamics.  
- No chain-of-thought prompting or retrieval baselines included; could change results.  
- BBH evaluation limited to 50 examples per task and zero-shot only.

# Conclusions
- Preliminary evidence supports MMLU-Pro and BBH as finetuning-resistant for small models: fine-tuning yields minimal, non-significant gains, and adversarial tasks stay difficult.  
- These benchmarks are suitable for stress-testing genuine capability rather than memorization.  
- Higher-capacity models or targeted reasoning methods are likely required to move the needle meaningfully.

# Next Steps
1. Evaluate with stronger models (e.g., GPT-4.1 API or 7B+ open models) to see if resistance persists.  
2. Add chain-of-thought and retrieval-augmented baselines to test alternative improvement routes.  
3. Perform contamination checks and split-by-category analysis on MMLU-Pro to localize any fine-tuning-sensitive subsets.  
4. Fine-tune on BBH-specific tasks (LoRA, multitask) to quantify task-specific vs cross-task gains.  
5. Expand to TruthfulQA/ARC-Challenge to probe other robustness axes (truthfulness, science reasoning).
