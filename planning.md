# Research Question
Identify whether any currently available benchmarks behave as "fine‑tuning‑proof" datasets—i.e., benchmarks where supervised fine‑tuning yields minimal or no performance gains compared to prompting‑only approaches—focusing on MMLU‑Pro and BIG‑Bench Hard (BBH).

# Background and Motivation
- Many benchmarks report large gains from task‑specific fine‑tuning, but those gains can stem from train/test overlap or shortcut exploitation.
- MMLU‑Pro explicitly targets contamination reduction and harder reasoning; BBH is adversarially constructed to resist superficial learning.
- Understanding which datasets are resistant to fine‑tuning clarifies how to measure genuine capability vs memorization and guides robust evaluation design.

# Hypothesis Decomposition
- H1: MMLU‑Pro accuracy improves only marginally after supervised fine‑tuning a small open model on a subset of validation data, relative to a zero/few‑shot baseline.
- H2: BBH tasks show limited or unstable gains from small‑scale fine‑tuning compared to prompting, reflecting adversarial difficulty.
- Variables: independent—training regimen (none, few‑shot prompting, supervised fine‑tuning); model family/size; dataset (MMLU‑Pro, selected BBH tasks). Dependent—task accuracy/EM and relative improvement (%).
- Success criteria: ≤5–10 percentage‑point gain after fine‑tuning on small subset indicates resistance; larger gains would refute the finetuning‑proof claim.

# Proposed Methodology

## Approach
- Use pre-downloaded datasets: MMLU‑Pro as primary, BBH subset as secondary.
- Baseline: zero‑shot and few‑shot prompting on small open model (Flan‑T5‑Small) plus a stronger API model if available; compare to supervised fine‑tuning (small subset) on the same open model.
- Measure fine‑tuning sensitivity as absolute/relative accuracy lift; inspect per‑category/task variance.

## Experimental Steps
1. Data inspection: load MMLU‑Pro and BBH samples; verify fields and label formats.  
2. Baseline prompting: map MCQ to text‑to‑text format for Flan‑T5‑Small zero‑shot and few‑shot; record accuracy on small held‑out batches.  
3. Fine‑tuning: supervised train Flan‑T5‑Small on a small MMLU‑Pro validation subset (e.g., 300–500 examples) with label text targets; evaluate on disjoint test subset.  
4. BBH probe: run zero‑shot and few‑shot prompting on 3–5 BBH tasks; (optional if time) lightweight fine‑tuning on one task.  
5. Analysis: compute accuracy deltas vs baseline, per‑category breakdown, and statistical significance via bootstrap CIs on proportions.  
6. Error analysis: inspect misclassified examples to see if errors are systematic (reasoning) vs memorization gaps.  
7. Robustness: repeat evaluation with different random seeds or prompt variants to assess stability.

## Baselines
- Zero‑shot Flan‑T5‑Small.
- Few‑shot Flan‑T5‑Small (e.g., 2–3 exemplars).
- Strong reference (if API key available): GPT‑4.1 zero‑shot for upper bound.

## Evaluation Metrics
- Accuracy on MCQ (MMLU‑Pro) and exact match on BBH tasks.
- Relative improvement (%) from fine‑tuning over zero‑shot baseline.
- Bootstrap 95% confidence intervals for accuracy to assess significance of gains.

## Statistical Analysis Plan
- For each condition, estimate accuracy and 95% CI via bootstrap (1k resamples).  
- Test difference in proportions between baseline and fine‑tuned using two‑sided z‑test; p<0.05 considered significant.  
- Report effect sizes as absolute and relative % gains.

# Expected Outcomes
- Anticipate small or unstable gains from fine‑tuning on MMLU‑Pro/BBH with small model, supporting finetuning‑resistance; larger, consistent gains would challenge the hypothesis.

# Timeline and Milestones
- Setup & data inspection: 20–30 min.
- Baseline prompting runs: 30–45 min.
- Fine‑tuning small model: 45–60 min (including training/eval).
- Analysis & plots: 30–45 min.
- Reporting: 20–30 min.

# Potential Challenges
- Compute/time limits for training; mitigate by subsetting and reducing epochs/batch size.
- Prompt formatting sensitivity; mitigate with consistent templates and prompt checks.
- API key availability may restrict strong baseline; fall back to open models.
- Dataset size vs small model capacity may yield noisy estimates; use multiple seeds and CIs.

# Success Criteria
- Completed experiments with documented baselines and fine‑tuning comparisons.  
- Evidence (numbers + CIs) showing limited fine‑tuning gains on MMLU‑Pro/BBH, or clear refutation if gains are significant.  
- Reproducible scripts/notebooks and summarized findings in REPORT.md.
