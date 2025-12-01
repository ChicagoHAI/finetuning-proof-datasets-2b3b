# Finetuning-Proof Datasets Probe
- Question: Do hardened benchmarks like MMLU-Pro and BBH resist gains from supervised fine-tuning?  
- Key findings: Flan-T5-small fine-tuned on 400 MMLU-Pro examples improved accuracy only slightly (10.5% → 14.0%, p=0.29). BBH remained very low (0–44% on sampled tasks). Results suggest limited sensitivity to small-scale fine-tuning.

## Reproduce
```bash
uv venv
source .venv/bin/activate
uv sync            # installs dependencies from pyproject.toml
python experiments.py
```
Outputs are saved under `results/` (metrics, predictions, fine-tuned checkpoint).

## Files
- `experiments.py` – end-to-end pipeline (baselines, fine-tuning, evaluation).  
- `planning.md` – research plan.  
- `REPORT.md` – full report with results and analysis.  
- `datasets/` – pre-downloaded MMLU-Pro and BBH data (local, excluded from git).  
- `code/` – upstream repos for reference (lm-evaluation-harness, mmlu-pro).

## Notes
- Model: `google/flan-t5-small`; seed=42; train 400 / eval 200 MMLU-Pro probe split.  
- Hardware: CUDA available; training ~22s.  
- Limitations: small model and subset; train/eval drawn from test split for exploratory purposes; BBH only zero-shot.  
- See `REPORT.md` for detailed methodology, metrics, and next steps.
