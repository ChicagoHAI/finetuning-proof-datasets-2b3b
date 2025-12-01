# Cloned Repositories

## lm-evaluation-harness
- **URL**: https://github.com/EleutherAI/lm-evaluation-harness
- **Purpose**: Standard evaluation harness covering MMLU, BBH, TruthfulQA, ARC, etc.; useful for consistent pre/post-finetuning comparisons.
- **Location**: `code/lm-evaluation-harness`
- **Key files**: `main.py` (CLI), `lm_eval/tasks/` (task definitions), `docs/`.
- **Notes**: Supports plug-in models (HuggingFace, vLLM, OpenAI). Requirements listed in `requirements.txt`.

## mmlu-pro
- **URL**: https://github.com/TIGER-AI-Lab/MMLU-Pro
- **Purpose**: Official scripts and baselines for MMLU-Pro; includes data card, evaluation configs, and prompts.
- **Location**: `code/mmlu-pro`
- **Key files**: `README.md`, `benchmark/` (task configs), `utils/`.
- **Notes**: Helpful for replicating the contamination-reduced evaluation and comparing to MMLU.
