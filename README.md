# SynWOZ

A toolkit for generating, moderating, deduplicating, and exploring synthetic task-oriented dialogues.

**ACL 2025 Student Research Workshop (SRW) Submission**

## Quick Start

```bash
# 1. Create and activate virtual environment
uv venv && source .venv/bin/activate

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env.local
# Edit .env.local and add: OPENAI_KEY=sk-...

# 4. Download models and datasets
uv run python -m synwoz resources --

# 5. (Optional) Install spaCy model if needed
uv run python -m spacy download en_core_web_sm
```

## Pipeline

```
generate → moderate → postprocess (FAISS deduplication)
```

### Full Pipeline
```bash
uv run python -m synwoz pipeline --output-dir artifacts/run1 --gen-cmd "--total_generations 100"
```

### Individual Stages
```bash
# Generate dialogues
uv run python -m synwoz gen-parallel -- --total_generations 10 --output_file artifacts/generated.jsonl

# Moderate content
uv run python -m synwoz moderate -- artifacts/generated.jsonl --output artifacts/moderated.jsonl

# Deduplicate
uv run python -m synwoz post-embed-dedup -- artifacts/moderated.jsonl artifacts/deduped.jsonl -t 0.9
```

### Explore Results
```bash
uv run python -m synwoz dashboard --
```

## Project Structure

```
synwoz/
├── generation/      # Dialogue generation (parallel/serial)
├── moderation/      # OpenAI moderation API wrapper
├── postprocessing/  # FAISS deduplication, filtering
├── pipeline/        # End-to-end orchestration
├── dashboard/       # Streamlit visualization
└── common/          # Shared config and utilities
```

## Analysis & Experiments

- `ablation_experiments.py` - Ablation studies
- `threshold_sweep_ablation.py` - Threshold analysis
- `model_comparison.py` - Model comparisons
- `analyze_dataset.py` - Dataset analytics
- `downstream_tod_eval.py` - Downstream evaluation

## Paper

LaTeX source in `latex/`. See `formatting.md` for ACL style guidelines.
