# SynWOZ

A toolkit for generating, moderating, deduplicating, and exploring synthetic task-oriented dialogues.

**ACL 2025 Student Research Workshop (SRW) Submission**

## Dataset

The generated SynWOZ dataset is available on HuggingFace:

🤗 **[Ayushnangia/SynWOZ](https://huggingface.co/datasets/Ayushnangia/SynWOZ)**

50k+ synthetic task-oriented dialogues covering restaurants, hotels, taxis, flights, and more.

### Load the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("Ayushnangia/SynWOZ")
```

### Dataset Features

- **Multi-service dialogues**: Restaurant, hotel, taxi, train, flight, attraction, hospital, bus
- **Rich annotations**: Intent, emotions (user & assistant), scenario category, resolution status
- **Diverse personas**: Generated using FinePersonas for realistic user profiles
- **Quality assured**: Moderated via OpenAI API + FAISS semantic deduplication

## Quick Start

```bash
# 1. Create and activate virtual environment
uv venv && source .venv/bin/activate

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env.local
# Edit .env.local and add: OPENAI_KEY=sk-...

# 4. Download models and datasets (includes SynWOZ from HuggingFace)
uv run python -m synwoz resources --

# 5. (Optional) Install spaCy model if needed
uv run python -m spacy download en_core_web_sm
```

### Resource Download Options

```bash
# Download everything (models + all datasets)
uv run python -m synwoz resources --

# Skip SynWOZ dataset (only source datasets for generation)
uv run python -m synwoz resources -- --skip-synwoz

# Skip persona dataset
uv run python -m synwoz resources -- --skip-persona

# Force re-download
uv run python -m synwoz resources -- --force
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
