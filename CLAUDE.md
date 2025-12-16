# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SynWOZ is a toolkit for generating, moderating, deduplicating, and exploring synthetic task-oriented dialogues. This project is being developed for submission to **ACL 2025 Student Research Workshop (SRW)**.

The codebase uses a modular CLI dispatcher (`python -m synwoz`) that routes commands to specialized scripts, supporting a multi-stage pipeline:

```
generate → moderate → postprocess (FAISS deduplication)
    ↓           ↓              ↓
         artifacts/      synwoz/dashboard
```

## Environment Setup

**Package Manager**: This project uses `uv` (a fast Python package installer). All commands should be prefixed with `uv run`.

**Environment file**: Copy `.env.example` to `.env.local` and add your OpenAI API key:
```bash
cp .env.example .env.local
# Edit .env.local and set OPENAI_KEY=sk-...
```

**Initial setup**:
```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv run python -m synwoz resources --
```

**spaCy model** (if missing model errors):
```bash
uv run python -m spacy download en_core_web_sm
```

## Command Patterns

All commands use the dispatcher pattern: `uv run python -m synwoz <subcommand> -- <args>`. The `--` separator ensures arguments pass through to underlying scripts.

### Quick Reference

| Task | Command |
|------|---------|
| Full pipeline | `uv run python -m synwoz pipeline --output-dir artifacts/run1 --gen-cmd "--total_generations 10"` |
| Generate (parallel) | `uv run python -m synwoz gen-parallel -- --total_generations 10 --output_file artifacts/generated.jsonl` |
| Generate (serial/debug) | `uv run python -m synwoz gen-serial -- --total_generations 5 --output_file artifacts/generated.json` |
| Moderate | `uv run python -m synwoz moderate -- input.jsonl --output moderated.jsonl --workers 5` |
| Deduplicate | `uv run python -m synwoz post-embed-dedup -- moderated.jsonl deduped.jsonl -t 0.9` |
| Filter by turns | `uv run python -m synwoz post-filter -- deduped.jsonl filtered.jsonl --threshold 5` |
| Dashboard | `uv run python -m synwoz dashboard --` |
| Download resources | `uv run python -m synwoz resources -- --skip-persona` |

## Architecture

### Module Dispatcher (`synwoz/__main__.py`)

Central entry point routing subcommands to modules:
- `gen-serial`, `gen-parallel`, `gen-perf` → `synwoz.generation.*`
- `moderate` → `synwoz.moderation.omni`
- `post-embed-dedup`, `post-filter`, `post-hf-upload` → `synwoz.postprocessing.*`
- `pipeline` → `synwoz.pipeline`
- `dashboard` → `synwoz.dashboard.dashboard` (Streamlit)
- `resources` → `synwoz.resources`

The dispatcher preserves all arguments using `argparse.REMAINDER` and forwards via `sys.argv` manipulation.

### Package Structure

```
synwoz/
├── __main__.py              # CLI dispatcher
├── common/
│   ├── config.py            # Shared constants (emotions, categories, regions)
│   └── logging_utils.py     # Logging configuration helpers
├── generation/
│   ├── parallel_gen.py      # Async generation (recommended, JSONL output)
│   ├── serial_gen.py        # Sync generation (debugging, JSON output)
│   ├── performance_parallel.py
│   ├── configuration.py     # GenerationConfig dataclass
│   ├── persona.py           # Persona sampling from dataset
│   └── storage.py           # Shared persistence (hashes, embeddings)
├── moderation/
│   └── omni.py              # OpenAI moderation API wrapper
├── postprocessing/
│   ├── embedding_faiss.py   # FAISS deduplication
│   ├── filter_num_line_gt_5.py
│   └── hugging_face_upload.py
├── pipeline/
│   └── __init__.py          # PipelineConfig, run_pipeline()
├── dashboard/
│   └── dashboard.py         # Streamlit UI
└── resources/
    └── setup_helpers.py     # Model/dataset download utilities
```

### Generation Flow

1. Sample personas from dataset (cluster/summary indices)
2. Select services (weighted: 30% single, 40% double, 20% triple, 10% quadruple)
3. Generate scenario (category, time slot, region, emotions from `config.py`)
4. Call OpenAI `gpt-4o-mini` for dialogue generation
5. Compute SHA-256 hash (exact dedup) + embedding (semantic dedup)
6. Check similarity threshold, write if unique

### Data Formats

| Stage | Input | Output | Notes |
|-------|-------|--------|-------|
| `gen-parallel` | - | JSONL | Append mode, incremental |
| `gen-serial` | - | JSON | Full rewrite, debugging |
| `moderate` | JSONL | JSONL | Separate flagged/failed outputs |
| `post-embed-dedup` | JSON/JSONL | JSON | FAISS similarity |
| `post-filter` | JSON | JSON | Turn count threshold |

### Key Configuration (`synwoz/common/config.py`)

- `USER_EMOTIONS`, `ASSISTANT_EMOTIONS` - Emotion vocabularies
- `CATEGORIES` - Service types (restaurant, hotel, train, flight, taxi, etc.)
- `TRAVEL_TIME_SLOTS` - Time periods for scenarios
- `RESOLUTION_STATUSES` - Dialogue resolution states
- `REGIONS` - 50+ predefined cities

## Working with Artifacts

- All outputs go to `artifacts/` (git-ignored)
- Pipeline creates timestamped subdirectories: `artifacts/run-YYYYMMDD-HHMMSS/`
- Expected files: `generated.jsonl`, `moderated.jsonl`, `flagged.jsonl`, `deduped.json`

## Models and Datasets

- **LLM**: `gpt-4o-mini` (OpenAI)
- **Embeddings**: `all-MiniLM-L6-v2` → `./models/sentence_transformer/`
- **MultiWOZ**: `pfb30/multi_woz_v22` → `./local_datasets/multi_woz_v22`
- **Personas**: `argilla/FinePersonas-v0.1-clustering-100k` → `./local_datasets/FinePersonas-v0.1-clustering-100k`

Override persona dataset with `--persona-dataset-path`.

## Development Workflow

**Test generation with small batch**:
```bash
uv run python -m synwoz gen-parallel -- --total_generations 5 --output_file artifacts/test/generated.jsonl
```

**Manual pipeline stages**:
```bash
# 1. Generate
uv run python -m synwoz gen-parallel -- --total_generations 10 --output_file artifacts/run1/generated.jsonl

# 2. Moderate
uv run python -m synwoz moderate -- artifacts/run1/generated.jsonl --output artifacts/run1/moderated.jsonl

# 3. Deduplicate
uv run python -m synwoz post-embed-dedup -- artifacts/run1/moderated.jsonl artifacts/run1/deduped.jsonl -t 0.9
```

**Explore with dashboard**:
```bash
uv run python -m synwoz dashboard --
```

## Coding Conventions

- 4-space indentation, PEP 8
- `snake_case` for functions/modules, `PascalCase` for classes, `UPPERCASE` for constants
- Type hints for public APIs
- CLI flags follow existing patterns (`--total_generations`, `--output_file`)
- Commit messages: `feat(scope): summary`, `refactor: description`

## Testing

No automated test suite yet. Validate changes by:
1. Running relevant CLI stage
2. Inspecting outputs in `artifacts/`
3. For new utilities, create `pytest` modules under `tests/` and run with `uv run python -m pytest`

## Notebooks

- `notebooks/gen_insights.ipynb` - Generation analytics and quality metrics
- `notebooks/post_process.ipynb` - Deduplication effectiveness analysis

Strip execution counts and large outputs before committing notebooks.
