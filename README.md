# SynWOZ Dialogue Toolkit

Toolkit for generating, moderating, deduplicating, and exploring synthetic dialogues. The repo now centers on a slim command interface so you can run individual stages or wire up a full pipeline.

```
 ┌──────────┐     ┌──────────┐     ┌──────────────────┐
 │generate  ├────►│moderate  ├────►│postprocess (faiss│
 └──────────┘     └──────────┘     └──────────────────┘
        │                                │
        ▼                                ▼
    artifacts/                     synwoz/dashboard
```

---

## Quick Start (uv)

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/features/)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   macOS + Homebrew: `brew install uv`.

2. Create and activate a virtual environment
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. Install dependencies from `requirements.txt`
   ```bash
   uv pip install -r requirements.txt
   ```
   `uv pip` is drop-in compatible with pip, just faster.

4. Set secrets
   ```bash
   cp .env.example .env.local  # or create manually
   ```
   Edit `.env.local` and add your OpenAI key:
   ```
   OPENAI_KEY=sk-...
   ```

5. Prime local resources (optional but recommended)
   ```bash
   uv run python -m synwoz resources --
   ```
   Models default to `./models/sentence_transformer/` and datasets to `./local_datasets/`. Add flags such as `--skip-persona` or `--persona-dataset-path` after the `--` if you need a custom layout.

---

## Working With The Pipeline

All commands go through the module dispatcher. Keep the `--` separator so arguments pass through untouched.

| Stage | Command | Notes |
|-------|---------|-------|
| Full pipeline | `uv run python -m synwoz pipeline --output-dir artifacts/run1 --gen-cmd "--total_generations 10" --dedupe-cmd "-t 0.9"` | Orchestrates generation → moderation → dedupe. Use `--skip-moderation` / `--skip-dedupe` as needed. |
| Resources | `uv run python -m synwoz resources -- --skip-persona` | Downloads models/datasets (wraps `setup.py`), add flags after `--` to customise paths. |
| Generation | `uv run python -m synwoz gen-parallel -- --total_generations 10` | Async generator with hashing + embeddings. `gen-serial` and `gen-perf` stay available. |
| Moderation | `uv run python -m synwoz moderate -- generated.jsonl --output moderated.jsonl --workers 5 --retry-failed` | Wraps `synwoz.moderation.omni`; add `--flagged-output` / `--failed-output` for audit trails. |
| Embedding dedupe | `uv run python -m synwoz post-embed-dedup -- moderated.jsonl deduped.jsonl -t 0.9` | Uses `postprocessing/embedding_faiss.py`. |
| Turn filter | `uv run python -m synwoz post-filter -- deduped.jsonl filtered.jsonl --threshold 5` | Keeps dialogues with more than N turns. |
| Dashboard | `uv run python -m synwoz dashboard --` | Launches Streamlit app (point it at a JSONL output). |

### CLI pattern examples

Run the full pipeline into `artifacts/` with custom knobs:
```bash
uv run python -m synwoz pipeline \
  --output-dir artifacts/run1 \
  --gen-cmd "--total_generations 5" \
  --dedupe-cmd "-t 0.9"
```
This produces `generated.jsonl`, `moderated.jsonl`, `deduped.jsonl`, and moderation audit logs under the chosen artifact directory.

Generate a small batch and save into `artifacts/`:
```bash
uv run python -m synwoz gen-parallel -- \
  --total_generations 5 \
  --output_file artifacts/run1/generated.jsonl
```

Run end-to-end (generate → dedupe) manually:
```bash
uv run python -m synwoz gen-parallel -- --total_generations 3 --output_file artifacts/run1/generated.jsonl
uv run python -m synwoz moderate -- artifacts/run1/generated.jsonl --output artifacts/run1/moderated.jsonl --workers 2
uv run python -m synwoz post-embed-dedup -- artifacts/run1/moderated.jsonl artifacts/run1/deduped.jsonl -t 0.9
```

---

## Config Cheatsheet

- Shared constants live in `synwoz/common/config.py` (emotion vocab, scenario categories, regions).
- Generation scripts load datasets from `./local_datasets/` and the sentence transformer from `./models/sentence_transformer/` by default. Update your config dict if you store them elsewhere.
- Override the persona dataset location with `--persona-dataset-path` on `gen-parallel`/`gen-serial` or by editing the config dict.
- `.env.local` is loaded automatically for `OPENAI_KEY` and other overrides.

---

## Repository Map

```
resources/                # Git-ignored cache for downloaded models/datasets
artifacts/                # Git-ignored outputs from pipeline runs
synwoz/
  __main__.py             # Dispatcher for module commands
  generation/
    configuration.py      # Typed configuration helper feeding the generators
    persona.py            # Persona dataset sampling utilities
    storage.py            # Shared persistence helpers for dialogues/hashes/embeddings
    parallel_gen.py       # Async generation engine
    serial_gen.py         # Serial generation engine
  moderation/             # JSONL moderation pipeline
  postprocessing/         # Embedding dedupe, filtering, HF upload helpers
  dashboard/              # Streamlit UI + helpers
  resources/              # uv-enabled download helpers
  scripts/                # Operational utilities (batch driver, etc.)
setup.py                  # Legacy bootstrap script (still used by resources sync)
requirements.txt          # Runtime dependencies (sync via uv)
```

---

## Troubleshooting

- Missing spaCy model? Install once: `uv run python -m spacy download en_core_web_sm`.
- Dataset/model not found? Re-run `uv run python setup.py` or point the scripts at your preferred folders.
- Need to pin dependencies? `uv pip compile requirements.txt > requirements.lock` (optional step).
- Dashboard not seeing data? Ensure you feed it a JSONL produced by the moderation or dedupe stages.

---

## Next Steps

- Explore notebooks in `notebooks/` for deeper analytics.
- Tailor CLI configs to your study (temperature/top_p ranges, persona clusters, etc.).
- Package artifacts from `artifacts/` when preparing results.
