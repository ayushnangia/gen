# Repository Guidelines

## Project Structure & Module Organization
Core sources live in `synwoz/`, with the dispatcher in `synwoz/__main__.py` routing to modular packages for `generation`, `moderation`, `postprocessing`, the Streamlit `dashboard`, and orchestration helpers under `synwoz/pipeline`. Shared constants sit in `synwoz/common`, while scratch outputs belong in `artifacts/` and cached assets in the git-ignored `resources/`, `models/`, or `local_datasets/` folders. Keep exploratory notebooks in `notebooks/` tidy and checkpoint-free.

## Build, Test, and Development Commands
Bootstrap the environment with `uv venv && source .venv/bin/activate`, then install dependencies via `uv pip install -r requirements.txt`. The full pipeline runs through `uv run python -m synwoz pipeline --output-dir artifacts/run1`, chaining generation → moderation → dedupe. Iterate on individual stages using `uv run python -m synwoz gen-parallel -- --total_generations 5`, `uv run python -m synwoz moderate -- input.jsonl --output moderated.jsonl`, or `uv run python -m synwoz post-embed-dedup -- moderated.jsonl deduped.jsonl -t 0.9`. Launch the dashboard locally with `uv run python -m synwoz dashboard --`.

## Coding Style & Naming Conventions
Use 4-space indentation and PEP 8 conventions: snake_case for modules and functions, PascalCase for classes, and UPPERCASE for shared constants (see `synwoz/common/config.py`). Add type hints for new public APIs, keep docstrings focused on intent, and align CLI flag names with existing patterns like `--total_generations`. Notebook commits should strip execution counts and large outputs.

## Testing Guidelines
Automated coverage is light; validate changes by running the relevant CLI stage and inspecting outputs under `artifacts/`. When adding deterministic utilities, create targeted `pytest` modules under a new `tests/` package and invoke them with `uv run python -m pytest`. Preserve moderation audit hooks (`--flagged-output`, `--failed-output`) so dashboards and postprocessing remain stable.

## Commit & Pull Request Guidelines
Recent history uses conventional prefixes such as `feat(scope): summary` and `refactor: tidy config usage`; mirror that style and keep each commit scoped. Document configuration or artifact impacts in the body. Pull requests should explain the motivation, list validation commands, attach dashboard screenshots when UI changes occur, and link issues or design docs. Remove temporary artifacts before pushing.

## Security & Configuration Tips
Never commit secrets; copy `.env.example` to `.env.local` and keep API keys local. Confirm `OPENAI_KEY` loads before running generation stages. Large downloads land in git-ignored directories—verify paths under `synwoz/resources` before packaging results. Note dataset sources and licenses in PR descriptions when adding external assets.
