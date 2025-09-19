"""High-level orchestration helpers for the SynWOZ pipeline."""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class PipelineConfig:
    """Arguments accepted by :func:`run_pipeline`."""

    output_dir: Path
    generator_command: List[str]
    moderation_command: List[str]
    dedupe_command: List[str]
    run_moderation: bool = True
    run_dedupe: bool = True


def _default_output_dir(provided: Path | None) -> Path:
    if provided is not None:
        return provided
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("artifacts") / f"run-{timestamp}"


def _ensure_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def _run_module(command: List[str]) -> None:
    subprocess.run(command, check=True)


def build_pipeline_config(
    output_dir: str | None,
    generator_args: str,
    moderation_args: str,
    dedupe_args: str,
    skip_moderation: bool,
    skip_dedupe: bool,
) -> PipelineConfig:
    base_dir = _default_output_dir(Path(output_dir) if output_dir else None)
    _ensure_directory(base_dir)

    generated_path = base_dir / "generated.jsonl"
    moderated_path = base_dir / "moderated.jsonl"
    flagged_path = base_dir / "flagged.jsonl"
    failed_path = base_dir / "moderation_failed.jsonl"
    deduped_path = base_dir / "deduped.jsonl"

    generator_args_tokens = shlex.split(generator_args)
    has_total_generations = any(
        token == "--total_generations" or token.startswith("--total_generations=")
        for token in generator_args_tokens
    )
    if not has_total_generations:
        generator_args_tokens.extend(["--total_generations", "10"])

    generator_tokens = [
        sys.executable,
        "-m",
        "synwoz",
        "gen-parallel",
        "--",
        "--output_file",
        str(generated_path),
    ] + generator_args_tokens

    moderation_tokens = [
        sys.executable,
        "-m",
        "synwoz",
        "moderate",
        "--",
        str(generated_path),
        "--output",
        str(moderated_path),
        "--flagged-output",
        str(flagged_path),
        "--failed-output",
        str(failed_path),
    ] + shlex.split(moderation_args)

    dedupe_input = moderated_path if not skip_moderation else generated_path
    dedupe_tokens = [
        sys.executable,
        "-m",
        "synwoz",
        "post-embed-dedup",
        "--",
        str(dedupe_input),
        str(deduped_path),
    ] + shlex.split(dedupe_args)

    return PipelineConfig(
        output_dir=base_dir,
        generator_command=generator_tokens,
        moderation_command=moderation_tokens,
        dedupe_command=dedupe_tokens,
        run_moderation=not skip_moderation,
        run_dedupe=not skip_dedupe,
    )


def run_pipeline(config: PipelineConfig) -> None:
    """Execute the configured pipeline sequentially."""

    _run_module(config.generator_command)

    if config.run_moderation:
        _run_module(config.moderation_command)

    if config.run_dedupe:
        _run_module(config.dedupe_command)

    print(f"Pipeline artifacts stored in {config.output_dir}")


__all__ = ["PipelineConfig", "build_pipeline_config", "run_pipeline"]
