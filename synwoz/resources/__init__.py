"""Resource management helpers for SynWOZ."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from synwoz.resources.setup_helpers import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_PERSONA_NAME,
    DEFAULT_PERSONA_PATH,
    setup_environment,
)


@dataclass
class ResourceSyncConfig:
    model_name: str = DEFAULT_MODEL_NAME
    model_path: Path = field(default_factory=lambda: Path(DEFAULT_MODEL_PATH))
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_path: Path = field(default_factory=lambda: Path(DEFAULT_DATASET_PATH))
    persona_dataset_name: Optional[str] = DEFAULT_PERSONA_NAME
    persona_dataset_path: Optional[Path] = field(default_factory=lambda: Path(DEFAULT_PERSONA_PATH))
    force: bool = False


def sync_resources(config: ResourceSyncConfig) -> None:
    setup_environment(
        model_name=config.model_name,
        model_path=str(config.model_path),
        dataset_name=config.dataset_name,
        dataset_path=str(config.dataset_path),
        persona_dataset_name=config.persona_dataset_name,
        persona_dataset_path=str(config.persona_dataset_path) if config.persona_dataset_path else None,
        force=config.force,
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download models and datasets used by SynWOZ")
    parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME, help='SentenceTransformer model identifier')
    parser.add_argument('--model-path', default=str(DEFAULT_MODEL_PATH), help='Directory to store the SentenceTransformer model')
    parser.add_argument('--dataset-name', default=DEFAULT_DATASET_NAME, help='Primary dataset name on Hugging Face')
    parser.add_argument('--dataset-path', default=str(DEFAULT_DATASET_PATH), help='Directory to store the primary dataset')
    parser.add_argument('--persona-dataset-name', default=DEFAULT_PERSONA_NAME, help='Persona dataset name on Hugging Face')
    parser.add_argument('--persona-dataset-path', default=str(DEFAULT_PERSONA_PATH), help='Directory to store the persona dataset')
    parser.add_argument('--skip-persona', action='store_true', help='Skip downloading the persona dataset')
    parser.add_argument('--force', action='store_true', help='Redownload resources even if they already exist')
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = ResourceSyncConfig(
        model_name=args.model_name,
        model_path=Path(args.model_path),
        dataset_name=args.dataset_name,
        dataset_path=Path(args.dataset_path),
        persona_dataset_name=None if args.skip_persona else args.persona_dataset_name,
        persona_dataset_path=None if args.skip_persona else Path(args.persona_dataset_path),
        force=args.force,
    )
    sync_resources(config)


__all__ = [
    'ResourceSyncConfig',
    'sync_resources',
    'parse_args',
    'main',
]
