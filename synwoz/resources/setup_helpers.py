"""Utilities for downloading models and datasets used by SynWOZ."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH = Path("./models/sentence_transformer")
DEFAULT_DATASET_NAME = "pfb30/multi_woz_v22"
DEFAULT_DATASET_PATH = Path("./local_datasets/multi_woz_v22")
DEFAULT_PERSONA_NAME = "argilla/FinePersonas-v0.1-clustering-100k"
DEFAULT_PERSONA_PATH = Path("./local_datasets/FinePersonas-v0.1-clustering-100k")
DEFAULT_SYNWOZ_NAME = "Ayushnangia/SynWOZ"
DEFAULT_SYNWOZ_PATH = Path("./local_datasets/SynWOZ")


def download_sentence_transformer(
    model_name: str = DEFAULT_MODEL_NAME,
    save_path: Path | str = DEFAULT_MODEL_PATH,
) -> Path:
    """Download a SentenceTransformer model if not already cached."""
    destination = Path(save_path)
    destination.mkdir(parents=True, exist_ok=True)

    if (destination / "config.json").exists():
        print(f"Model already exists at {destination}. Skipping download.")
        return destination

    print(f"Downloading SentenceTransformer model '{model_name}' → {destination}")
    model = SentenceTransformer(model_name)
    model.save(str(destination))
    print("Model successfully cached.")
    return destination


def download_and_save_dataset(
    dataset_name: str,
    save_path: Path | str,
    force: bool = False,
) -> Path:
    """Download a Hugging Face dataset and persist it with save_to_disk."""
    destination = Path(save_path)
    if destination.exists() and not force:
        print(f"Dataset already exists at {destination}. Skipping download.")
        return destination

    print(f"Downloading dataset '{dataset_name}'")
    download_mode = "force_redownload" if force else None
    dataset = load_dataset(dataset_name, trust_remote_code=True, download_mode=download_mode)

    destination.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(destination)
    print(f"Dataset saved to {destination}")
    return destination


def setup_environment(
    model_name: str = DEFAULT_MODEL_NAME,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    persona_dataset_name: Optional[str] = DEFAULT_PERSONA_NAME,
    persona_dataset_path: Optional[Path | str] = DEFAULT_PERSONA_PATH,
    synwoz_dataset_name: Optional[str] = DEFAULT_SYNWOZ_NAME,
    synwoz_dataset_path: Optional[Path | str] = DEFAULT_SYNWOZ_PATH,
    force: bool = False,
) -> None:
    """Download core resources needed for generation."""
    model_dir = download_sentence_transformer(model_name, model_path)

    try:
        SentenceTransformer(str(model_dir))
        print("SentenceTransformer model loaded successfully.")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"Warning: could not reload model from {model_dir}: {exc}")

    download_and_save_dataset(dataset_name, dataset_path, force=force)

    if persona_dataset_name and persona_dataset_path:
        download_and_save_dataset(persona_dataset_name, persona_dataset_path, force=force)

    if synwoz_dataset_name and synwoz_dataset_path:
        download_and_save_dataset(synwoz_dataset_name, synwoz_dataset_path, force=force)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download SynWOZ resources")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--persona-dataset-name", default=DEFAULT_PERSONA_NAME)
    parser.add_argument("--persona-dataset-path", default=str(DEFAULT_PERSONA_PATH))
    parser.add_argument("--synwoz-dataset-name", default=DEFAULT_SYNWOZ_NAME)
    parser.add_argument("--synwoz-dataset-path", default=str(DEFAULT_SYNWOZ_PATH))
    parser.add_argument("--skip-persona", action="store_true")
    parser.add_argument("--skip-synwoz", action="store_true", help="Skip downloading the SynWOZ dataset")
    parser.add_argument("--force", action="store_true", help="Redownload datasets even if they exist")
    return parser


def parse_arguments(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.skip_persona:
        args.persona_dataset_name = None
        args.persona_dataset_path = None
    if args.skip_synwoz:
        args.synwoz_dataset_name = None
        args.synwoz_dataset_path = None
    return args


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_DATASET_NAME",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_PERSONA_NAME",
    "DEFAULT_PERSONA_PATH",
    "DEFAULT_SYNWOZ_NAME",
    "DEFAULT_SYNWOZ_PATH",
    "download_sentence_transformer",
    "download_and_save_dataset",
    "setup_environment",
    "build_arg_parser",
    "parse_arguments",
]
