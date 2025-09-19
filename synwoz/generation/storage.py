"""Shared persistence utilities for dialogue generation."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import numpy as np

_logger = logging.getLogger(__name__)


# JSONL helpers ------------------------------------------------------------

def load_dialogue_ids_jsonl(path: str | Path, logger: logging.Logger | None = None) -> Set[str]:
    path = Path(path)
    logger = logger or _logger
    existing_ids: Set[str] = set()
    if not path.exists():
        return existing_ids

    try:
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                dialogue = json.loads(line)
                dialogue_id = dialogue.get('dialogue_id')
                if dialogue_id:
                    existing_ids.add(dialogue_id)
        logger.info("Loaded %d existing dialogues from '%s'.", len(existing_ids), path)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.warning("Could not load existing dialogues from '%s': %s", path, exc)
    return existing_ids


def append_dialogues_jsonl(path: str | Path, dialogues: Sequence[dict], logger: logging.Logger | None = None) -> None:
    path = Path(path)
    logger = logger or _logger
    if not dialogues:
        return
    try:
        with path.open('a', encoding='utf-8') as handle:
            for dialogue in dialogues:
                handle.write(json.dumps(dialogue, ensure_ascii=False) + '\n')
        logger.info("Appended %d dialogues to '%s'.", len(dialogues), path)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.error("Failed to append dialogues to '%s': %s", path, exc)


# JSON helpers -------------------------------------------------------------

def load_dialogues_json(path: str | Path, logger: logging.Logger | None = None) -> List[dict]:
    path = Path(path)
    logger = logger or _logger
    if not path.exists():
        return []
    try:
        with path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
        logger.info("Loaded %d dialogues from '%s'.", len(data), path)
        return data
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.warning("Could not load dialogues from '%s': %s", path, exc)
        return []


def save_dialogues_json(path: str | Path, dialogues: Sequence[dict], logger: logging.Logger | None = None) -> None:
    path = Path(path)
    logger = logger or _logger
    try:
        with path.open('w', encoding='utf-8') as handle:
            json.dump(list(dialogues), handle, indent=4, ensure_ascii=False)
        logger.info("Saved %d dialogues to '%s'.", len(dialogues), path)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.error("Failed to save dialogues to '%s': %s", path, exc)


# Hash helpers -------------------------------------------------------------

def load_hashes_jsonl(path: str | Path, logger: logging.Logger | None = None) -> Set[str]:
    path = Path(path)
    logger = logger or _logger
    hashes: Set[str] = set()
    if not path.exists():
        return hashes
    try:
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                value = entry.get('hash')
                if value:
                    hashes.add(value)
        logger.info("Loaded %d hashes from '%s'.", len(hashes), path)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.warning("Could not load hashes from '%s': %s", path, exc)
    return hashes


def load_hashes_json(path: str | Path, logger: logging.Logger | None = None) -> Set[str]:
    path = Path(path)
    logger = logger or _logger
    if not path.exists():
        return set()
    try:
        with path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
        hashes = set(data if isinstance(data, list) else [])
        logger.info("Loaded %d hashes from '%s'.", len(hashes), path)
        return hashes
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.warning("Could not load hashes from '%s': %s", path, exc)
        return set()


def append_hashes_jsonl(path: str | Path, hashes: Iterable[str], logger: logging.Logger | None = None) -> None:
    path = Path(path)
    logger = logger or _logger
    hash_list = list(hashes)
    if not hash_list:
        return
    try:
        with path.open('a', encoding='utf-8') as handle:
            for item in hash_list:
                handle.write(json.dumps({'hash': item}) + '\n')
        logger.info("Appended %d hashes to '%s'.", len(hash_list), path)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.error("Failed to append hashes to '%s': %s", path, exc)


def save_hashes_json(path: str | Path, hashes: Iterable[str], logger: logging.Logger | None = None) -> None:
    path = Path(path)
    logger = logger or _logger
    hash_list = list(hashes)
    try:
        with path.open('w', encoding='utf-8') as handle:
            json.dump(hash_list, handle, indent=4)
        logger.info("Saved %d hashes to '%s'.", len(hash_list), path)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.error("Failed to save hashes to '%s': %s", path, exc)


# Embedding helpers --------------------------------------------------------

def load_embeddings(path: str | Path, logger: logging.Logger | None = None) -> np.ndarray:
    path = Path(path)
    logger = logger or _logger
    if not path.exists():
        return np.array([])
    try:
        embeddings = np.load(path)
        logger.info("Loaded %d embeddings from '%s'.", embeddings.shape[0], path)
        return embeddings
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.warning("Could not load embeddings from '%s': %s", path, exc)
        return np.array([])


def save_embeddings_atomic(path: str | Path, embeddings: np.ndarray, logger: logging.Logger | None = None) -> None:
    path = Path(path)
    logger = logger or _logger
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = Path(tmp.name)
        np.save(temp_path, embeddings)
        os.replace(temp_path, path)
        logger.info("Saved %d embeddings to '%s'.", embeddings.shape[0], path)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.error("Failed to save embeddings to '%s': %s", path, exc)


__all__ = [
    'append_dialogues_jsonl',
    'append_hashes_jsonl',
    'load_dialogue_ids_jsonl',
    'load_dialogues_json',
    'load_embeddings',
    'load_hashes_json',
    'load_hashes_jsonl',
    'save_dialogues_json',
    'save_embeddings_atomic',
    'save_hashes_json',
]
