"""Persona dataset helpers for dialogue generation."""

from __future__ import annotations

import ast
import logging
import random
from collections import defaultdict
from typing import Tuple

from datasets import load_from_disk


class PersonaManager:
    """Manages persona sampling using pre-clustered persona datasets."""

    def __init__(self, dataset_path: str, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        dataset = load_from_disk(dataset_path)
        first_split = next(iter(dataset.keys()))
        self._persona_dataset = dataset[first_split]
        self._cluster_index = defaultdict(list)
        self._summary_index = defaultdict(list)
        self._populate_indices()

    def _populate_indices(self) -> None:
        for idx, (cluster, summary) in enumerate(
            zip(self._persona_dataset['cluster_label'], self._persona_dataset['summary_label'])
        ):
            self._cluster_index[cluster].append(idx)
            summary_labels = self._normalize_summary(summary)
            self._summary_index[summary_labels].append(idx)

    def _normalize_summary(self, label) -> Tuple:
        parsed = self._safe_eval(label)
        if isinstance(parsed, list):
            return tuple(sorted(parsed))
        if isinstance(parsed, str):
            return (label,)
        return tuple()

    @staticmethod
    def _safe_eval(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

    def select_random_persona(self) -> str:
        """Return a random persona string using cluster/summary indices."""
        if random.choice([True, False]):
            cluster = random.choice(list(self._cluster_index.keys()))
            persona_idx = random.choice(self._cluster_index[cluster])
        else:
            summary = random.choice(list(self._summary_index.keys()))
            persona_idx = random.choice(self._summary_index[summary])

        return self._persona_dataset[persona_idx]['persona']


__all__ = ["PersonaManager"]
