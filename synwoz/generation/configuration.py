"""Structured configuration helpers for the generation modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple


@dataclass
class GenerationConfig:
    output_file: str
    hash_file: str
    embedding_file: str
    similarity_threshold: float
    dataset_name: str
    min_turns_range: Tuple[int, int]
    max_turns_range: Tuple[int, int]
    temperature_range: Tuple[float, float]
    top_p_range: Tuple[float, float]
    frequency_penalty_range: Tuple[float, float]
    presence_penalty_range: Tuple[float, float]
    total_generations: int
    persona_dataset_path: str = './local_datasets/FinePersonas-v0.1-clustering-100k'
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Convert the dataclass into the dict expected by DialogueGenerator."""
        base = {
            'output_file': self.output_file,
            'hash_file': self.hash_file,
            'embedding_file': self.embedding_file,
            'similarity_threshold': self.similarity_threshold,
            'dataset_name': self.dataset_name,
            'min_turns_range': self.min_turns_range,
            'max_turns_range': self.max_turns_range,
            'temperature_range': self.temperature_range,
            'top_p_range': self.top_p_range,
            'frequency_penalty_range': self.frequency_penalty_range,
            'presence_penalty_range': self.presence_penalty_range,
            'total_generations': self.total_generations,
            'persona_dataset_path': self.persona_dataset_path,
        }
        base.update(self.extra)
        return base

    @classmethod
    def from_args(cls, args) -> "GenerationConfig":
        def _range(values: Iterable[float]) -> Tuple[float, float]:
            values = list(values)
            return (min(values), max(values))

        persona_path = getattr(args, 'persona_dataset_path', cls.persona_dataset_path)

        return cls(
            output_file=args.output_file,
            hash_file=args.hash_file,
            embedding_file=args.embedding_file,
            similarity_threshold=args.similarity_threshold,
            dataset_name=args.dataset_name,
            min_turns_range=tuple(args.min_turns),
            max_turns_range=tuple(args.max_turns),
            temperature_range=_range(args.temperature),
            top_p_range=_range(args.top_p),
            frequency_penalty_range=_range(args.frequency_penalty),
            presence_penalty_range=_range(args.presence_penalty),
            total_generations=args.total_generations,
            persona_dataset_path=persona_path,
        )


__all__ = ["GenerationConfig"]
