"""
SynWOZ Evaluation Module

This module provides comprehensive evaluation metrics for task-oriented dialogues:
- Perplexity (language model fluency)
- Precision/Recall (entity and response matching)
- Coherence (dialogue flow quality)
"""

from .metrics import (
    compute_perplexity,
    compute_entity_precision_recall,
    compute_response_precision_recall,
    compute_coherence,
    DialogueEvaluator,
)

__all__ = [
    'compute_perplexity',
    'compute_entity_precision_recall',
    'compute_response_precision_recall',
    'compute_coherence',
    'DialogueEvaluator',
]
