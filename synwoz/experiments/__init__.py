"""
SynWOZ Experiments Module

Provides ablation study framework for evaluating the contribution of
individual pipeline components:
- Deduplication (exact + semantic)
- Safety validation (moderation)
- Diversity sampling (stratified emotions, regions, services)
"""

from .ablation_config import AblationConfig, ABLATION_CONFIGS
from .ablation_runner import AblationRunner, run_ablation_study

__all__ = [
    'AblationConfig',
    'ABLATION_CONFIGS',
    'AblationRunner',
    'run_ablation_study',
]
