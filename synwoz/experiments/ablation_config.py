#!/usr/bin/env python3
"""
Ablation Experiment Configuration

Defines feature flags and configurations for ablation studies.
Each configuration enables/disables specific pipeline components
to measure their individual contribution to dialogue quality.

=============================================================================
ABLATION STUDY DESIGN
=============================================================================

We evaluate three key pipeline components:

1. DEDUPLICATION
   - Exact-match: SHA-256 hash comparison
   - Semantic: FAISS embedding similarity (threshold-based)
   - Ablation: disable_deduplication, disable_semantic_dedup

2. SAFETY VALIDATION
   - OpenAI Moderation API for content filtering
   - Flags harmful content (hate, violence, etc.)
   - Ablation: disable_safety_validation

3. DIVERSITY SAMPLING
   - Stratified sampling across emotions, regions, services
   - Weighted service selection (30/40/20/10% for 1/2/3/4 services)
   - Ablation: disable_diversity_sampling (use random instead)

=============================================================================
EXPERIMENTAL CONDITIONS
=============================================================================

| Condition           | Dedup | Semantic | Safety | Diversity |
|---------------------|-------|----------|--------|-----------|
| Full Pipeline       |   ✓   |    ✓     |   ✓    |     ✓     |
| No Deduplication    |   ✗   |    ✗     |   ✓    |     ✓     |
| Exact-Only Dedup    |   ✓   |    ✗     |   ✓    |     ✓     |
| No Safety           |   ✓   |    ✓     |   ✗    |     ✓     |
| No Diversity        |   ✓   |    ✓     |   ✓    |     ✗     |
| Minimal (Baseline)  |   ✗   |    ✗     |   ✗    |     ✗     |

=============================================================================
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class DedupMode(Enum):
    """Deduplication mode options."""
    FULL = "full"           # Both exact-match and semantic
    EXACT_ONLY = "exact"    # Only exact-match (hash-based)
    NONE = "none"           # No deduplication


class SamplingMode(Enum):
    """Diversity sampling mode options."""
    STRATIFIED = "stratified"   # Stratified across emotions, regions, services
    RANDOM = "random"           # Uniform random selection


@dataclass
class AblationConfig:
    """
    Configuration for a single ablation experiment.

    Attributes:
        name: Unique identifier for this configuration
        description: Human-readable description of what's being tested
        dedup_mode: Deduplication strategy (full, exact_only, none)
        semantic_threshold: Similarity threshold for semantic dedup (0.0-1.0)
        enable_safety_validation: Whether to run moderation API
        sampling_mode: Diversity sampling strategy (stratified, random)
        enable_weighted_services: Use weighted service selection (30/40/20/10)
        enable_emotion_diversity: Sample from emotion vocabulary
        enable_region_diversity: Sample from region list
        num_dialogues: Number of dialogues to generate for this experiment
        random_seed: Seed for reproducibility
    """
    name: str
    description: str

    # Deduplication settings
    dedup_mode: DedupMode = DedupMode.FULL
    semantic_threshold: float = 0.9

    # Safety validation
    enable_safety_validation: bool = True

    # Diversity sampling
    sampling_mode: SamplingMode = SamplingMode.STRATIFIED
    enable_weighted_services: bool = True
    enable_emotion_diversity: bool = True
    enable_region_diversity: bool = True

    # Experiment settings
    num_dialogues: int = 100
    random_seed: int = 42

    # Feature flags (computed properties for convenience)
    @property
    def disable_deduplication(self) -> bool:
        return self.dedup_mode == DedupMode.NONE

    @property
    def disable_semantic_dedup(self) -> bool:
        return self.dedup_mode in (DedupMode.NONE, DedupMode.EXACT_ONLY)

    @property
    def disable_safety_validation(self) -> bool:
        return not self.enable_safety_validation

    @property
    def disable_diversity_sampling(self) -> bool:
        return self.sampling_mode == SamplingMode.RANDOM

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['dedup_mode'] = self.dedup_mode.value
        result['sampling_mode'] = self.sampling_mode.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AblationConfig':
        """Create from dictionary."""
        data = data.copy()
        data['dedup_mode'] = DedupMode(data['dedup_mode'])
        data['sampling_mode'] = SamplingMode(data['sampling_mode'])
        return cls(**data)

    def get_summary(self) -> str:
        """Get a short summary of enabled/disabled features."""
        features = []
        if self.dedup_mode == DedupMode.FULL:
            features.append("Dedup:Full")
        elif self.dedup_mode == DedupMode.EXACT_ONLY:
            features.append("Dedup:Exact")
        else:
            features.append("Dedup:OFF")

        features.append("Safety:ON" if self.enable_safety_validation else "Safety:OFF")
        features.append("Diversity:ON" if self.sampling_mode == SamplingMode.STRATIFIED else "Diversity:OFF")

        return " | ".join(features)


# =============================================================================
# PREDEFINED ABLATION CONFIGURATIONS
# =============================================================================

ABLATION_CONFIGS: Dict[str, AblationConfig] = {
    # Full pipeline (control)
    "full_pipeline": AblationConfig(
        name="full_pipeline",
        description="Full pipeline with all features enabled (control condition)",
        dedup_mode=DedupMode.FULL,
        semantic_threshold=0.9,
        enable_safety_validation=True,
        sampling_mode=SamplingMode.STRATIFIED,
        enable_weighted_services=True,
        enable_emotion_diversity=True,
        enable_region_diversity=True,
    ),

    # Deduplication ablations
    "no_deduplication": AblationConfig(
        name="no_deduplication",
        description="Disable all deduplication (both exact-match and semantic)",
        dedup_mode=DedupMode.NONE,
        enable_safety_validation=True,
        sampling_mode=SamplingMode.STRATIFIED,
    ),

    "exact_only_dedup": AblationConfig(
        name="exact_only_dedup",
        description="Exact-match deduplication only (no semantic similarity check)",
        dedup_mode=DedupMode.EXACT_ONLY,
        enable_safety_validation=True,
        sampling_mode=SamplingMode.STRATIFIED,
    ),

    "high_threshold_dedup": AblationConfig(
        name="high_threshold_dedup",
        description="Semantic dedup with higher threshold (0.95) - less aggressive",
        dedup_mode=DedupMode.FULL,
        semantic_threshold=0.95,
        enable_safety_validation=True,
        sampling_mode=SamplingMode.STRATIFIED,
    ),

    "low_threshold_dedup": AblationConfig(
        name="low_threshold_dedup",
        description="Semantic dedup with lower threshold (0.85) - more aggressive",
        dedup_mode=DedupMode.FULL,
        semantic_threshold=0.85,
        enable_safety_validation=True,
        sampling_mode=SamplingMode.STRATIFIED,
    ),

    # Safety validation ablation
    "no_safety_validation": AblationConfig(
        name="no_safety_validation",
        description="Disable safety validation (no moderation filtering)",
        dedup_mode=DedupMode.FULL,
        enable_safety_validation=False,
        sampling_mode=SamplingMode.STRATIFIED,
    ),

    # Diversity sampling ablations
    "no_diversity_sampling": AblationConfig(
        name="no_diversity_sampling",
        description="Random sampling instead of stratified (no diversity controls)",
        dedup_mode=DedupMode.FULL,
        enable_safety_validation=True,
        sampling_mode=SamplingMode.RANDOM,
        enable_weighted_services=False,
        enable_emotion_diversity=False,
        enable_region_diversity=False,
    ),

    "uniform_services": AblationConfig(
        name="uniform_services",
        description="Uniform service selection (not weighted 30/40/20/10)",
        dedup_mode=DedupMode.FULL,
        enable_safety_validation=True,
        sampling_mode=SamplingMode.STRATIFIED,
        enable_weighted_services=False,
        enable_emotion_diversity=True,
        enable_region_diversity=True,
    ),

    # Minimal baseline
    "minimal_baseline": AblationConfig(
        name="minimal_baseline",
        description="Minimal pipeline: no dedup, no safety, random sampling",
        dedup_mode=DedupMode.NONE,
        enable_safety_validation=False,
        sampling_mode=SamplingMode.RANDOM,
        enable_weighted_services=False,
        enable_emotion_diversity=False,
        enable_region_diversity=False,
    ),
}


def get_ablation_config(name: str) -> AblationConfig:
    """Get a predefined ablation configuration by name."""
    if name not in ABLATION_CONFIGS:
        available = ", ".join(ABLATION_CONFIGS.keys())
        raise ValueError(f"Unknown ablation config: {name}. Available: {available}")
    return ABLATION_CONFIGS[name]


def list_ablation_configs() -> List[str]:
    """List all available ablation configuration names."""
    return list(ABLATION_CONFIGS.keys())


def print_ablation_matrix():
    """Print a matrix showing all ablation configurations and their settings."""
    print("\n" + "=" * 80)
    print("ABLATION CONFIGURATION MATRIX")
    print("=" * 80)
    print(f"{'Configuration':<25} {'Dedup':<12} {'Safety':<10} {'Diversity':<12} {'Threshold':<10}")
    print("-" * 80)

    for name, config in ABLATION_CONFIGS.items():
        dedup = config.dedup_mode.value
        safety = "ON" if config.enable_safety_validation else "OFF"
        diversity = config.sampling_mode.value
        threshold = f"{config.semantic_threshold:.2f}" if config.dedup_mode == DedupMode.FULL else "N/A"
        print(f"{name:<25} {dedup:<12} {safety:<10} {diversity:<12} {threshold:<10}")

    print("=" * 80)


if __name__ == "__main__":
    print_ablation_matrix()

    print("\nConfiguration Details:")
    for name, config in ABLATION_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Description: {config.description}")
        print(f"  Summary: {config.get_summary()}")
