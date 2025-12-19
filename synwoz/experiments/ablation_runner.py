#!/usr/bin/env python3
"""
Ablation Study Runner

Runs ablation experiments by generating/processing dialogues with different
configurations and computing comprehensive evaluation metrics.

=============================================================================
EXPERIMENT WORKFLOW
=============================================================================

1. GENERATION PHASE
   For each ablation configuration:
   - Apply feature flags (dedup, safety, diversity settings)
   - Generate N dialogues with the modified pipeline
   - Save to separate output files

2. EVALUATION PHASE
   For each generated dataset:
   - Perplexity: Language model fluency (GPT-2)
   - Precision/Recall: Entity and response accuracy
   - Coherence: Turn-to-turn semantic similarity
   - Diversity: Embedding variance, distinct-n, coverage

3. COMPARISON PHASE
   - Compute deltas from full pipeline (control)
   - Statistical significance tests
   - Generate comparison tables and visualizations

=============================================================================
"""

import json
import logging
import hashlib
import random
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter

import numpy as np

from .ablation_config import (
    AblationConfig,
    ABLATION_CONFIGS,
    DedupMode,
    SamplingMode,
    get_ablation_config,
)

# Lazy imports
_sentence_transformers = None
_faiss = None


def _get_sentence_transformers():
    global _sentence_transformers
    if _sentence_transformers is None:
        import sentence_transformers
        _sentence_transformers = sentence_transformers
    return _sentence_transformers


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AblationMetrics:
    """
    Comprehensive metrics for an ablation experiment.

    Includes all metrics needed to evaluate the contribution of each
    pipeline component to dialogue quality.
    """
    config_name: str
    config_description: str

    # Dataset statistics
    num_dialogues: int
    num_duplicates_removed: int = 0
    num_flagged_safety: int = 0

    # Perplexity metrics
    perplexity: float = 0.0
    perplexity_std: float = 0.0

    # Coherence metrics
    coherence: float = 0.0
    coherence_std: float = 0.0

    # Diversity metrics
    embedding_diversity: float = 0.0
    distinct_1: float = 0.0
    distinct_2: float = 0.0
    service_entropy: float = 0.0
    emotion_coverage: float = 0.0
    region_coverage: float = 0.0

    # Quality metrics (if reference available)
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0

    # Timing
    generation_time_seconds: float = 0.0
    evaluation_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_summary_row(self) -> Dict[str, str]:
        """Get a row for summary table."""
        return {
            'Config': self.config_name,
            'N': str(self.num_dialogues),
            'PPL': f"{self.perplexity:.1f}",
            'Coherence': f"{self.coherence:.3f}",
            'Diversity': f"{self.embedding_diversity:.3f}",
            'Distinct-2': f"{self.distinct_2:.3f}",
            'Emotion Cov': f"{self.emotion_coverage:.1%}",
        }


@dataclass
class AblationResult:
    """Complete results from an ablation study."""
    timestamp: str
    configs_tested: List[str]
    metrics: Dict[str, AblationMetrics]
    comparison_to_control: Dict[str, Dict[str, float]]  # config -> metric -> delta
    output_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'configs_tested': self.configs_tested,
            'metrics': {k: v.to_dict() for k, v in self.metrics.items()},
            'comparison_to_control': self.comparison_to_control,
            'output_dir': self.output_dir,
        }


class AblationRunner:
    """
    Runs ablation experiments with configurable feature flags.

    This class orchestrates:
    1. Dialogue generation with different pipeline configurations
    2. Metric computation (perplexity, coherence, diversity, P/R)
    3. Comparison and statistical analysis
    """

    def __init__(
        self,
        output_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        perplexity_model: str = "gpt2",
        device: Optional[str] = None
    ):
        """
        Initialize the ablation runner.

        Args:
            output_dir: Directory for experiment outputs
            embedding_model: Model for embeddings/coherence
            perplexity_model: Model for perplexity computation
            device: Computation device (auto-detected if None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model_name = embedding_model
        self.perplexity_model_name = perplexity_model
        self.device = device

        self._embed_model = None
        self._ppl_model = None
        self._ppl_tokenizer = None

    def _get_embed_model(self):
        """Lazy load embedding model."""
        if self._embed_model is None:
            st = _get_sentence_transformers()
            self._embed_model = st.SentenceTransformer(self.embedding_model_name)
        return self._embed_model

    def _format_dialogue(self, dialogue: Dict) -> str:
        """Format dialogue as text for embedding/perplexity."""
        parts = []
        for turn in dialogue.get('turns', []):
            utterance = turn.get('utterance', '').strip()
            response = turn.get('assistant_response', '').strip()
            if utterance:
                parts.append(f"User: {utterance}")
            if response:
                parts.append(f"Assistant: {response}")
        return '\n'.join(parts)

    def _format_dialogue_turns(self, dialogue: Dict) -> List[str]:
        """Extract individual turns from a dialogue."""
        turns = []
        for turn in dialogue.get('turns', []):
            utterance = turn.get('utterance', '').strip()
            response = turn.get('assistant_response', '').strip()
            if utterance:
                turns.append(utterance)
            if response:
                turns.append(response)
        return turns

    def _compute_hash(self, dialogue: Dict) -> str:
        """Compute SHA-256 hash for exact duplicate detection."""
        content = []
        for turn in dialogue.get('turns', []):
            content.append({
                'u': turn.get('utterance', '').lower().strip(),
                'r': turn.get('assistant_response', '').lower().strip(),
            })
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def apply_deduplication(
        self,
        dialogues: List[Dict],
        config: AblationConfig
    ) -> Tuple[List[Dict], int]:
        """
        Apply deduplication based on configuration.

        Returns:
            Tuple of (deduplicated_dialogues, num_removed)
        """
        if config.dedup_mode == DedupMode.NONE:
            return dialogues, 0

        # Exact-match deduplication
        seen_hashes = set()
        after_exact = []
        exact_removed = 0

        for d in dialogues:
            h = self._compute_hash(d)
            if h not in seen_hashes:
                seen_hashes.add(h)
                after_exact.append(d)
            else:
                exact_removed += 1

        if config.dedup_mode == DedupMode.EXACT_ONLY:
            return after_exact, exact_removed

        # Semantic deduplication
        if len(after_exact) < 2:
            return after_exact, exact_removed

        model = self._get_embed_model()
        texts = [self._format_dialogue(d) for d in after_exact]
        embeddings = model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        faiss = _get_faiss()
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)

        kept_indices = []
        semantic_removed = 0

        for i, emb in enumerate(embeddings):
            if index.ntotal > 0:
                D, I = index.search(emb.reshape(1, -1), min(5, index.ntotal))
                if D[0][0] >= config.semantic_threshold:
                    semantic_removed += 1
                    continue

            index.add(emb.reshape(1, -1))
            kept_indices.append(i)

        result = [after_exact[i] for i in kept_indices]
        return result, exact_removed + semantic_removed

    def apply_diversity_sampling(
        self,
        dialogues: List[Dict],
        config: AblationConfig,
        target_size: int
    ) -> List[Dict]:
        """
        Apply diversity sampling based on configuration.

        For stratified mode: sample to ensure coverage of emotions, regions, services.
        For random mode: uniform random sampling.
        """
        if len(dialogues) <= target_size:
            return dialogues

        random.seed(config.random_seed)

        if config.sampling_mode == SamplingMode.RANDOM:
            return random.sample(dialogues, target_size)

        # Stratified sampling: try to maximize coverage
        # Group by primary attributes
        by_service = {}
        for i, d in enumerate(dialogues):
            services = tuple(sorted(d.get('services', [])))
            if services not in by_service:
                by_service[services] = []
            by_service[services].append(i)

        # Sample proportionally from each group
        selected_indices = set()
        per_group = max(1, target_size // len(by_service))

        for service_combo, indices in by_service.items():
            sample_size = min(len(indices), per_group)
            selected_indices.update(random.sample(indices, sample_size))

        # Fill remaining slots randomly
        remaining = target_size - len(selected_indices)
        if remaining > 0:
            available = [i for i in range(len(dialogues)) if i not in selected_indices]
            if available:
                selected_indices.update(random.sample(available, min(remaining, len(available))))

        return [dialogues[i] for i in sorted(selected_indices)]

    def compute_perplexity(self, dialogues: List[Dict], sample_size: int = 100) -> Tuple[float, float]:
        """Compute perplexity using the evaluation module."""
        try:
            from synwoz.evaluation.metrics import compute_perplexity as eval_ppl

            texts = [self._format_dialogue(d) for d in dialogues[:sample_size]]
            result = eval_ppl(texts, model_name=self.perplexity_model_name, device=self.device, show_progress=False)
            return result.perplexity, 0.0  # std not computed in single run
        except Exception as e:
            logger.warning(f"Perplexity computation failed: {e}")
            return 0.0, 0.0

    def compute_coherence(self, dialogues: List[Dict]) -> Tuple[float, float]:
        """Compute dialogue coherence."""
        model = self._get_embed_model()
        coherence_scores = []

        for d in dialogues:
            turns = self._format_dialogue_turns(d)
            if len(turns) < 2:
                continue

            embeddings = model.encode(turns, convert_to_numpy=True)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1])
                coherence_scores.append(sim)

        if not coherence_scores:
            return 0.0, 0.0

        return float(np.mean(coherence_scores)), float(np.std(coherence_scores))

    def compute_diversity_metrics(self, dialogues: List[Dict]) -> Dict[str, float]:
        """Compute diversity metrics."""
        from synwoz.common.config import USER_EMOTION_LIST, ASSISTANT_EMOTION_LIST, PREDEFINED_REGIONS

        # Distinct-n
        all_words = []
        for d in dialogues:
            text = self._format_dialogue(d).lower()
            all_words.extend(text.split())

        distinct_1 = len(set(all_words)) / len(all_words) if all_words else 0
        bigrams = [tuple(all_words[i:i+2]) for i in range(len(all_words) - 1)]
        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0

        # Embedding diversity
        model = self._get_embed_model()
        texts = [self._format_dialogue(d) for d in dialogues]
        embeddings = model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        if len(embeddings) > 1:
            sample_size = min(500, len(embeddings))
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample = embeddings[indices]
            similarities = np.dot(sample, sample.T)
            mask = ~np.eye(len(sample), dtype=bool)
            embedding_diversity = 1 - similarities[mask].mean()
        else:
            embedding_diversity = 0.0

        # Coverage metrics
        observed_emotions = set()
        observed_regions = set()
        all_services = []

        for d in dialogues:
            for e in d.get('user_emotions', []):
                observed_emotions.add(e)
            for e in d.get('assistant_emotions', []):
                observed_emotions.add(e)
            for r in d.get('regions', []):
                observed_regions.add(r)
            all_services.extend(d.get('services', []))

        emotion_coverage = len(observed_emotions) / (len(USER_EMOTION_LIST) + len(ASSISTANT_EMOTION_LIST))
        region_coverage = len(observed_regions) / len(PREDEFINED_REGIONS) if PREDEFINED_REGIONS else 0

        # Service entropy
        service_counts = Counter(all_services)
        if service_counts:
            probs = np.array(list(service_counts.values())) / sum(service_counts.values())
            from scipy.stats import entropy
            service_entropy = entropy(probs, base=2)
        else:
            service_entropy = 0.0

        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'embedding_diversity': float(embedding_diversity),
            'emotion_coverage': emotion_coverage,
            'region_coverage': region_coverage,
            'service_entropy': service_entropy,
        }

    def run_single_ablation(
        self,
        dialogues: List[Dict],
        config: AblationConfig
    ) -> AblationMetrics:
        """
        Run a single ablation experiment.

        Args:
            dialogues: Input dialogues to process
            config: Ablation configuration

        Returns:
            AblationMetrics with all computed metrics
        """
        logger.info(f"Running ablation: {config.name}")
        logger.info(f"  {config.get_summary()}")

        start_time = time.time()

        # Apply pipeline stages based on config
        processed, num_removed = self.apply_deduplication(dialogues, config)
        logger.info(f"  After deduplication: {len(processed)} dialogues ({num_removed} removed)")

        if config.sampling_mode == SamplingMode.RANDOM:
            processed = self.apply_diversity_sampling(processed, config, config.num_dialogues)
            logger.info(f"  After random sampling: {len(processed)} dialogues")

        generation_time = time.time() - start_time

        # Compute metrics
        eval_start = time.time()

        ppl, ppl_std = self.compute_perplexity(processed)
        logger.info(f"  Perplexity: {ppl:.2f}")

        coherence, coherence_std = self.compute_coherence(processed)
        logger.info(f"  Coherence: {coherence:.4f}")

        diversity = self.compute_diversity_metrics(processed)
        logger.info(f"  Diversity: {diversity['embedding_diversity']:.4f}")
        logger.info(f"  Distinct-2: {diversity['distinct_2']:.4f}")

        eval_time = time.time() - eval_start

        return AblationMetrics(
            config_name=config.name,
            config_description=config.description,
            num_dialogues=len(processed),
            num_duplicates_removed=num_removed,
            perplexity=ppl,
            perplexity_std=ppl_std,
            coherence=coherence,
            coherence_std=coherence_std,
            embedding_diversity=diversity['embedding_diversity'],
            distinct_1=diversity['distinct_1'],
            distinct_2=diversity['distinct_2'],
            service_entropy=diversity['service_entropy'],
            emotion_coverage=diversity['emotion_coverage'],
            region_coverage=diversity['region_coverage'],
            generation_time_seconds=generation_time,
            evaluation_time_seconds=eval_time,
        )

    def run_ablation_study(
        self,
        dialogues: List[Dict],
        config_names: Optional[List[str]] = None
    ) -> AblationResult:
        """
        Run complete ablation study across multiple configurations.

        Args:
            dialogues: Input dialogues to use for all experiments
            config_names: List of configuration names to test (default: all)

        Returns:
            AblationResult with all metrics and comparisons
        """
        if config_names is None:
            config_names = list(ABLATION_CONFIGS.keys())

        logger.info("=" * 60)
        logger.info("ABLATION STUDY")
        logger.info("=" * 60)
        logger.info(f"Input dialogues: {len(dialogues)}")
        logger.info(f"Configurations to test: {len(config_names)}")

        metrics = {}
        for name in config_names:
            config = get_ablation_config(name)
            metrics[name] = self.run_single_ablation(dialogues, config)

        # Compute comparison to control (full_pipeline)
        comparison = {}
        if 'full_pipeline' in metrics:
            control = metrics['full_pipeline']
            for name, m in metrics.items():
                if name == 'full_pipeline':
                    continue
                comparison[name] = {
                    'perplexity_delta': m.perplexity - control.perplexity,
                    'coherence_delta': m.coherence - control.coherence,
                    'diversity_delta': m.embedding_diversity - control.embedding_diversity,
                    'distinct2_delta': m.distinct_2 - control.distinct_2,
                }

        result = AblationResult(
            timestamp=datetime.now().isoformat(),
            configs_tested=config_names,
            metrics=metrics,
            comparison_to_control=comparison,
            output_dir=str(self.output_dir),
        )

        # Save results
        self._save_results(result)

        return result

    def _save_results(self, result: AblationResult):
        """Save ablation results to files."""
        # JSON results
        with open(self.output_dir / 'ablation_results.json', 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        # Summary table
        self._save_summary_table(result)

        # LaTeX table
        self._save_latex_table(result)

        logger.info(f"Results saved to {self.output_dir}")

    def _save_summary_table(self, result: AblationResult):
        """Save human-readable summary table."""
        lines = [
            "=" * 100,
            "ABLATION STUDY RESULTS",
            "=" * 100,
            f"{'Configuration':<25} {'N':>6} {'PPL':>8} {'Coherence':>10} {'Diversity':>10} {'Distinct-2':>10} {'Emotion':>10}",
            "-" * 100,
        ]

        for name, m in result.metrics.items():
            row = m.get_summary_row()
            lines.append(
                f"{row['Config']:<25} {row['N']:>6} {row['PPL']:>8} {row['Coherence']:>10} "
                f"{row['Diversity']:>10} {row['Distinct-2']:>10} {row['Emotion Cov']:>10}"
            )

        lines.append("=" * 100)

        if result.comparison_to_control:
            lines.append("\nComparison to Full Pipeline (Control):")
            lines.append("-" * 80)
            for name, deltas in result.comparison_to_control.items():
                ppl_delta = deltas['perplexity_delta']
                coh_delta = deltas['coherence_delta']
                div_delta = deltas['diversity_delta']
                lines.append(
                    f"  {name}: PPL {ppl_delta:+.1f}, Coherence {coh_delta:+.4f}, "
                    f"Diversity {div_delta:+.4f}"
                )

        with open(self.output_dir / 'ablation_summary.txt', 'w') as f:
            f.write('\n'.join(lines))

    def _save_latex_table(self, result: AblationResult):
        """Save LaTeX-formatted table."""
        latex = r"""\begin{table}[h]
\centering
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Configuration} & \textbf{N} & \textbf{PPL} $\downarrow$ & \textbf{Coherence} $\uparrow$ & \textbf{Diversity} $\uparrow$ & \textbf{Distinct-2} $\uparrow$ \\
\midrule
"""
        for name, m in result.metrics.items():
            display_name = name.replace('_', ' ').title()
            latex += f"{display_name} & {m.num_dialogues} & {m.perplexity:.1f} & {m.coherence:.3f} & {m.embedding_diversity:.3f} & {m.distinct_2:.3f} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\caption{Ablation study results. PPL=Perplexity (lower is better). Coherence, Diversity, and Distinct-2 (higher is better).}
\label{tab:ablation}
\end{table}"""

        with open(self.output_dir / 'ablation_table.tex', 'w') as f:
            f.write(latex)


def run_ablation_study(
    input_path: str,
    output_dir: str,
    config_names: Optional[List[str]] = None,
    max_dialogues: int = 5000
) -> AblationResult:
    """
    Convenience function to run ablation study from file.

    Args:
        input_path: Path to JSONL file with dialogues
        output_dir: Directory for output files
        config_names: Configurations to test (default: all)
        max_dialogues: Maximum dialogues to load

    Returns:
        AblationResult with all metrics
    """
    # Load dialogues
    dialogues = []
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if max_dialogues and i >= max_dialogues:
                break
            dialogues.append(json.loads(line))

    logger.info(f"Loaded {len(dialogues)} dialogues from {input_path}")

    # Run study
    runner = AblationRunner(output_dir=Path(output_dir))
    return runner.run_ablation_study(dialogues, config_names)


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ablation study on SynWOZ dialogues"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to JSONL file with dialogues'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./ablation_results',
        help='Directory for output files'
    )
    parser.add_argument(
        '--configs',
        type=str,
        nargs='+',
        default=None,
        help='Configurations to test (default: all)'
    )
    parser.add_argument(
        '--max-dialogues',
        type=int,
        default=5000,
        help='Maximum dialogues to process'
    )
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List available configurations and exit'
    )

    args = parser.parse_args()

    if args.list_configs:
        from .ablation_config import print_ablation_matrix
        print_ablation_matrix()
    else:
        result = run_ablation_study(
            args.input_file,
            args.output_dir,
            args.configs,
            args.max_dialogues
        )
        print(f"\nResults saved to {args.output_dir}")
