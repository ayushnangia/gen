#!/usr/bin/env python3
"""
Ablation experiments for ACL SRW paper.
Evaluates contributions of each pipeline component:
- RQ1: Semantic deduplication vs exact-match only
- RQ2: Multi-stage safety validation vs single-stage
- RQ3: Stratified sampling vs random sampling
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import hashlib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATASET_PATH = "./SynWOZ-dataset/dataset.jsonl"
OUTPUT_DIR = Path("./ablation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Expected distributions from config.py
EXPECTED_REGIONS = [
    "Tokyo", "Delhi", "Shanghai", "Sao Paulo", "Mumbai", "Beijing", "Cairo",
    "Mexico City", "Dhaka", "Osaka", "Karachi", "Chongqing", "Istanbul",
    "Buenos Aires", "Kolkata", "Kinshasa", "Lagos", "Manila", "Rio de Janeiro",
    "Guangzhou", "Los Angeles", "Moscow", "Paris", "Bangkok", "Jakarta",
    "London", "Lima", "New York", "Shenzhen", "Bangalore", "Ho Chi Minh City",
    "Hyderabad", "Bogota", "Tianjin", "Santiago", "Sydney", "Berlin", "Madrid",
    "Toronto", "Johannesburg", "Dubai", "Singapore", "Tehran", "Baghdad",
    "Riyadh", "Rome", "Cape Town", "Casablanca", "Barcelona", "Seoul",
    "Melbourne", "Copenhagen", "Zurich", "Kuala Lumpur",
]

EXPECTED_USER_EMOTIONS = [
    "Frustrated", "Angry", "Confused", "Worried", "Disappointed",
    "Happy", "Anxious", "Impatient", "Skeptical", "Desperate",
    "Overwhelmed", "Hopeful", "Satisfied", "Stressed", "Suspicious",
    "Tired", "Excited", "Indifferent", "Grateful", "Demanding",
]

EXPECTED_ASSISTANT_EMOTIONS = [
    "Professional", "Informative", "Reassuring", "Diplomatic", "Patient",
    "Efficient", "Accommodating", "Solution-focused", "Methodical", "Proactive",
    "Analytical", "Composed", "Detail-oriented", "Responsive", "Thorough",
    "Systematic", "Precise", "Objective", "Resourceful", "Knowledgeable",
]


def load_dataset(path: str) -> List[Dict]:
    """Load JSONL dataset."""
    dialogues = []
    with open(path, 'r') as f:
        for line in f:
            dialogues.append(json.loads(line))
    return dialogues


def get_dialogue_text(dialogue: Dict) -> str:
    """Extract full text from dialogue for embedding."""
    text_parts = []
    if 'generated_scenario' in dialogue:
        text_parts.append(dialogue['generated_scenario'])
    for turn in dialogue.get('turns', []):
        utterance = turn.get('utterance', '').strip()
        intent = turn.get('intent', '').strip()
        response = turn.get('assistant_response', '').strip()
        if utterance:
            text_parts.append(f"User: {utterance}")
        if intent:
            text_parts.append(f"Intent: {intent}")
        if response:
            text_parts.append(f"Assistant: {response}")
    return ' '.join(text_parts)


def compute_dialogue_hash(dialogue: Dict) -> str:
    """Compute content hash for exact duplicate detection."""
    dialogue_content = []
    for turn in dialogue.get('turns', []):
        content = {
            'utterance': turn.get('utterance', '').lower().strip(),
            'intent': turn.get('intent', '').lower().strip(),
            'response': turn.get('assistant_response', '').lower().strip()
        }
        dialogue_content.append(content)
    content_str = json.dumps(dialogue_content, sort_keys=True)
    return hashlib.md5(content_str.encode()).hexdigest()


def compute_diversity_metrics(dialogues: List[Dict], embeddings: np.ndarray = None) -> Dict:
    """Compute diversity metrics for a set of dialogues."""
    # Unique services
    all_services = []
    for d in dialogues:
        all_services.extend(d.get('services', []))
    service_counts = Counter(all_services)

    # Service entropy
    service_probs = np.array(list(service_counts.values())) / sum(service_counts.values())
    service_entropy = entropy(service_probs, base=2)

    # Unique intents
    all_intents = set()
    for d in dialogues:
        for turn in d.get('turns', []):
            intent = turn.get('intent', '').strip()
            if intent:
                all_intents.add(intent.lower())

    # Embedding diversity (average pairwise distance if embeddings provided)
    embedding_diversity = None
    if embeddings is not None and len(embeddings) > 1:
        # Sample for efficiency
        sample_size = min(1000, len(embeddings))
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
        similarities = cosine_similarity(sample_embeddings)
        # Average of non-diagonal elements
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        embedding_diversity = 1 - similarities[mask].mean()  # Convert similarity to diversity

    return {
        'num_dialogues': len(dialogues),
        'unique_services': len(service_counts),
        'service_entropy': service_entropy,
        'unique_intents': len(all_intents),
        'embedding_diversity': embedding_diversity
    }


def ablation_deduplication(dialogues: List[Dict], model: SentenceTransformer) -> Dict:
    """
    RQ1: Compare semantic deduplication vs exact-match only.
    """
    print("\n=== RQ1: Deduplication Ablation ===")

    # Generate embeddings
    print("Generating embeddings...")
    texts = [get_dialogue_text(d) for d in dialogues]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Exact-match deduplication
    print("Computing exact-match duplicates...")
    hashes = [compute_dialogue_hash(d) for d in dialogues]
    hash_to_idx = {}
    exact_duplicates = set()
    for idx, h in enumerate(hashes):
        if h in hash_to_idx:
            exact_duplicates.add(idx)
        else:
            hash_to_idx[h] = idx

    # Semantic deduplication (similarity >= 0.9)
    print("Computing semantic near-matches...")
    semantic_duplicates = set()
    threshold = 0.9

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Process in batches for efficiency
    batch_size = 1000
    for i in range(0, len(dialogues), batch_size):
        batch_end = min(i + batch_size, len(dialogues))
        batch_embeddings = normalized_embeddings[i:batch_end]

        # Compare with all previous embeddings
        if i > 0:
            similarities = np.dot(batch_embeddings, normalized_embeddings[:i].T)
            for j in range(batch_embeddings.shape[0]):
                if similarities[j].max() >= threshold:
                    semantic_duplicates.add(i + j)

        # Compare within batch
        if batch_end - i > 1:
            batch_similarities = np.dot(batch_embeddings, batch_embeddings.T)
            for j in range(batch_similarities.shape[0]):
                for k in range(j + 1, batch_similarities.shape[1]):
                    if batch_similarities[j, k] >= threshold:
                        semantic_duplicates.add(i + k)

    # Compute metrics for each configuration
    all_indices = set(range(len(dialogues)))

    # Full pipeline (exact + semantic)
    full_duplicates = exact_duplicates | semantic_duplicates
    full_indices = list(all_indices - full_duplicates)
    full_dialogues = [dialogues[i] for i in full_indices]
    full_embeddings = embeddings[full_indices]

    # Exact-only
    exact_indices = list(all_indices - exact_duplicates)
    exact_dialogues = [dialogues[i] for i in exact_indices]
    exact_embeddings = embeddings[exact_indices]

    # No deduplication
    no_dedup_dialogues = dialogues
    no_dedup_embeddings = embeddings

    results = {
        'full_pipeline': {
            'removed': len(full_duplicates),
            'remaining': len(full_dialogues),
            **compute_diversity_metrics(full_dialogues, full_embeddings)
        },
        'exact_only': {
            'removed': len(exact_duplicates),
            'remaining': len(exact_dialogues),
            **compute_diversity_metrics(exact_dialogues, exact_embeddings)
        },
        'no_deduplication': {
            'removed': 0,
            'remaining': len(no_dedup_dialogues),
            **compute_diversity_metrics(no_dedup_dialogues, no_dedup_embeddings)
        }
    }

    print(f"\nResults:")
    print(f"  Full pipeline: {results['full_pipeline']['removed']} removed, diversity={results['full_pipeline']['embedding_diversity']:.4f}")
    print(f"  Exact-only: {results['exact_only']['removed']} removed, diversity={results['exact_only']['embedding_diversity']:.4f}")
    print(f"  No dedup: {results['no_deduplication']['removed']} removed, diversity={results['no_deduplication']['embedding_diversity']:.4f}")

    return results


def ablation_sampling(dialogues: List[Dict]) -> Dict:
    """
    RQ3: Compare stratified sampling vs random sampling.
    Measures coverage of target distributions.
    """
    print("\n=== RQ3: Sampling Ablation ===")

    # Extract actual distributions
    regions = []
    user_emotions = []
    assistant_emotions = []

    for d in dialogues:
        # Regions
        for r in d.get('regions', []):
            regions.append(r)
        # Emotions
        for e in d.get('user_emotions', []):
            user_emotions.append(e)
        for e in d.get('assistant_emotions', []):
            assistant_emotions.append(e)

    region_counts = Counter(regions)
    user_emotion_counts = Counter(user_emotions)
    assistant_emotion_counts = Counter(assistant_emotions)

    # Coverage metrics
    region_coverage = len(region_counts) / len(EXPECTED_REGIONS)
    user_emotion_coverage = len(user_emotion_counts) / len(EXPECTED_USER_EMOTIONS)
    assistant_emotion_coverage = len(assistant_emotion_counts) / len(EXPECTED_ASSISTANT_EMOTIONS)

    # Distribution entropy (higher = more uniform)
    region_probs = np.array([region_counts.get(r, 0) for r in EXPECTED_REGIONS])
    region_probs = region_probs / region_probs.sum() if region_probs.sum() > 0 else region_probs
    region_entropy = entropy(region_probs + 1e-10, base=2)
    max_region_entropy = np.log2(len(EXPECTED_REGIONS))

    user_emotion_probs = np.array([user_emotion_counts.get(e, 0) for e in EXPECTED_USER_EMOTIONS])
    user_emotion_probs = user_emotion_probs / user_emotion_probs.sum() if user_emotion_probs.sum() > 0 else user_emotion_probs
    user_emotion_entropy = entropy(user_emotion_probs + 1e-10, base=2)
    max_emotion_entropy = np.log2(len(EXPECTED_USER_EMOTIONS))

    # Simulate random sampling baseline (theoretical)
    # Random sampling with same total would have different coverage
    # We estimate based on coupon collector problem
    n_samples = len(regions)
    n_categories = len(EXPECTED_REGIONS)
    # Expected coverage for random sampling: 1 - (1 - 1/n)^samples
    expected_random_coverage = 1 - ((1 - 1/n_categories) ** (n_samples / n_categories))

    results = {
        'stratified': {
            'region_coverage': region_coverage,
            'user_emotion_coverage': user_emotion_coverage,
            'assistant_emotion_coverage': assistant_emotion_coverage,
            'region_entropy': region_entropy,
            'region_entropy_ratio': region_entropy / max_region_entropy,
            'emotion_entropy': user_emotion_entropy,
            'emotion_entropy_ratio': user_emotion_entropy / max_emotion_entropy,
        },
        'random_baseline': {
            'estimated_coverage': expected_random_coverage,
        }
    }

    print(f"\nResults:")
    print(f"  Stratified sampling:")
    print(f"    Region coverage: {region_coverage:.1%} ({len(region_counts)}/{len(EXPECTED_REGIONS)})")
    print(f"    User emotion coverage: {user_emotion_coverage:.1%}")
    print(f"    Assistant emotion coverage: {assistant_emotion_coverage:.1%}")
    print(f"    Region entropy ratio: {region_entropy/max_region_entropy:.1%}")
    print(f"  Random baseline estimated coverage: {expected_random_coverage:.1%}")

    return results


def create_ablation_table(dedup_results: Dict, sampling_results: Dict) -> pd.DataFrame:
    """Create LaTeX-ready ablation results table."""
    rows = [
        {
            'Configuration': 'Full pipeline',
            'Diversity': f"{dedup_results['full_pipeline']['embedding_diversity']:.3f}",
            'Removed (%)': f"{100 * dedup_results['full_pipeline']['removed'] / (dedup_results['full_pipeline']['removed'] + dedup_results['full_pipeline']['remaining']):.1f}",
            'Region Coverage (%)': f"{100 * sampling_results['stratified']['region_coverage']:.1f}",
            'Emotion Coverage (%)': f"{100 * sampling_results['stratified']['user_emotion_coverage']:.1f}",
        },
        {
            'Configuration': 'No semantic dedup',
            'Diversity': f"{dedup_results['exact_only']['embedding_diversity']:.3f}",
            'Removed (%)': f"{100 * dedup_results['exact_only']['removed'] / (dedup_results['exact_only']['removed'] + dedup_results['exact_only']['remaining']):.1f}",
            'Region Coverage (%)': '-',
            'Emotion Coverage (%)': '-',
        },
        {
            'Configuration': 'No deduplication',
            'Diversity': f"{dedup_results['no_deduplication']['embedding_diversity']:.3f}",
            'Removed (%)': '0.0',
            'Region Coverage (%)': '-',
            'Emotion Coverage (%)': '-',
        },
    ]

    df = pd.DataFrame(rows)
    return df


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table from DataFrame."""
    latex = df.to_latex(index=False, escape=False)
    return latex


def main():
    print("Loading dataset...")
    dialogues = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dialogues)} dialogues")

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Run ablation experiments
    dedup_results = ablation_deduplication(dialogues, model)
    sampling_results = ablation_sampling(dialogues)

    # Create results table
    ablation_table = create_ablation_table(dedup_results, sampling_results)
    print("\n=== Ablation Results Table ===")
    print(ablation_table.to_string(index=False))

    # Save results
    results = {
        'deduplication': dedup_results,
        'sampling': sampling_results
    }

    with open(OUTPUT_DIR / 'ablation_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(results, f, indent=2, default=convert)

    # Save LaTeX table
    latex_table = generate_latex_table(ablation_table)
    with open(OUTPUT_DIR / 'ablation_table.tex', 'w') as f:
        f.write(latex_table)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("  - ablation_results.json")
    print("  - ablation_table.tex")


if __name__ == "__main__":
    main()
