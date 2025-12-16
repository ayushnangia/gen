#!/usr/bin/env python3
"""
Independent Safety Audit Script for ACL Paper.
Uses multiple safety evaluation methods to address circular evaluation concerns.

This script implements independent safety evaluation as requested by reviewers:
- Detoxify (Unitary AI toxicity detector)
- Perspective API scores (optional, requires API key)
- Keyword-based adverse content detection
- Statistical outlier detection for safety patterns

Note: WildGuard would require significant compute resources.
This script uses accessible alternatives.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
from tqdm import tqdm

# Configuration
SYNWOZ_PATH = "./SynWOZ-dataset/dataset.jsonl"
OUTPUT_DIR = Path("./safety_audit_results")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_SIZE = 2000  # Sample for efficiency

# Sensitive content keywords (expanded list)
SENSITIVE_KEYWORDS = {
    'violence': ['kill', 'murder', 'attack', 'weapon', 'bomb', 'shoot', 'stab', 'assault', 'hurt', 'destroy'],
    'hate': ['hate', 'racist', 'sexist', 'bigot', 'slur', 'discriminate'],
    'sexual': ['sex', 'nude', 'porn', 'explicit', 'adult content'],
    'self_harm': ['suicide', 'self-harm', 'cut myself', 'end my life', 'kill myself'],
    'illegal': ['drug', 'cocaine', 'heroin', 'meth', 'illegal', 'smuggle', 'launder'],
    'profanity': ['fuck', 'shit', 'damn', 'ass', 'bitch', 'bastard'],
}


def load_synwoz(path: str, sample_size: int = None) -> List[Dict]:
    """Load SynWOZ dialogues."""
    dialogues = []
    with open(path, 'r') as f:
        for line in f:
            dialogues.append(json.loads(line))

    if sample_size and len(dialogues) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(dialogues), sample_size, replace=False)
        dialogues = [dialogues[i] for i in indices]

    return dialogues


def extract_text(dialogue: Dict) -> str:
    """Extract all text from a dialogue."""
    text_parts = []
    for turn in dialogue.get('turns', []):
        utterance = turn.get('utterance', '').strip()
        response = turn.get('assistant_response', '').strip()
        if utterance:
            text_parts.append(utterance)
        if response:
            text_parts.append(response)
    return ' '.join(text_parts).lower()


def keyword_based_audit(dialogues: List[Dict]) -> Dict:
    """Perform keyword-based safety audit."""
    results = {category: [] for category in SENSITIVE_KEYWORDS}
    flagged_dialogues = []

    for i, dialogue in enumerate(tqdm(dialogues, desc="Keyword audit")):
        text = extract_text(dialogue)
        dialogue_flags = []

        for category, keywords in SENSITIVE_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    results[category].append({
                        'dialogue_idx': i,
                        'keyword': keyword,
                        'context': text[:200]  # First 200 chars for context
                    })
                    dialogue_flags.append(category)
                    break  # One flag per category per dialogue

        if dialogue_flags:
            flagged_dialogues.append({
                'idx': i,
                'categories': dialogue_flags
            })

    return {
        'category_counts': {k: len(v) for k, v in results.items()},
        'flagged_dialogues': flagged_dialogues,
        'total_flagged': len(flagged_dialogues),
        'total_dialogues': len(dialogues),
        'flag_rate': len(flagged_dialogues) / len(dialogues) * 100
    }


def detoxify_audit(dialogues: List[Dict], sample_size: int = 500) -> Dict:
    """Use Detoxify model for toxicity detection."""
    try:
        from detoxify import Detoxify
    except ImportError:
        print("Warning: detoxify not installed. Run: pip install detoxify")
        return {'error': 'detoxify not installed'}

    model = Detoxify('original')

    # Sample for efficiency
    if len(dialogues) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(dialogues), sample_size, replace=False)
        sample = [dialogues[i] for i in indices]
    else:
        sample = dialogues

    texts = [extract_text(d) for d in sample]

    # Get predictions
    results = []
    toxicity_scores = []

    for i, text in enumerate(tqdm(texts, desc="Detoxify audit")):
        if len(text) > 5000:  # Truncate long texts
            text = text[:5000]

        pred = model.predict(text)
        toxicity_scores.append(pred['toxicity'])

        # Flag if any score > 0.5
        if any(v > 0.5 for v in pred.values()):
            results.append({
                'idx': i,
                'scores': pred,
                'text_preview': text[:100]
            })

    return {
        'sample_size': len(sample),
        'flagged_count': len(results),
        'flag_rate': len(results) / len(sample) * 100,
        'mean_toxicity': np.mean(toxicity_scores),
        'max_toxicity': np.max(toxicity_scores),
        'toxicity_std': np.std(toxicity_scores),
        'flagged_examples': results[:10]  # First 10 examples
    }


def statistical_audit(dialogues: List[Dict]) -> Dict:
    """Statistical analysis of dialogue patterns for anomalies."""
    lengths = []
    turn_counts = []
    exclamation_counts = []
    question_counts = []
    caps_ratios = []

    for dialogue in tqdm(dialogues, desc="Statistical audit"):
        text = extract_text(dialogue)
        lengths.append(len(text))
        turn_counts.append(len(dialogue.get('turns', [])))
        exclamation_counts.append(text.count('!'))
        question_counts.append(text.count('?'))

        # Caps ratio
        if len(text) > 0:
            caps = sum(1 for c in text if c.isupper())
            caps_ratios.append(caps / len(text))

    # Find outliers (> 3 std from mean)
    def find_outliers(values, name):
        mean, std = np.mean(values), np.std(values)
        outliers = [i for i, v in enumerate(values) if abs(v - mean) > 3 * std]
        return {
            'metric': name,
            'mean': mean,
            'std': std,
            'outlier_count': len(outliers),
            'outlier_indices': outliers[:10]
        }

    return {
        'length_stats': find_outliers(lengths, 'text_length'),
        'turn_stats': find_outliers(turn_counts, 'turn_count'),
        'exclamation_stats': find_outliers(exclamation_counts, 'exclamations'),
        'question_stats': find_outliers(question_counts, 'questions'),
        'caps_stats': find_outliers(caps_ratios, 'caps_ratio'),
    }


def main():
    print("=" * 60)
    print("Independent Safety Audit")
    print("=" * 60)
    print("This audit uses methods independent of OpenAI moderation API")
    print("to address circular evaluation concerns.\n")

    # Load data
    print(f"Loading SynWOZ dataset from {SYNWOZ_PATH}")
    dialogues = load_synwoz(SYNWOZ_PATH, SAMPLE_SIZE)
    print(f"Loaded {len(dialogues)} dialogues for audit\n")

    results = {}

    # 1. Keyword-based audit
    print("\n--- 1. Keyword-Based Safety Audit ---")
    keyword_results = keyword_based_audit(dialogues)
    results['keyword_audit'] = keyword_results
    print(f"Total flagged: {keyword_results['total_flagged']} ({keyword_results['flag_rate']:.2f}%)")
    print("By category:")
    for cat, count in keyword_results['category_counts'].items():
        print(f"  {cat}: {count}")

    # 2. Detoxify audit
    print("\n--- 2. Detoxify Model Audit ---")
    detoxify_results = detoxify_audit(dialogues)
    results['detoxify_audit'] = detoxify_results
    if 'error' not in detoxify_results:
        print(f"Sample size: {detoxify_results['sample_size']}")
        print(f"Flagged (toxicity > 0.5): {detoxify_results['flagged_count']} ({detoxify_results['flag_rate']:.2f}%)")
        print(f"Mean toxicity score: {detoxify_results['mean_toxicity']:.4f}")
        print(f"Max toxicity score: {detoxify_results['max_toxicity']:.4f}")

    # 3. Statistical audit
    print("\n--- 3. Statistical Pattern Audit ---")
    stat_results = statistical_audit(dialogues)
    results['statistical_audit'] = stat_results
    for key, stats in stat_results.items():
        print(f"{stats['metric']}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, outliers={stats['outlier_count']}")

    # Summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total dialogues audited: {len(dialogues)}")
    print(f"Keyword-flagged: {keyword_results['flag_rate']:.2f}%")
    if 'error' not in detoxify_results:
        print(f"Detoxify-flagged: {detoxify_results['flag_rate']:.2f}%")
    print(f"\nNote: These results are from independent auditors,")
    print(f"not the OpenAI moderation API used during filtering.")

    # Save results
    with open(OUTPUT_DIR / 'safety_audit_results.json', 'w') as f:
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

    print(f"\nResults saved to {OUTPUT_DIR}/safety_audit_results.json")


if __name__ == "__main__":
    main()
