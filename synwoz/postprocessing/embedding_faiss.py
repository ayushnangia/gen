import json
import os
import argparse
import logging
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


@dataclass
class NearMatch:
    """Represents a near-match relationship between two dialogues."""
    kept_index: int
    removed_index: int
    similarity_score: float
    kept_dialogue_preview: str
    removed_dialogue_preview: str
    reason: str


@dataclass
class DeduplicationStats:
    """Statistics and metadata from a deduplication run."""
    threshold: float
    model_name: str
    input_file: str
    output_file: str
    timestamp: str
    total_input: int
    total_output: int
    total_removed: int
    similarity_distribution: Dict[str, int]  # Buckets of similarity scores
    near_matches_logged: int

def setup_logging(log_level=logging.INFO):
    """
    Sets up the logging configuration.
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Deduplicate JSON dialogues based on similarity."
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input JSON file.'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to the output deduplicated JSON file.'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.9,
        help='Cosine similarity threshold above which dialogues are considered duplicates (default: 0.9).'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence Transformer model name (default: all-MiniLM-L6-v2).'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging.'
    )
    parser.add_argument(
        '--near-match-log',
        type=str,
        default=None,
        help='Path to output file for logging near-match details (JSONL format).'
    )
    parser.add_argument(
        '--validation-sample',
        type=str,
        default=None,
        help='Path to output file for validation sample (JSON format with sample pairs for manual review).'
    )
    parser.add_argument(
        '--validation-sample-size',
        type=int,
        default=50,
        help='Number of near-match pairs to include in validation sample (default: 50).'
    )
    parser.add_argument(
        '--stats-output',
        type=str,
        default=None,
        help='Path to output file for deduplication statistics and threshold metadata (JSON format).'
    )
    parser.add_argument(
        '--preview-length',
        type=int,
        default=200,
        help='Maximum character length for dialogue previews in logs (default: 200).'
    )
    return parser.parse_args()

def load_json(input_file):
    """
    Loads JSON data from a file.
    """
    logging.info(f"Loading JSON data from '{input_file}'...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} entries.")
    return data

def extract_dialogue(entry):
    """
    Extracts and concatenates the dialogue from an entry.
    """
    turns = entry.get('turns', [])
    dialogue = []
    for turn in turns:
        user_utterance = turn.get('utterance', '').strip()
        assistant_response = turn.get('assistant_response', '').strip()
        if user_utterance:
            dialogue.append(f"User: {user_utterance}")
        if assistant_response:
            dialogue.append(f"Assistant: {assistant_response}")
    return " ".join(dialogue)

def generate_embeddings(dialogues, model_name):
    """
    Generates embeddings for a list of dialogues using SentenceTransformer.
    """
    logging.info(f"Loading SentenceTransformer model '{model_name}'...")
    model = SentenceTransformer(model_name)
    logging.info("Generating embeddings...")
    embeddings = model.encode(dialogues, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

def truncate_preview(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length with ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def get_similarity_bucket(similarity: float) -> str:
    """Categorize similarity score into buckets for distribution tracking."""
    if similarity >= 0.99:
        return "0.99-1.00"
    elif similarity >= 0.95:
        return "0.95-0.99"
    elif similarity >= 0.90:
        return "0.90-0.95"
    elif similarity >= 0.85:
        return "0.85-0.90"
    elif similarity >= 0.80:
        return "0.80-0.85"
    else:
        return "below-0.80"


def deduplicate_entries(
    data: List[Dict[str, Any]],
    embeddings: np.ndarray,
    dialogues: List[str],
    threshold: float = 0.9,
    preview_length: int = 200,
    collect_near_matches: bool = False
) -> Tuple[List[Dict[str, Any]], int, int, List[NearMatch], Dict[str, int]]:
    """
    Deduplicates entries based on cosine similarity threshold using FAISS.

    Args:
        data: List of dialogue entries
        embeddings: Normalized embeddings array
        dialogues: List of extracted dialogue strings (for previews)
        threshold: Similarity threshold for considering duplicates
        preview_length: Max length for dialogue previews
        collect_near_matches: Whether to collect detailed near-match info

    Returns:
        Tuple of (deduplicated_data, included_count, excluded_count,
                  near_matches, similarity_distribution)
    """
    logging.info("Setting up FAISS for similarity search...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity (normalized vectors)
    index.add(embeddings)

    logging.info("Performing deduplication...")
    logging.info(f"Using similarity threshold: {threshold}")

    # Track which entries to remove and why
    to_remove = set()
    near_matches: List[NearMatch] = []
    similarity_distribution: Dict[str, int] = {
        "0.99-1.00": 0,
        "0.95-0.99": 0,
        "0.90-0.95": 0,
        "0.85-0.90": 0,
        "0.80-0.85": 0,
        "below-0.80": 0
    }

    for i in tqdm(range(len(data)), desc="Deduplicating", unit="entry"):
        if i in to_remove:
            continue
        # Search for similar entries
        k = 10  # Number of nearest neighbors to search
        D, I = index.search(np.expand_dims(embeddings[i], axis=0), k)
        similar_indices = I[0]
        similarities = D[0]

        for j, sim in zip(similar_indices, similarities):
            if j == i:
                continue  # Skip self
            if j in to_remove:
                continue  # Already marked for removal

            # Track similarity distribution for all above-threshold matches
            if sim >= 0.80:  # Track similarities even below threshold for analysis
                bucket = get_similarity_bucket(sim)
                similarity_distribution[bucket] += 1

            if sim >= threshold:
                to_remove.add(j)

                # Collect detailed near-match info if requested
                if collect_near_matches:
                    reason = (
                        f"Similarity {sim:.4f} >= threshold {threshold}. "
                        f"Entry {j} removed as duplicate of entry {i}."
                    )
                    near_match = NearMatch(
                        kept_index=i,
                        removed_index=j,
                        similarity_score=float(sim),
                        kept_dialogue_preview=truncate_preview(dialogues[i], preview_length),
                        removed_dialogue_preview=truncate_preview(dialogues[j], preview_length),
                        reason=reason
                    )
                    near_matches.append(near_match)

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(
                            f"Near-match found: entry {j} matches entry {i} "
                            f"with similarity {sim:.4f}"
                        )

        if (i + 1) % 10000 == 0:
            logging.info(f"Processed {i + 1}/{len(data)} entries...")

    included_data = [entry for idx, entry in enumerate(data) if idx not in to_remove]
    excluded_count = len(to_remove)
    included_count = len(included_data)

    logging.info(f"Total entries after deduplication: {included_count}")
    logging.info(f"Total entries removed: {excluded_count}")
    logging.info(f"Near-matches collected: {len(near_matches)}")

    # Log similarity distribution summary
    logging.info("Similarity distribution of matches:")
    for bucket, count in similarity_distribution.items():
        if count > 0:
            logging.info(f"  {bucket}: {count}")

    return included_data, included_count, excluded_count, near_matches, similarity_distribution

def save_json(data, output_file):
    """
    Saves JSON data to a file.
    """
    logging.info(f"Saving deduplicated data to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info("Data saved successfully.")


def save_near_match_log(near_matches: List[NearMatch], output_file: str) -> None:
    """
    Saves near-match details to a JSONL file for analysis.

    Each line contains one near-match record with:
    - Indices of kept and removed entries
    - Similarity score
    - Dialogue previews
    - Reasoning for removal
    """
    logging.info(f"Saving near-match log to '{output_file}'...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for nm in near_matches:
            f.write(json.dumps(asdict(nm), ensure_ascii=False) + "\n")

    logging.info(f"Saved {len(near_matches)} near-match records.")


def create_validation_sample(
    near_matches: List[NearMatch],
    sample_size: int = 50,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Creates a stratified sample of near-matches for manual validation.

    The sample is stratified by similarity score buckets to ensure
    representation across different similarity levels.

    Args:
        near_matches: List of all near-matches found
        sample_size: Target number of samples
        seed: Random seed for reproducibility

    Returns:
        List of sample entries formatted for manual review
    """
    if not near_matches:
        return []

    random.seed(seed)

    # Stratify by similarity score
    buckets: Dict[str, List[NearMatch]] = {
        "0.99-1.00": [],
        "0.95-0.99": [],
        "0.90-0.95": [],
        "0.85-0.90": [],
    }

    for nm in near_matches:
        bucket = get_similarity_bucket(nm.similarity_score)
        if bucket in buckets:
            buckets[bucket].append(nm)

    # Calculate samples per bucket (proportional, but ensure at least 1 if available)
    total_available = sum(len(b) for b in buckets.values())
    if total_available == 0:
        return []

    samples: List[NearMatch] = []
    remaining_size = min(sample_size, total_available)

    for bucket_name, bucket_items in buckets.items():
        if not bucket_items:
            continue
        # Proportional allocation with minimum of 1
        proportion = len(bucket_items) / total_available
        bucket_sample_size = max(1, int(remaining_size * proportion))
        bucket_sample_size = min(bucket_sample_size, len(bucket_items))

        bucket_sample = random.sample(bucket_items, bucket_sample_size)
        samples.extend(bucket_sample)

    # Format for manual review
    validation_entries = []
    for idx, nm in enumerate(samples):
        entry = {
            "sample_id": idx + 1,
            "similarity_score": nm.similarity_score,
            "similarity_bucket": get_similarity_bucket(nm.similarity_score),
            "kept_index": nm.kept_index,
            "removed_index": nm.removed_index,
            "kept_dialogue": nm.kept_dialogue_preview,
            "removed_dialogue": nm.removed_dialogue_preview,
            "reason": nm.reason,
            "manual_validation": {
                "is_true_duplicate": None,  # To be filled by reviewer
                "notes": ""
            }
        }
        validation_entries.append(entry)

    return validation_entries


def save_validation_sample(
    near_matches: List[NearMatch],
    output_file: str,
    sample_size: int = 50
) -> None:
    """
    Saves a validation sample to a JSON file for manual review.
    """
    logging.info(f"Creating validation sample of {sample_size} entries...")

    sample = create_validation_sample(near_matches, sample_size)

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "description": "Validation sample for manual review of near-match duplicates",
                "sample_size": len(sample),
                "instructions": "Review each pair and set 'is_true_duplicate' to true/false. Add notes as needed.",
                "generated_at": datetime.now().isoformat()
            },
            "samples": sample
        }, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved {len(sample)} samples to '{output_file}'.")


def save_deduplication_stats(
    stats: DeduplicationStats,
    output_file: str
) -> None:
    """
    Saves deduplication statistics and metadata to a JSON file.
    """
    logging.info(f"Saving deduplication statistics to '{output_file}'...")

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(asdict(stats), f, ensure_ascii=False, indent=2)

    logging.info("Statistics saved successfully.")

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Set up logging
    if args.verbose:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.INFO)

    # Determine if we need to collect near-match details
    collect_near_matches = bool(args.near_match_log or args.validation_sample)

    # Load JSON data
    data = load_json(args.input_file)

    # Extract dialogues
    logging.info("Extracting dialogues from entries...")
    dialogues = []
    for entry in tqdm(data, desc="Extracting Dialogues", unit="entry"):
        dialogue = extract_dialogue(entry)
        dialogues.append(dialogue)

    # Generate embeddings
    embeddings = generate_embeddings(dialogues, args.model)

    # Deduplicate entries with optional near-match collection
    dedup_data, included_count, excluded_count, near_matches, similarity_dist = deduplicate_entries(
        data=data,
        embeddings=embeddings,
        dialogues=dialogues,
        threshold=args.threshold,
        preview_length=args.preview_length,
        collect_near_matches=collect_near_matches
    )

    # Save deduplicated data
    save_json(dedup_data, args.output_file)

    # Save near-match log if requested
    if args.near_match_log:
        save_near_match_log(near_matches, args.near_match_log)

    # Save validation sample if requested
    if args.validation_sample:
        save_validation_sample(
            near_matches,
            args.validation_sample,
            sample_size=args.validation_sample_size
        )

    # Save statistics if requested
    if args.stats_output:
        stats = DeduplicationStats(
            threshold=args.threshold,
            model_name=args.model,
            input_file=args.input_file,
            output_file=args.output_file,
            timestamp=datetime.now().isoformat(),
            total_input=len(data),
            total_output=included_count,
            total_removed=excluded_count,
            similarity_distribution=similarity_dist,
            near_matches_logged=len(near_matches) if collect_near_matches else 0
        )
        save_deduplication_stats(stats, args.stats_output)

    # Final summary
    logging.info("Deduplication process completed.")
    logging.info(f"Included dialogues: {included_count}")
    logging.info(f"Excluded dialogues: {excluded_count}")

    if args.near_match_log:
        logging.info(f"Near-match log: {args.near_match_log}")
    if args.validation_sample:
        logging.info(f"Validation sample: {args.validation_sample}")
    if args.stats_output:
        logging.info(f"Statistics: {args.stats_output}")

if __name__ == "__main__":
    main()
