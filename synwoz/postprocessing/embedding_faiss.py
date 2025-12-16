import json
import os
import argparse
import logging
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

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

def deduplicate_entries(data, embeddings, threshold=0.9):
    """
    Deduplicates entries based on cosine similarity threshold using FAISS.
    Returns deduplicated data and counts.
    """
    logging.info("Setting up FAISS for similarity search...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity (normalized vectors)
    index.add(embeddings)
    
    logging.info("Performing deduplication...")
    # Query for each vector to find similar vectors
    # We'll keep track of which entries to keep
    to_remove = set()
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
            if sim >= threshold:
                to_remove.add(j)
        if (i + 1) % 10000 == 0:
            logging.info(f"Processed {i + 1}/{len(data)} entries...")
    
    included_data = [entry for idx, entry in enumerate(data) if idx not in to_remove]
    excluded_count = len(to_remove)
    included_count = len(included_data)
    logging.info(f"Total entries after deduplication: {included_count}")
    logging.info(f"Total entries removed: {excluded_count}")
    return included_data, included_count, excluded_count

def save_json(data, output_file):
    """
    Saves JSON data to a file.
    """
    logging.info(f"Saving deduplicated data to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info("Data saved successfully.")

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    if args.verbose:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.INFO)
    
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
    
    # Deduplicate entries
    dedup_data, included_count, excluded_count = deduplicate_entries(data, embeddings, args.threshold)
    
    # Save deduplicated data
    save_json(dedup_data, args.output_file)
    
    # Final summary
    logging.info("Deduplication process completed.")
    logging.info(f"Included dialogues: {included_count}")
    logging.info(f"Excluded dialogues: {excluded_count}")

if __name__ == "__main__":
    main()
