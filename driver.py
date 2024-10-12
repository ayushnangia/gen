import os
import subprocess
import argparse
import itertools
import logging
from datetime import datetime
import random
import math

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("driver.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Driver script to automate dialogue generation using capstone.py.")
    
    parser.add_argument('--min_turns', type=int, nargs='+', default=[3, 4, 5],
                        help="List of minimum turns (e.g., 3 4 5).")
    parser.add_argument('--max_turns', type=int, nargs='+', default=[7, 8, 9, 10],
                        help="List of maximum turns (e.g., 7 8 9 10).")
    parser.add_argument('--num_generations', type=int, nargs='+', default=[10, 20, 30],
                        help="List of number of generations per run (e.g., 10 20 30).")
    parser.add_argument('--output_dir', type=str, default='generated_dialogues',
                        help="Directory to store the generated JSON files.")
    parser.add_argument('--total_dialogues', type=int, default=100,
                        help="Total number of dialogues to generate.")
    parser.add_argument('--dry_run', action='store_true',
                        help="If set, the script will only print the commands without executing them.")
    
    return parser.parse_args()

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    else:
        logger.info(f"Output directory already exists: {output_dir}")

def generate_filename(min_turn, max_turn, num_gen, run_index):
    """
    Generates a filename based on min_turn, max_turn, num_gen, and run_index.
    Example: min3_max7_gen10_run1_20240101_123456.json
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"min{min_turn}_max{max_turn}_gen{num_gen}_run{run_index}_{timestamp}.json"
    return filename

def calculate_runs(min_turns, max_turns, num_generations, total_dialogues):
    """
    Calculates how to distribute the total dialogues across different parameter combinations.
    """
    combinations = list(itertools.product(min_turns, max_turns, num_generations))
    random.shuffle(combinations)  # Shuffle to ensure randomness in parameter selection

    runs = []
    dialogues_remaining = total_dialogues

    while dialogues_remaining > 0:
        for combo in combinations:
            if dialogues_remaining <= 0:
                break
            min_turn, max_turn, num_gen = combo
            # Ensure min_turn is less than max_turn
            if min_turn >= max_turn:
                continue
            # Determine how many dialogues to generate in this run
            gen = min(num_gen, dialogues_remaining)
            runs.append({
                'min_turns': min_turn,
                'max_turns': max_turn,
                'num_generations': gen
            })
            dialogues_remaining -= gen
    return runs

def main():
    args = parse_arguments()
    
    min_turns = args.min_turns
    max_turns = args.max_turns
    num_generations = args.num_generations
    output_dir = args.output_dir
    total_dialogues = args.total_dialogues
    dry_run = args.dry_run

    ensure_output_directory(output_dir)

    # Calculate how to distribute total_dialogues across parameter combinations
    runs = calculate_runs(min_turns, max_turns, num_generations, total_dialogues)
    logger.info(f"Total runs to generate: {len(runs)} to reach {total_dialogues} dialogues.")

    for idx, run in enumerate(runs, start=1):
        min_turn = run['min_turns']
        max_turn = run['max_turns']
        num_gen = run['num_generations']
        
        output_filename = generate_filename(min_turn, max_turn, num_gen, idx)
        output_path = os.path.join(output_dir, output_filename)
        
        command = [
            "python", "capstone.py",
            "--num_generations", str(num_gen),
            "--min_turns", str(min_turn),
            "--max_turns", str(max_turn),
            "--output_file", output_path
        ]
        
        logger.info(f"Run {idx}: min_turns={min_turn}, max_turns={max_turn}, num_generations={num_gen}")
        logger.info(f"Output File: {output_path}")
        logger.debug(f"Command: {' '.join(command)}")
        
        if dry_run:
            print(f"DRY RUN: {' '.join(command)}")
        else:
            try:
                subprocess.run(command, check=True)
                logger.info(f"Successfully completed run {idx}.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error during run {idx}: {e}")
    
    logger.info("Dialogue generation completed.")

if __name__ == "__main__":
    main()
