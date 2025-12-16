import json
import os
import argparse
import logging
from tqdm import tqdm

def setup_logging(log_level=logging.INFO):
    """
    Sets up the logging configuration.
    
    Parameters:
    - log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
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
    
    Returns:
    - args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Filter JSON entries based on 'num_lines' field."
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input JSON file.'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to the output filtered JSON file.'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=5,
        help='Threshold for "num_lines" (default: 7).'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging.'
    )
    return parser.parse_args()

def filter_json_by_num_lines(input_file, output_file, threshold=7):
    """
    Filters entries in a JSON file where 'num_lines' is greater than the specified threshold.
    
    Parameters:
    - input_file (str): Path to the input JSON file.
    - output_file (str): Path to the output JSON file.
    - threshold (int): The 'num_lines' threshold for filtering.
    """
    logging.info(f"Starting to process the file: {input_file}")
    
    # Check if the input file exists
    if not os.path.isfile(input_file):
        logging.error(f"The file '{input_file}' does not exist.")
        return
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        return
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {e}")
        return
    
    # Ensure the data is a list
    if not isinstance(data, list):
        logging.error("The JSON file does not contain a list of entries.")
        return
    
    total_entries = len(data)
    logging.info(f"Total entries found: {total_entries}")
    
    # Initialize lists and counters for filtered and excluded data
    filtered_data = []
    excluded_count = 0
    included_count = 0
    
    # Use tqdm for progress bar
    logging.info(f"Filtering entries with 'num_lines' > {threshold}...")
    for entry in tqdm(data, desc="Processing Entries", unit="entry"):
        num_lines = entry.get('num_lines')
        if isinstance(num_lines, int):
            if num_lines > threshold:
                filtered_data.append(entry)
                included_count += 1
            else:
                excluded_count += 1
        else:
            logging.warning(f"Entry with dialogue_id '{entry.get('dialogue_id')}' has invalid 'num_lines': {num_lines}")
            excluded_count += 1  # Optionally count invalid entries as excluded
    
    logging.info(f"Total entries after filtering (num_lines > {threshold}): {included_count}")
    logging.info(f"Total entries excluded (num_lines <= {threshold} or invalid): {excluded_count}")
    
    # Check if any entries were found
    if not filtered_data:
        logging.warning(f"No entries found with 'num_lines' greater than {threshold}.")
    else:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=4)
            logging.info(f"Filtered data has been written to '{output_file}'.")
        except Exception as e:
            logging.error(f"An error occurred while writing to the file: {e}")

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    if args.verbose:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.INFO)
    
    # Call the filtering function
    filter_json_by_num_lines(args.input_file, args.output_file, args.threshold)

if __name__ == "__main__":
    main()
