from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from termcolor import colored
import sys
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log
import signal
import threading

# Custom exception for rate limiting
class RateLimitException(Exception):
    pass

# Global flag to handle graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    global shutdown_flag
    logging.info("Received shutdown signal. Initiating graceful shutdown...")
    print(colored("Shutdown signal received. Finishing current dialogues and exiting...", "yellow"))
    shutdown_flag = True

# Register the signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_jsonl_file(file_path):
    """
    Load data from a JSONL file, where each line is a valid JSON object.
    """
    dialogues = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    dialogue = json.loads(line)
                    dialogues.append(dialogue)
        return dialogues
    except Exception as e:
        logging.error(f"Error loading JSONL file: {e}")
        print(colored(f"Error loading JSONL file: {e}", "red"))
        return None

@retry(
    retry=retry_if_exception_type(RateLimitException),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(10),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    reraise=True
)
def call_moderation_api(client, full_conversation, dialogue_id):
    """
    Calls the OpenAI Moderation API with retry logic for rate limiting.
    """
    try:
        moderation = client.moderations.create(input=full_conversation)
        return moderation.results[0]
    except Exception as e:
        if hasattr(e, 'http_status') and e.http_status == 429:
            logging.warning(f"Rate limit exceeded for Dialogue {dialogue_id}. Retrying...")
            raise RateLimitException("Rate limit exceeded.")
        else:
            logging.error(f"Unexpected error for Dialogue {dialogue_id}: {e}")
            raise e

def moderate_conversation(dialogue, client, flagged_dialogues, failed_dialogues, lock):
    """
    Process the entire conversation for a single dialogue and collect flagged results.
    """
    if shutdown_flag:
        return

    full_conversation = ""
    for turn in dialogue.get('turns', []):
        full_conversation += f"User: {turn.get('utterance', '')}\n"
        full_conversation += f"Intent: {turn.get('intent', '')}\n"
        full_conversation += f"Assistant: {turn.get('assistant_response', '')}\n\n"

    dialogue_id = dialogue.get('dialogue_id', 'Unknown ID')
    logging.info(f"Moderating Dialogue {dialogue_id}")

    try:
        results = call_moderation_api(client, full_conversation, dialogue_id)
    except RateLimitException as e:
        logging.error(f"Rate limit exceeded after retries for Dialogue {dialogue_id}: {e}")
        print(colored(f"Rate limit exceeded after retries for Dialogue {dialogue_id}. Skipping...", "yellow"))
        # Optionally, add to failed_dialogues for later processing
        with lock:
            failed_dialogues.append(dialogue)
        return
    except Exception as e:
        logging.error(f"Error during moderation of Dialogue {dialogue_id}: {e}")
        print(colored(f"Error during moderation of Dialogue {dialogue_id}: {e}", "red"))
        # Optionally, add to failed_dialogues for later processing
        with lock:
            failed_dialogues.append(dialogue)
        return  # Skip to next dialogue

    if results.flagged:
        alert_msg = f"ALERT: Harmful content detected in Dialogue {dialogue_id}!"
        logging.warning(alert_msg)
        print(colored(alert_msg, "red", attrs=['bold']))
        print(colored("Content flagged for the following categories:", "red"))
        flagged_categories = []
        for category, flagged in results.categories.items():
            if flagged:
                flagged_categories.append(category)
                print(colored(f"- {category}", "red"))
        logging.warning(f"Flagged Categories for Dialogue {dialogue_id}: {flagged_categories}")

        # Collect flagged dialogues in a thread-safe manner
        with lock:
            flagged_dialogues.append({
                'dialogue_id': dialogue_id,
                'categories': flagged_categories
            })
    else:
        success_msg = f"Dialogue {dialogue_id}: All content is appropriate."
        logging.info(success_msg)
        print(colored(success_msg, "green"))

    # Log category scores
    print("Category Scores:")
    category_scores = results.category_scores
    for category, score in category_scores.items():
        score_display = f"{score:.4f}" if score is not None else "N/A"
        print(f"- {category}: {score_display}")
        logging.info(f"Dialogue {dialogue_id} - {category}: {score_display}")

    print("---\n")

def process_dialogues(file_path, client, max_workers=5, retry_failed=True):
    """
    Process all dialogues in the JSONL file using parallel processing with rate limit handling.
    Ensures that no dialogue is left unchecked by retrying failed dialogues.
    """
    dialogues = load_jsonl_file(file_path)
    if not dialogues:
        logging.error("No dialogues to process.")
        return

    total_dialogues = len(dialogues)
    logging.info(f"Total dialogues to process: {total_dialogues}")

    flagged_dialogues = []  # To store dialogues with harmful content
    failed_dialogues = []    # To store dialogues that failed due to errors
    lock = Lock()            # To synchronize access to shared lists

    # Function to submit dialogues in batches to handle graceful shutdown
    def submit_dialogues(executor, dialogues_subset):
        futures = [
            executor.submit(moderate_conversation, dialogue, client, flagged_dialogues, failed_dialogues, lock)
            for dialogue in dialogues_subset
        ]
        return futures

    # Initialize ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all dialogues to the executor
        futures = submit_dialogues(executor, dialogues)

        # Use tqdm to display progress as futures complete
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing Dialogues", unit="dialogue"):
            if shutdown_flag:
                logging.info("Shutdown flag detected. Cancelling remaining futures...")
                break  # Exit the loop if shutdown is initiated

    # Handle graceful shutdown: collect remaining dialogues that were not processed
    if shutdown_flag:
        remaining_dialogues = []
        for future, dialogue in zip(futures, dialogues):
            if not future.done():
                remaining_dialogues.append(dialogue)
        if remaining_dialogues:
            logging.info(f"Adding {len(remaining_dialogues)} remaining dialogues to failed_dialogues for later processing.")
            with lock:
                failed_dialogues.extend(remaining_dialogues)

    # Retry failed dialogues if any and if retry_failed is True
    if retry_failed and failed_dialogues and not shutdown_flag:
        logging.info(f"Retrying {len(failed_dialogues)} failed dialogues.")
        print(colored(f"Retrying {len(failed_dialogues)} failed dialogues...", "yellow"))

        # Copy failed_dialogues and clear the list for new failures
        dialogues_to_retry = failed_dialogues.copy()
        with lock:
            failed_dialogues.clear()

        # Retry processing the failed dialogues
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            retry_futures = submit_dialogues(executor, dialogues_to_retry)

            for _ in tqdm(as_completed(retry_futures), total=len(retry_futures), desc="Retrying Dialogues", unit="dialogue"):
                if shutdown_flag:
                    logging.info("Shutdown flag detected during retry. Cancelling remaining futures...")
                    break  # Exit the loop if shutdown is initiated

    # After retries, log any remaining failed dialogues
    if failed_dialogues:
        alert_msg = f"\nTotal dialogues that could not be processed successfully: {len(failed_dialogues)}"
        logging.error(alert_msg)
        print(colored(alert_msg, "red", attrs=['bold']))
        # Optionally, write failed dialogues to a separate JSONL file for manual review or future retries
        failed_file = "failed_dialogues.jsonl"
        try:
            with open(failed_file, 'w', encoding='utf-8') as f:
                for dialogue in failed_dialogues:
                    f.write(json.dumps(dialogue) + "\n")
            logging.info(f"Failed dialogues have been written to {failed_file}.")
            print(colored(f"Failed dialogues have been written to {failed_file}.", "yellow"))
        except Exception as e:
            logging.error(f"Error writing failed dialogues to file: {e}")
            print(colored(f"Error writing failed dialogues to file: {e}", "red"))
    else:
        success_msg = "All dialogues processed successfully. No harmful content detected."
        logging.info(success_msg)
        print(colored(success_msg, "green", attrs=['bold']))

    # Summary of flagged dialogues
    if flagged_dialogues:
        alert_msg = f"\nTotal dialogues flagged for harmful content: {len(flagged_dialogues)}"
        logging.warning(alert_msg)
        print(colored(alert_msg, "red", attrs=['bold']))
        for flagged in flagged_dialogues:
            print(colored(f"- Dialogue ID: {flagged['dialogue_id']}, Categories: {', '.join(flagged['categories'])}", "red"))
        sys.exit(1)  # Exit with error code after processing all dialogues
    else:
        logging.info("All dialogues processed successfully with no harmful content detected.")
        print(colored("All dialogues processed successfully. No harmful content detected.", "green", attrs=['bold']))

def main():
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Moderate dialogues from a JSONL file using OpenAI's moderation API.")
    parser.add_argument('input_file', help="Path to the input JSONL file containing the dialogues.")
    parser.add_argument('--workers', type=int, default=5, help="Number of parallel worker threads (default: 5).")
    parser.add_argument('--retry-failed', action='store_true', help="Retry failed dialogues after initial processing.")
    args = parser.parse_args()

    # Load environment variables and set up OpenAI client
    load_dotenv('.env.local')
    openai_api_key = os.getenv('OPENAI_KEY')
    
    if not openai_api_key:
        print(colored("Error: OPENAI_KEY not found in environment variables.", "red"))
        sys.exit(1)

    client = OpenAI(api_key=openai_api_key)

    # Set up logging
    log_file = "moderation_logs.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # Optional: also log to console
        ]
    )
    logging.info("Starting dialogue moderation process.")

    # Process the dialogues
    process_dialogues(args.input_file, client, max_workers=args.workers, retry_failed=args.retry_failed)

    logging.info("Dialogue moderation process completed.")

if __name__ == "__main__":
    main()
