from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from termcolor import colored
import sys

def load_json_file(file_path):
    """
    Load JSON data from a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def moderate_conversation(dialogue, client):
    """
    Process the entire conversation for a single dialogue and display interpretable results.
    """
    full_conversation = ""
    for turn in dialogue['turns']:
        full_conversation += f"User: {turn['utterance']}\n"
        full_conversation += f"Intent: {turn['intent']}\n"
        full_conversation += f"Assistant: {turn['assistant_response']}\n\n"
    
    print(f"Moderating Dialogue {dialogue['dialogue_id']}:")
    moderation = client.moderations.create(input=full_conversation)
    
    results = moderation.results[0]
    if results.flagged:
        print(colored(f"ALERT: Harmful content detected in Dialogue {dialogue['dialogue_id']}!", "red", attrs=['bold']))
        print(colored("Content flagged for the following categories:", "red"))
        for category, flagged in results.categories.model_dump().items():
            if flagged:
                print(colored(f"- {category}", "red"))
        print(colored("Execution stopped due to harmful content.", "red", attrs=['bold']))
        sys.exit(1)
    else:
        print(colored("All content is appropriate.", "green"))
    
    print("Category Scores:")
    for category, score in results.category_scores.model_dump().items():
        if score is not None:
            print(f"- {category}: {score:.4f}")
        else:
            print(f"- {category}: N/A")
    
    print("---\n")
def process_dialogues(file_path, client):
    """
    Process all dialogues in the JSON file.
    """
    dialogues = load_json_file(file_path)
    if not dialogues:
        return

    for dialogue in dialogues:
        moderate_conversation(dialogue, client)
    
    print(colored("All dialogues processed successfully. No harmful content detected.", "green", attrs=['bold']))


if __name__ == "__main__":
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Moderate dialogues from a JSON file using OpenAI's moderation API.")
    parser.add_argument('input_file', help="Path to the input JSON file containing the dialogues.")
    args = parser.parse_args()

    # Load environment variables and set up OpenAI client
    load_dotenv('.env.local')
    openai_api_key = os.getenv('OPENAI_KEY')
    client = OpenAI(api_key=openai_api_key)

    # Process the dialogues
    process_dialogues(args.input_file, client)