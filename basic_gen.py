import os
import json
import random
import re
import argparse
import logging
from time import sleep
from typing import List, Dict
from datasets import load_dataset
from openai import OpenAI, OpenAIError
from tqdm import tqdm
import spacy

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("dialogue_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy's English model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("spaCy model not found. Downloading 'en_core_web_sm'...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable not set.")
    exit(1)
client = OpenAI(api_key=openai_api_key)

def anonymize_text(text: str) -> str:
    """
    Anonymize specific entities in the text such as locations, times, and numbers.
    """
    doc = nlp(text)
    anonymized_text = text
    # Define entity replacements
    replacements = {
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "TIME": "TIME",
        "DATE": "DATE",
        "CARDINAL": "NUMBER",
        "ORDINAL": "NUMBER",
        "MONEY": "AMOUNT",
        "PERSON": "PERSON",
        "ORG": "ORGANIZATION"
    }
    
    # Sort entities by start index in reverse to avoid offset issues during replacement
    entities = sorted(doc.ents, key=lambda ent: ent.start_char, reverse=True)
    
    for ent in entities:
        if ent.label_ in replacements:
            placeholder = f"<{replacements[ent.label_].upper()}>"
            anonymized_text = anonymized_text[:ent.start_char] + placeholder + anonymized_text[ent.end_char:]
    
    return anonymized_text

def extract_and_anonymize_dialogue(dialogue_json: Dict) -> List[Dict]:
    """
    Extracts turns from the dialogue JSON and anonymizes the utterances.
    Returns a list of turns with anonymized utterances.
    """
    turns = []
    speakers = dialogue_json.get("speaker", [])
    utterances = dialogue_json.get("utterance", [])
    turn_ids = dialogue_json.get("turn_id", [])

    for turn_id, speaker, utterance in zip(turn_ids, speakers, utterances):
        if speaker == 0:
            speaker_label = "USER"
        elif speaker == 1:
            speaker_label = "ASSISTANT"
        else:
            speaker_label = "UNKNOWN"
        
        anonymized_utterance = anonymize_text(utterance)
        
        turns.append({
            "turn_id": turn_id,
            "speaker": speaker_label,
            "utterance": anonymized_utterance
        })
    
    return turns

def generate_base_conversation(turns: List[Dict]) -> str:
    """
    Formats the list of turns into a base conversation string.
    """
    conversation = ""
    for turn in turns:
        conversation += f"{turn['speaker']}: {turn['utterance']}\n"
    return conversation.strip()

def process_dialogue_json(dialogue_json: Dict) -> Dict:
    """
    Processes a single dialogue JSON to extract and anonymize the conversation.
    Returns a dictionary with the anonymized conversation.
    """
    anonymized_turns = extract_and_anonymize_dialogue(dialogue_json)
    base_conversation = generate_base_conversation(anonymized_turns)
    
    return {
        "dialogue_id": dialogue_json.get("dialogue_id", "unknown_id"),
        "services": dialogue_json.get("services", []),
        "turns": anonymized_turns,
        "base_conversation": base_conversation
    }

def generate_dialogue(service, prompt, min_turns, max_turns, max_retries=3):
    try:
        system_prompt = (
            f"You are an expert dialogue generator for the '{service}' service. "
            f"Create a high-quality, coherent, and relevant dialogue between a user and an assistant. "
            f"The dialogue should have between {min_turns} and {max_turns} turns (a turn is one user message and one assistant response). "
            f"The dialogue should not be the same as any existing dialogues and should be better and more engaging."
        )

        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model='gpt-4o-mini',  # Corrected model name
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.7,
                    n=1,
                )
                generated_dialogue = response.choices[0].message.content.strip()
                return generated_dialogue
            except OpenAIError as e:
                logger.warning(f"Attempt {attempt} - OpenAI API error: {e}")
                if attempt < max_retries:
                    sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed after {max_retries} attempts.")
                    return None
    except Exception as e:
        logger.error(f"Unexpected error in generate_dialogue: {e}")
        return None

def process_generated_dialogue(generated_dialogue: str) -> List[Dict]:
    """
    Processes the generated dialogue text into a list of turns.
    """
    generated_turns = []
    for line in generated_dialogue.split('\n'):
        line = line.strip()
        if line:
            if line.lower().startswith('user:'):
                speaker = 'USER'
                utterance = line.split(':', 1)[1].strip()
            elif line.lower().startswith(('assistant:', 'system:', 'agent:')):
                speaker = 'ASSISTANT'
                utterance = line.split(':', 1)[1].strip()
            else:
                continue
            generated_turns.append({
                'speaker': speaker,
                'utterance': utterance
            })
    return generated_turns

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate dialogues using OpenAI API.")
    parser.add_argument('--num_generations', type=int, required=True, help="Number of dialogues to generate.")
    parser.add_argument('--min_turns', type=int, default=3, help="Minimum number of dialogue turns.")
    parser.add_argument('--max_turns', type=int, default=10, help="Maximum number of dialogue turns.")
    parser.add_argument('--output_file', type=str, default='generated_dialogues.json', help="Output JSON file path.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    num_generations = args.num_generations
    min_turns = args.min_turns
    max_turns = args.max_turns
    output_file = args.output_file

    logger.info("Starting dialogue generation...")
    logger.info(f"Parameters: num_generations={num_generations}, min_turns={min_turns}, max_turns={max_turns}, output_file='{output_file}'")

    # Load dataset from Hugging Face
    try:
        dataset = load_dataset('Ayushnangia/transport_multiwoz_v22')
        data_split = dataset['train']
        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Prepare list to collect new dialogues
    new_dialogues = []

    # To ensure uniqueness, load existing dialogues if the output file exists
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_dialogues = json.load(f)
                existing_ids = {dialogue['dialogue_id'] for dialogue in existing_dialogues}
            logger.info(f"Loaded {len(existing_dialogues)} existing dialogues from '{output_file}'.")
        except Exception as e:
            logger.warning(f"Could not load existing dialogues: {e}")
            existing_dialogues = []
            existing_ids = set()
    else:
        existing_dialogues = []
        existing_ids = set()

    # Randomly select examples to generate dialogues for
    if num_generations > len(data_split):
        logger.error("Number of generations requested exceeds the dataset size.")
        return

    selected_indices = random.sample(range(len(data_split)), num_generations)

    for index in tqdm(selected_indices, desc="Generating dialogues"):
        example = data_split[index]
        services = example.get('services', [])
        dialogue_id = example.get('dialogue_id', f"dialogue_{index}")

        # Extract and anonymize existing dialogue
        processed_dialogue = process_dialogue_json(example)
        base_conversation = processed_dialogue['base_conversation']

        prompt = (
            f"Using the following base conversation as a reference, create a new dialogue for the service(s): {', '.join(services)}. "
            f"The dialogue should be completely new and more relevant than any existing dialogue. Do not copy any part of existing dialogues. "
            f"The dialogue should be between a user and an assistant.\n\n"
            f"Base Conversation:\n{base_conversation}"
        )

        generated_dialogue = generate_dialogue(services[0] if services else "general", prompt, min_turns, max_turns)

        if generated_dialogue:
            generated_turns = process_generated_dialogue(generated_dialogue)

            new_dialogue_id = f"{dialogue_id}_generated_{index}"
            if new_dialogue_id in existing_ids:
                logger.warning(f"Duplicate dialogue_id '{new_dialogue_id}' found. Skipping.")
                continue

            new_dialogues.append({
                'services': services,
                'dialogue_id': new_dialogue_id,
                'turns': generated_turns
            })
            existing_ids.add(new_dialogue_id)

    logger.info("Dialogue generation complete.")

    # Combine existing and new dialogues
    all_dialogues = existing_dialogues + new_dialogues

    # Save the new dialogues to a JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(all_dialogues, f, indent=4)
        logger.info(f"Generated dialogues saved to '{output_file}'. Total dialogues: {len(all_dialogues)}.")
    except Exception as e:
        logger.error(f"Failed to save dialogues to '{output_file}': {e}")

if __name__ == "__main__":
    main()
#python generate_dialogues.py --num_generations 50 --min_turns 3 --max_turns 7 --output_file my_generated_dialogues.json
