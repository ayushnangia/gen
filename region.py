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
from dotenv import load_dotenv
import hashlib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline

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

# Load environment variables
load_dotenv('.env.local')

# Initialize OpenAI client
openai_api_key = os.getenv('OPENAI_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable not set.")
    exit(1)
client = OpenAI(api_key=openai_api_key)

# Initialize Sentence Transformer model for embeddings
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}")
    exit(1)

# Initialize Emotion Classification Pipeline
try:
    emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
except Exception as e:
    logger.error(f"Failed to load Emotion Classification model: {e}")
    exit(1)

# Define a list of 32 distinct emotions based on GoEmotions and Plutchik's taxonomy
EMOTION_LIST = [
    "Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring",
    "Confusion", "Curiosity", "Desire", "Disappointment", "Disgust",
    "Embarrassment", "Excitement", "Fear", "Gratitude", "Grief",
    "Joy", "Love", "Nervousness", "Optimism", "Pride",
    "Realization", "Relief", "Remorse", "Sadness", "Surprise",
    "Elation", "Contentment", "Terror", "Envy", "Jealousy", "Boredom"
]

# *** Scenario Diversification: Define various dialogue scenarios ***
# Previously, SCENARIO_LIST was a fixed list. Now, we will generate scenarios dynamically based on categories.
SCENARIO_CATEGORIES = [
    "booking",
    "cancellation",
    "complaint",
    "feedback",
    "inquiry",
    "rescheduling",
    "information request",
    "issue reporting",
    "assistance seeking",
    "personal details update"
]

# *** Predefined List of Regions ***
PREDEFINED_REGIONS = [
    "Tokyo", "Delhi", "Shanghai", "Sao Paulo", "Mumbai",
    "Beijing", "Cairo", "Mexico City", "Dhaka", "Osaka",
    "Karachi", "Chongqing", "Istanbul", "Buenos Aires", "Kolkata",
    "Kinshasa", "Lagos", "Manila", "Rio de Janeiro", "Guangzhou",
    "Los Angeles", "Moscow", "Paris", "Bangkok", "Jakarta",
    "London", "Lima", "New York", "Shenzhen", "Bangalore",
    "Ho Chi Minh City", "Hyderabad", "Bogota", "Tianjin", "Santiago",
    "Sydney", "Berlin", "Madrid", "Toronto", "Johannesburg",
    "Dubai", "Singapore", "Tehran", "Baghdad", "Riyadh",
    "Rome", "Cape Town", "Lagos", "Casablanca", "Barcelona",
    "Seoul", "Melbourne", "Copenhagen", "Zurich", "Kuala Lumpur"
]


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

def load_existing_hashes(output_file: str, hash_file: str = 'dialogue_hashes.json') -> set:
    """
    Loads existing dialogue hashes from a hash file or the output JSON file.
    """
    if os.path.exists(hash_file):
        try:
            with open(hash_file, 'r', encoding='utf-8') as f:
                hashes = set(json.load(f))
            logger.info(f"Loaded {len(hashes)} existing dialogue hashes from '{hash_file}'.")
            return hashes
        except Exception as e:
            logger.warning(f"Could not load existing hashes: {e}")
    elif os.path.exists(output_file):
        # Fallback to existing output file
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_dialogues = json.load(f)
                hashes = set()
                for dialogue in existing_dialogues:
                    dialogue_text = dialogue.get('base_conversation', '')
                    dialogue_hash = hashlib.sha256(dialogue_text.encode('utf-8')).hexdigest()
                    hashes.add(dialogue_hash)
            logger.info(f"Loaded {len(hashes)} existing dialogue hashes from '{output_file}'.")
            # Save to hash file for future runs
            with open(hash_file, 'w', encoding='utf-8') as hf:
                json.dump(list(hashes), hf, indent=4)
            return hashes
        except Exception as e:
            logger.warning(f"Could not load existing dialogues: {e}")
    return set()

def load_existing_embeddings(embedding_file: str = 'dialogue_embeddings.npy') -> np.ndarray:
    """
    Loads existing dialogue embeddings from a file.
    """
    if os.path.exists(embedding_file):
        try:
            embeddings = np.load(embedding_file)
            logger.info(f"Loaded {embeddings.shape[0]} existing dialogue embeddings from '{embedding_file}'.")
            return embeddings
        except Exception as e:
            logger.warning(f"Could not load existing embeddings: {e}")
    return np.array([])

def save_embeddings(embeddings: np.ndarray, embedding_file: str = 'dialogue_embeddings.npy'):
    """
    Saves dialogue embeddings to a file.
    """
    try:
        np.save(embedding_file, embeddings)
        logger.info(f"Saved {embeddings.shape[0]} dialogue embeddings to '{embedding_file}'.")
    except Exception as e:
        logger.error(f"Failed to save embeddings to '{embedding_file}': {e}")

def is_unique(conversation_embedding: np.ndarray, existing_embeddings: np.ndarray, threshold: float = 0.9) -> bool:
    """
    Checks if the generated conversation is unique compared to existing embeddings.
    Returns True if unique, False otherwise.
    """
    if existing_embeddings.size == 0:
        return True
    similarities = cosine_similarity(conversation_embedding, existing_embeddings)
    max_similarity = similarities.max()
    if max_similarity >= threshold:
        return False
    return True

def assign_emotions(turns: List[Dict], selected_emotions: List[str]) -> List[Dict]:
    """
    Assigns emotions to each turn in the dialogue using an emotion classification model.
    Ensures that the emotions are among the selected emotions for the dialogue.
    """
    for turn in turns:
        utterance = turn['utterance']
        try:
            # Get emotion scores
            emotions = emotion_classifier(utterance)
            # Filter emotions to only include selected emotions
            filtered_emotions = [em for em in emotions[0] if em['label'] in selected_emotions]
            if not filtered_emotions:
                # If no selected emotions are detected, assign a random selected emotion
                primary_emotion = random.choice(selected_emotions)
            else:
                # Select the emotion with the highest score among the filtered emotions
                primary_emotion = max(filtered_emotions, key=lambda x: x['score'])['label']
            turn['emotion'] = primary_emotion
        except Exception as e:
            logger.error(f"Failed to assign emotion to utterance: '{utterance}'. Error: {e}")
            turn['emotion'] = "UNKNOWN"
    return turns

def generate_dynamic_scenario(category: str) -> str:
    """
    Generates a specific scenario based on the given category using OpenAI's API.
    """
    try:
        system_prompt = (
            "You are a creative assistant tasked with generating specific scenarios relevant to the given category. "
            "Each scenario should be detailed and fall under the provided category. "
            "Provide one unique scenario for the category."
        )
        user_prompt = f"Generate a detailed scenario for the category: {category}."

        response = client.chat.completions.create(
            model='gpt-4o-mini',  # Use a capable model for better scenario generation
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            n=1,
            stop=None,
        )

        scenario = response.choices[0].message.content.strip()
        logger.info(f"Generated scenario for category '{category}': {scenario}")
        return scenario

    except OpenAIError as e:
        logger.error(f"OpenAI API error during scenario generation for category '{category}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during scenario generation for category '{category}': {e}")
        return None

def generate_dialogue(service, prompt, min_turns, max_turns, 
                     temperature=0.9, top_p=0.95, frequency_penalty=0.5, presence_penalty=0.5,
                     max_retries=3, selected_emotions: List[str] = None, scenario: str = None,
                     regions: List[str] = None):
    """
    Generates a dialogue using OpenAI's chat completions API with uniqueness checks.
    Allows dynamic parameter tuning for the API call.
    """
    try:
        # Incorporate selected emotions, scenario, and regions into the system prompt
        emotions_str = ' and '.join(selected_emotions) if selected_emotions else ""
        scenario_str = f"The scenario is {scenario}." if scenario else ""
        regions_str = f"The dialogue is set in the following regions/areas: {', '.join(regions)}." if regions else ""
        system_prompt = (
            f"You are an expert dialogue generator for the '{service}' service. "
            f"{scenario_str} "
            f"{regions_str} "
            f"Create a high-quality, coherent, and emotionally rich dialogue between a user and an assistant. "
            f"The dialogue should express the following emotions: {emotions_str}. "
            f"The dialogue should have between {min_turns} and {max_turns} turns (a turn is one user message and one assistant response). "
            f"The dialogue should not be the same as any existing dialogues and should be better and more engaging. "
            f"Encourage diverse linguistic expressions and response styles to mimic real human interactions.\n\n"
            f"Please format the dialogue as follows, with each user message starting with 'User:' and each assistant response starting with 'Assistant:'.\n"
            f"Example:\n"
            f"User: Hello!\n"
            f"Assistant: Hi there! How can I assist you today?\n"
        )

        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model='gpt-4o-mini', 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=temperature,  # Dynamic temperature
                    top_p=top_p,              # Dynamic top_p
                    frequency_penalty=frequency_penalty,  # Dynamic frequency_penalty
                    presence_penalty=presence_penalty,    # Dynamic presence_penalty
                    n=1,  # Generate 1 completion at a time for quality control
                )
                generated_dialogues = [choice.message.content.strip() for choice in response.choices]

                for gen_dialogue in generated_dialogues:
                    # Check if the dialogue contains expected speaker labels
                    if re.search(r'^(User:|Assistant:)', gen_dialogue, re.MULTILINE):
                        return gen_dialogue  # Return the first valid formatted dialogue

                logger.warning(f"Attempt {attempt} - No valid dialogue found in generated completions.")
                if attempt < max_retries:
                    sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to generate properly formatted dialogue after {max_retries} attempts.")
                    return None
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
    parser = argparse.ArgumentParser(description="Generate dialogues using OpenAI API with advanced uniqueness checks, emotion assignment, dynamic parameter tuning, and dynamic scenario generation.")
    parser.add_argument('--num_generations', type=int, required=True, help="Number of dialogues to generate.")
    parser.add_argument('--min_turns', type=int, default=3, help="Minimum number of dialogue turns.")
    parser.add_argument('--max_turns', type=int, default=10, help="Maximum number of dialogue turns.")
    parser.add_argument('--output_file', type=str, default='generated_dialogues.json', help="Output JSON file path.")
    # Dynamic parameters
    parser.add_argument('--temperature', type=float, nargs='+', default=[0.7, 0.8, 0.9, 1.0], help="Temperature values for OpenAI API.")
    parser.add_argument('--top_p', type=float, nargs='+', default=[0.8, 0.85, 0.9, 0.95, 1.0], help="Top_p values for OpenAI API.")
    parser.add_argument('--frequency_penalty', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.7], help="Frequency penalty values for OpenAI API.")
    parser.add_argument('--presence_penalty', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.7], help="Presence penalty values for OpenAI API.")
    parser.add_argument('--similarity_threshold', type=float, default=0.9, help="Cosine similarity threshold for uniqueness.")
    parser.add_argument('--embedding_file', type=str, default='dialogue_embeddings.npy', help="File to store dialogue embeddings.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    num_generations = args.num_generations
    min_turns = args.min_turns
    max_turns = args.max_turns
    output_file = args.output_file

    # Dynamic parameters
    temperature_options = args.temperature
    top_p_options = args.top_p
    frequency_penalty_options = args.frequency_penalty
    presence_penalty_options = args.presence_penalty
    similarity_threshold = args.similarity_threshold
    embedding_file = args.embedding_file

    logger.info("Starting dialogue generation...")
    logger.info(f"Parameters: num_generations={num_generations}, min_turns={min_turns}, max_turns={max_turns}, output_file='{output_file}'")
    logger.info(f"Dynamic Parameters: temperature={temperature_options}, top_p={top_p_options}, frequency_penalty={frequency_penalty_options}, presence_penalty={presence_penalty_options}")
    logger.info(f"Similarity Threshold: {similarity_threshold}")
    logger.info(f"Embedding File: '{embedding_file}'")

    # Load dataset from Hugging Face
    try:
        dataset = load_dataset('Ayushnangia/transport_multiwoz_v22')
        data_split = dataset['train']
        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Load existing dialogues and their hashes
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
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

    # Load existing hashes
    existing_hashes = load_existing_hashes(output_file, 'dialogue_hashes.json')

    # Load existing embeddings
    existing_embeddings = load_existing_embeddings(embedding_file)

    # Prepare list to collect new dialogues
    new_dialogues = []

    # Randomly select examples to generate dialogues for
    if num_generations > len(data_split):
        logger.error("Number of generations requested exceeds the dataset size.")
        return

    selected_indices = random.sample(range(len(data_split)), num_generations)

    for index in tqdm(selected_indices, desc="Generating dialogues"):
        example = data_split[index]
        services = example.get('services', [])
        dialogue_id = example.get('dialogue_id', f"dialogue_{index}")

        # *** Assign a Region from the Predefined List ***
        assigned_region = random.choice(PREDEFINED_REGIONS)
        logger.info(f"Assigned region for dialogue_id '{dialogue_id}': {assigned_region}")
        regions = [assigned_region]  # List to hold regions; can be extended if needed

        # Extract and anonymize existing dialogue
        processed_dialogue = extract_and_anonymize_dialogue(example)
        base_conversation = generate_base_conversation(processed_dialogue)

        # Create hash of the base conversation to check for duplicates
        dialogue_hash = hashlib.sha256(base_conversation.encode('utf-8')).hexdigest()
        if dialogue_hash in existing_hashes:
            logger.info(f"Duplicate dialogue detected for dialogue_id '{dialogue_id}'. Skipping.")
            continue

        # *** Scenario Diversification: Dynamically generate a scenario for this dialogue ***
        selected_category = random.choice(SCENARIO_CATEGORIES)
        generated_scenario = generate_dynamic_scenario(selected_category)
        if not generated_scenario:
            logger.warning(f"Could not generate scenario for category '{selected_category}'. Skipping dialogue_id '{dialogue_id}'.")
            continue
        logger.info(f"Selected category for dialogue_id '{dialogue_id}': {selected_category}")
        logger.info(f"Generated scenario for dialogue_id '{dialogue_id}': {generated_scenario}")

        # Randomly select 2 emotions from the EMOTION_LIST
        selected_emotions = random.sample(EMOTION_LIST, 2)
        logger.info(f"Selected emotions for dialogue_id '{dialogue_id}': {selected_emotions}")

        prompt = (
            f"Using the following base conversation as a reference, create a new dialogue for the service(s): {', '.join(services)}. "
            f"The dialogue should be completely new and more relevant than any existing dialogue. Do not copy any part of existing dialogues. "
            f"The dialogue should be between a user and an assistant.\n\n"
            f"Base Conversation:\n{base_conversation}"
        )

        # Randomly select dynamic parameters for this generation
        temperature = random.choice(temperature_options)
        top_p = random.choice(top_p_options)
        frequency_penalty = random.choice(frequency_penalty_options)
        presence_penalty = random.choice(presence_penalty_options)

        generated_dialogue = generate_dialogue(
            services[0] if services else "bus", 
            prompt, 
            min_turns, 
            max_turns, 
            temperature=temperature, 
            top_p=top_p, 
            frequency_penalty=frequency_penalty, 
            presence_penalty=presence_penalty,
            selected_emotions=selected_emotions,  # Pass selected emotions to the dialogue generator
            scenario=generated_scenario,            # Pass generated scenario to the dialogue generator
            regions=regions                         # Pass assigned regions to the dialogue generator
        )

        if generated_dialogue:
            generated_turns = process_generated_dialogue(generated_dialogue)
            # Assign emotions to each turn based on selected emotions
            generated_turns = assign_emotions(generated_turns, selected_emotions)
            generated_conversation = generate_base_conversation(generated_turns)
            generated_hash = hashlib.sha256(generated_conversation.encode('utf-8')).hexdigest()

            if generated_hash in existing_hashes:
                logger.warning(f"Generated dialogue is a duplicate for dialogue_id '{dialogue_id}'. Skipping.")
                continue

            # Generate embedding for the new conversation
            try:
                conversation_embedding = embedding_model.encode(generated_conversation, convert_to_numpy=True).reshape(1, -1)
            except Exception as e:
                logger.error(f"Failed to generate embedding for dialogue_id '{dialogue_id}': {e}")
                continue

            # Check for semantic uniqueness
            if not is_unique(conversation_embedding, existing_embeddings, threshold=similarity_threshold):
                logger.warning(f"Generated dialogue is too similar to existing dialogues for dialogue_id '{dialogue_id}'. Skipping.")
                continue

            # Update existing_embeddings with the new embedding
            existing_embeddings = np.vstack([existing_embeddings, conversation_embedding]) if existing_embeddings.size else conversation_embedding

            # Count number of lines in the conversation
            num_lines = len(generated_turns)

            new_dialogue_id = f"{dialogue_id}_generated_{index}"
            if new_dialogue_id in existing_ids:
                logger.warning(f"Duplicate dialogue_id '{new_dialogue_id}' found. Skipping.")
                continue

            new_dialogues.append({
                'services': services,
                'dialogue_id': new_dialogue_id,
                'turns': generated_turns,
                'base_conversation': generated_conversation,
                'num_lines': num_lines,  # Added number of lines
                'emotions': selected_emotions,  # Record the selected emotions
                'scenario': selected_category,    # Record the selected category
                'generated_scenario': generated_scenario,  # Record the generated scenario
                'regions': regions                # Added regions information
            })
            existing_ids.add(new_dialogue_id)
            existing_hashes.add(generated_hash)

    logger.info("Dialogue generation complete.")

    # Combine existing and new dialogues
    all_dialogues = existing_dialogues + new_dialogues

    # Save the new dialogues to a JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_dialogues, f, indent=4, ensure_ascii=False)
        logger.info(f"Generated dialogues saved to '{output_file}'. Total dialogues: {len(all_dialogues)}.")
    except Exception as e:
        logger.error(f"Failed to save dialogues to '{output_file}': {e}")

    # Save the updated embeddings
    save_embeddings(existing_embeddings, embedding_file)

    # Update the dialogue hashes
    try:
        with open('dialogue_hashes.json', 'w', encoding='utf-8') as hf:
            json.dump(list(existing_hashes), hf, indent=4)
        logger.info(f"Updated 'dialogue_hashes.json' with {len(existing_hashes)} hashes.")
    except Exception as e:
        logger.error(f"Failed to update 'dialogue_hashes.json': {e}")

if __name__ == "__main__":
    main()
