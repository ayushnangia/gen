# dialogue_generation.py

import os
import json
import random
import logging
from time import sleep
from typing import List, Dict, Tuple
from datasets import load_dataset
from openai import OpenAI, OpenAIError
from tqdm import tqdm
import spacy
from dotenv import load_dotenv
import hashlib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import uuid

class DialogueGenerator:
    def __init__(self, config: Dict):
        self.output_file = config.get('output_file', 'generated_dialogues.json')
        self.hash_file = config.get('hash_file', 'dialogue_hashes.json')
        self.embedding_file = config.get('embedding_file', 'dialogue_embeddings.npy')
        self.similarity_threshold = config.get('similarity_threshold', 0.9)
        self.dataset_name = config.get('dataset_name', 'Ayushnangia/transport_multiwoz_v22')
        self.min_turns_range = config.get('min_turns_range', (5, 10))
        self.max_turns_range = config.get('max_turns_range', (7, 20))
        self.temperature_range = config.get('temperature_range', (0.7, 1.0))
        self.top_p_range = config.get('top_p_range', (0.8, 1.0))
        self.frequency_penalty_range = config.get('frequency_penalty_range', (0.0, 0.7))
        self.presence_penalty_range = config.get('presence_penalty_range', (0.0, 0.7))
        self.total_generations = config.get('total_generations', 10)

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler("dialogue_generation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Load spaCy's English model for NER
        try:
            self.logger.info("Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.info("spaCy model not found. Downloading 'en_core_web_sm'...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Load environment variables
        load_dotenv('../.env.local')

        # Initialize OpenAI client
        openai_api_key = os.getenv('OPENAI_KEY')
        if not openai_api_key:
            self.logger.error("OPENAI_API_KEY environment variable not set.")
            raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=openai_api_key)

        # Initialize Sentence Transformer model for embeddings
        try:
            self.logger.info("Loading SentenceTransformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise e

        # Define emotion lists
        self.USER_EMOTION_LIST = [
            "Frustration", "Confusion", "Gratitude", "Impatience", "Disappointment",
            "Relief", "Annoyance", "Appreciation", "Curiosity", "Urgency",
            "Hesitation", "Hopefulness", "Satisfaction", "Concern", "Suspicion",
            "Irritation", "Interest", "Indifference", "Acceptance", "Disbelief"
        ]

        self.ASSISTANT_EMOTION_LIST = [
            "Empathy", "Politeness", "Reassurance", "Understanding", "Patience",
            "Sympathy", "Encouragement", "Clarification", "Helpfulness", "Professionalism",
            "Confidence", "Calmness", "Supportiveness", "Attentiveness",
            "Apologetic", "Proactive", "Respectfulness", "Neutrality", "Cheerfulness"
        ]

        # Scenario categories
        self.SCENARIO_CATEGORIES = [
            "booking",
            "cancellation",
            "complaint",
            "feedback",
            "inquiry",
            "rescheduling",
            "issue reporting",
            "assistance seeking",
            "personal details update",
            "refund request",
            "payment issues",
            "review",
             
        ]


        # Predefined regions
        self.PREDEFINED_REGIONS = [
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

        # Load dataset once during initialization
        self.dataset = self.load_dataset()

        # Load existing dialogues and hashes
        self.existing_dialogues, self.existing_ids = self.load_existing_dialogues()
        self.existing_hashes = self.load_existing_hashes()
        self.existing_embeddings = self.load_existing_embeddings()

    def load_dataset(self) -> any:
        """
        Loads the dataset from Hugging Face once during initialization.
        """
        try:
            self.logger.info(f"Loading dataset '{self.dataset_name}' from Hugging Face...")
            dataset = load_dataset(self.dataset_name)
            self.logger.info("Dataset loaded successfully.")
            return dataset['train']
        except Exception as e:
            self.logger.error(f"Failed to load dataset '{self.dataset_name}': {e}")
            raise e

    def anonymize_text(self, text: str) -> str:
        """
        Anonymize specific entities in the text such as locations, times, and numbers.
        """
        doc = self.nlp(text)
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

    def extract_and_anonymize_dialogue(self, dialogue_json: Dict) -> List[Dict]:
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

            anonymized_utterance = self.anonymize_text(utterance)

            turns.append({
                "turn_id": turn_id,
                "speaker": speaker_label,
                "utterance": anonymized_utterance
            })

        return turns

    def generate_base_conversation(self, turns: List[Dict]) -> str:
        """
        Formats the list of turns into a base conversation string.
        """
        conversation = ""
        for turn in turns:
            conversation += f"{turn['speaker']}: {turn['utterance']}\n"
        return conversation.strip()

    def load_existing_dialogues(self) -> Tuple[List[Dict], set]:
        """
        Loads existing dialogues and their IDs from the output file.
        """
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_dialogues = json.load(f)
                    existing_ids = {dialogue['dialogue_id'] for dialogue in existing_dialogues}
                self.logger.info(f"Loaded {len(existing_dialogues)} existing dialogues from '{self.output_file}'.")
                return existing_dialogues, existing_ids
            except Exception as e:
                self.logger.warning(f"Could not load existing dialogues: {e}")
        return [], set()

    def load_existing_hashes(self) -> set:
        """
        Loads existing dialogue hashes from a hash file or the output JSON file.
        """
        if os.path.exists(self.hash_file):
            try:
                with open(self.hash_file, 'r', encoding='utf-8') as f:
                    hashes = set(json.load(f))
                self.logger.info(f"Loaded {len(hashes)} existing dialogue hashes from '{self.hash_file}'.")
                return hashes
            except Exception as e:
                self.logger.warning(f"Could not load existing hashes: {e}")
        elif os.path.exists(self.output_file):
            # Fallback to existing output file
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_dialogues = json.load(f)
                    hashes = set()
                    for dialogue in existing_dialogues:
                        dialogue_text = dialogue.get('base_conversation', '')
                        dialogue_hash = hashlib.sha256(dialogue_text.encode('utf-8')).hexdigest()
                        hashes.add(dialogue_hash)
                self.logger.info(f"Loaded {len(hashes)} existing dialogue hashes from '{self.output_file}'.")
                # Save to hash file for future runs
                with open(self.hash_file, 'w', encoding='utf-8') as hf:
                    json.dump(list(hashes), hf, indent=4)
                return hashes
            except Exception as e:
                self.logger.warning(f"Could not load existing dialogues: {e}")
        return set()

    def load_existing_embeddings(self) -> np.ndarray:
        """
        Loads existing dialogue embeddings from a file.
        """
        if os.path.exists(self.embedding_file):
            try:
                embeddings = np.load(self.embedding_file)
                self.logger.info(f"Loaded {embeddings.shape[0]} existing dialogue embeddings from '{self.embedding_file}'.")
                return embeddings
            except Exception as e:
                self.logger.warning(f"Could not load existing embeddings: {e}")
        return np.array([])

    def save_embeddings(self, embeddings: np.ndarray):
        """
        Saves dialogue embeddings to a file.
        """
        try:
            np.save(self.embedding_file, embeddings)
            self.logger.info(f"Saved {embeddings.shape[0]} dialogue embeddings to '{self.embedding_file}'.")
        except Exception as e:
            self.logger.error(f"Failed to save embeddings to '{self.embedding_file}': {e}")

    def is_unique(self, conversation_embedding: np.ndarray) -> bool:
        """
        Checks if the generated conversation is unique compared to existing embeddings.
        Returns True if unique, False otherwise.
        """
        if self.existing_embeddings.size == 0:
            return True
        similarities = cosine_similarity(conversation_embedding, self.existing_embeddings)
        max_similarity = similarities.max()
        if max_similarity >= self.similarity_threshold:
            return False
        return True

    def assign_selected_emotions(self, turns: List[Dict], user_emotions: List[str], assistant_emotions: List[str]) -> List[Dict]:
        """
        Assigns emotions to each turn in the dialogue based on the speaker.
        """
        for turn in turns:
            if turn['speaker'] == 'USER':
                turn['emotion'] = random.choice(user_emotions)
            elif turn['speaker'] == 'ASSISTANT':
                turn['emotion'] = random.choice(assistant_emotions)
            else:
                turn['emotion'] = "NEUTRAL"  # Default emotion for unknown speakers
        return turns

    def generate_dynamic_scenario(self, category: str, service: str) -> str:
        """
        Generates a specific scenario based on the given category and service using OpenAI's API.
        Ensures the scenario is relevant to the service and limited to 2-3 lines.
        """
        try:
            system_prompt = (
                "You are a creative assistant tasked with generating specific scenarios relevant to the given category and service. "
                "Each scenario should be detailed, pertinent to the provided service, and confined to 2-3 lines. "
                "Provide one unique scenario for the category and service."
            )
            user_prompt = f"Generate a concise (2-3 lines) scenario for the category: '{category}' and transport service: '{service}'.\n Please ensure that the traspost service is always kept in mind "

            response = self.client.chat.completions.create(
                model='gpt-4o-mini',  # Ensure this is the correct model name
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,  # Increased tokens to accommodate instructions
                temperature=0.7,
                top_p=0.9,
                n=1,
                stop=None,
            )

            scenario = response.choices[0].message.content.strip()
            self.logger.info(f"Generated scenario for category '{category}' and service '{service}': {scenario}")
            return scenario

        except OpenAIError as e:
            self.logger.error(f"OpenAI API error during scenario generation for category '{category}' and service '{service}': {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during scenario generation for category '{category}' and service '{service}': {e}")
            return None

    def generate_dialogue(self, service: str, prompt: str, min_turns: int, max_turns: int, 
                         temperature: float = 0.9, top_p: float = 0.95, frequency_penalty: float = 0.5, presence_penalty: float = 0.5,
                         max_retries: int = 3, user_emotions: List[str] = None, assistant_emotions: List[str] = None, 
                         scenario: str = None, regions: List[str] = None) -> str:
        """
        Generates a dialogue using OpenAI's chat completions API with uniqueness checks.
        Allows dynamic parameter tuning for the API call.
        """
        try:
            # Incorporate selected emotions, scenario, and regions into the system prompt
            user_emotions_str = ' and '.join(user_emotions) if user_emotions else ""
            assistant_emotions_str = ' and '.join(assistant_emotions) if assistant_emotions else ""
            scenario_str = f"The scenario is: {scenario}" if scenario else ""
            regions_str = f"The dialogue is set in the following region: {', '.join(regions)}." if regions else ""
            system_prompt = (
                f"You are an expert dialogue generator for the '{service}' service. "
                f"{scenario_str} "
                f"{regions_str} "
                f"Create a high-quality, coherent, and emotionally rich dialogue between a user and an assistant. "
                f"The user should express the following emotions: {user_emotions_str}. "
                f"The assistant should express the following emotions: {assistant_emotions_str}. "
                f"The dialogue should have between {min_turns} and {max_turns} turns (a turn is one user message and one assistant response). "
                f"The dialogue should not be the same as any existing dialogues and should be better and more engaging. "
                f"Make sure that these dialogues are representative of converstation between a text/voice interface hence the assitant might not be present."
                f"These diagloues should encompass so as to showcase the different senarios and outcomes in a stricly transport service setting."
                f"Encourage diverse linguistic expressions and response styles to mimic real human interactions.\n\n"
                f"Please format the dialogue as follows, with each user message starting with 'User:' and each assistant response starting with 'Assistant:'.\n"
                f"Example:\n"
                f"User: Hello!\n"
                f"Assistant: Hi there! How can I assist you today?\n"
            )

            for attempt in range(1, max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model='gpt-4o-mini',  # Ensure this is the correct model name
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

                    self.logger.warning(f"Attempt {attempt} - No valid dialogue found in generated completions.")
                    if attempt < max_retries:
                        sleep(2 ** attempt)  # Exponential backoff
                    else:
                        self.logger.error(f"Failed to generate properly formatted dialogue after {max_retries} attempts.")
                        return None
                except OpenAIError as e:
                    self.logger.warning(f"Attempt {attempt} - OpenAI API error: {e}")
                    if attempt < max_retries:
                        sleep(2 ** attempt)  # Exponential backoff
                    else:
                        self.logger.error(f"Failed after {max_retries} attempts.")
                        return None
        except Exception as e:
            self.logger.error(f"Unexpected error in generate_dialogue: {e}")
            return None

    def process_generated_dialogue(self, generated_dialogue: str) -> List[Dict]:
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

    def generate_unique_dialogues(self, num_generations: int, min_turns: int, max_turns: int,
                                  temperature_options: List[float], top_p_options: List[float],
                                  frequency_penalty_options: List[float], presence_penalty_options: List[float]) -> List[Dict]:
        """
        Generates a specified number of unique dialogues.
        """
        if len(self.dataset) == 0:
            self.logger.error("Dataset is empty. Cannot generate dialogues.")
            return []

        # Sampling with replacement to allow num_generations > len(self.dataset)
        selected_indices = random.choices(range(len(self.dataset)), k=num_generations)
        new_dialogues = []
        batch_size = 10000  # Define a batch size suitable for your system

        for idx, dataset_index in enumerate(tqdm(selected_indices, desc="Generating dialogues")):
            example = self.dataset[dataset_index]
            services = example.get('services', [])
            original_dialogue_id = example.get('dialogue_id', f"dialogue_{dataset_index}")

            # Assign a Region from the Predefined List
            assigned_region = random.choice(self.PREDEFINED_REGIONS)
            self.logger.info(f"Assigned region for dialogue_id '{original_dialogue_id}': {assigned_region}")
            regions = [assigned_region]

            # Extract and anonymize existing dialogue
            processed_dialogue = self.extract_and_anonymize_dialogue(example)
            base_conversation = self.generate_base_conversation(processed_dialogue)
            self.logger.info(f"base conver{base_conversation}")
            # Create hash of the base conversation to check for duplicates
            dialogue_hash = hashlib.sha256(base_conversation.encode('utf-8')).hexdigest()
            if dialogue_hash in self.existing_hashes:
                self.logger.info(f"Duplicate dialogue detected for dialogue_id '{original_dialogue_id}'. Skipping.")
                continue

            # Scenario Diversification: Dynamically generate a scenario for this dialogue
            primary_service = services[0] if services else "general"

            selected_category = random.choice(self.SCENARIO_CATEGORIES)
            generated_scenario = self.generate_dynamic_scenario(selected_category, primary_service)
            if not generated_scenario:
                self.logger.warning(f"Could not generate scenario for category '{selected_category}' and service '{primary_service}'. Skipping dialogue_id '{original_dialogue_id}'.")
                continue
            self.logger.info(f"Selected category for dialogue_id '{original_dialogue_id}': {selected_category}")
            self.logger.info(f"Generated scenario for dialogue_id '{original_dialogue_id}': {generated_scenario}")

            # Randomly select emotions from the respective emotion lists
            selected_user_emotions = random.sample(self.USER_EMOTION_LIST, 1)
            selected_assistant_emotions = random.sample(self.ASSISTANT_EMOTION_LIST, 1)
            self.logger.info(f"Selected user emotions for dialogue_id '{original_dialogue_id}': {selected_user_emotions}")
            self.logger.info(f"Selected assistant emotions for dialogue_id '{original_dialogue_id}': {selected_assistant_emotions}")

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

            generated_dialogue = self.generate_dialogue(
                primary_service,
                prompt, 
                min_turns, 
                max_turns, 
                temperature=temperature, 
                top_p=top_p, 
                frequency_penalty=frequency_penalty, 
                presence_penalty=presence_penalty,
                user_emotions=selected_user_emotions,
                assistant_emotions=selected_assistant_emotions,
                scenario=generated_scenario,
                regions=regions
            )

            if generated_dialogue:
                generated_turns = self.process_generated_dialogue(generated_dialogue)
                # Assign emotions to each turn based on the speaker
                generated_turns = self.assign_selected_emotions(generated_turns, selected_user_emotions, selected_assistant_emotions)
                # Generate conversation text without including it in the JSON
                generated_conversation_text = "\n".join([f"{turn['speaker']}: {turn['utterance']}" for turn in generated_turns])
                generated_hash = hashlib.sha256(generated_conversation_text.encode('utf-8')).hexdigest()

                if generated_hash in self.existing_hashes:
                    self.logger.warning(f"Generated dialogue is a duplicate based on hash for dialogue_id '{original_dialogue_id}'. Skipping.")
                    continue

                # Generate embedding for the new conversation
                try:
                    conversation_embedding = self.embedding_model.encode(generated_conversation_text, convert_to_numpy=True).reshape(1, -1)
                except Exception as e:
                    self.logger.error(f"Failed to generate embedding for dialogue_id '{original_dialogue_id}': {e}")
                    continue

                # Check for semantic uniqueness
                if not self.is_unique(conversation_embedding):
                    self.logger.warning(f"Generated dialogue is too similar to existing dialogues for dialogue_id '{original_dialogue_id}'. Skipping.")
                    continue

                # Update existing_embeddings with the new embedding
                if self.existing_embeddings.size:
                    self.existing_embeddings = np.vstack([self.existing_embeddings, conversation_embedding])
                else:
                    self.existing_embeddings = conversation_embedding

                # Count number of lines in the conversation
                num_lines = len(generated_turns)

                # **Generate a UUID-based dialogue_id**
                unique_id = uuid.uuid4()
                new_dialogue_id = f"{original_dialogue_id}_generated_{unique_id}"
                # Although UUID collisions are highly unlikely, implement a check
                while new_dialogue_id in self.existing_ids:
                    self.logger.warning(f"Duplicate dialogue_id '{new_dialogue_id}' found. Regenerating UUID.")
                    unique_id = uuid.uuid4()
                    new_dialogue_id = f"{original_dialogue_id}_generated_{unique_id}"

                new_dialogues.append({
                    'services': services,
                    'dialogue_id': new_dialogue_id,
                    'turns': generated_turns,
                    'num_lines': num_lines,  # Added number of lines
                    'user_emotions': selected_user_emotions,      # Record the selected user emotions
                    'assistant_emotions': selected_assistant_emotions,  # Record the selected assistant emotions
                    'scenario_category': selected_category,         # Record the selected category
                    'generated_scenario': generated_scenario,       # Record the generated scenario
                    'regions': regions                              # Added regions information
                })
                self.existing_ids.add(new_dialogue_id)
                self.existing_hashes.add(generated_hash)

                # **Incremental Saving: Save in Batches**
                if len(new_dialogues) >= 10000:
                    self.save_new_dialogues(new_dialogues)
                    self.logger.info(f"Saved a batch of {len(new_dialogues)} dialogues.")
                    new_dialogues = []

        # Save any remaining dialogues after the loop
        if new_dialogues:
            self.save_new_dialogues(new_dialogues)
            self.logger.info(f"Saved the final batch of {len(new_dialogues)} dialogues.")

        self.logger.info("Dialogue generation complete.")
        return []

    def save_new_dialogues(self, new_dialogues: List[Dict]):
        """
        Saves new dialogues to the output file and updates embeddings and hashes.
        """
        # Load existing dialogues if the output file exists
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    all_dialogues = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load existing dialogues from '{self.output_file}': {e}")
                all_dialogues = []
        else:
            all_dialogues = []

        # Combine existing and new dialogues
        all_dialogues.extend(new_dialogues)

        # Save the combined dialogues to the output file
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(all_dialogues, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Saved {len(new_dialogues)} new dialogues to '{self.output_file}'. Total dialogues: {len(all_dialogues)}.")
        except Exception as e:
            self.logger.error(f"Failed to save dialogues to '{self.output_file}': {e}")

        # Save the updated embeddings
        self.save_embeddings(self.existing_embeddings)

        # Update the dialogue hashes
        try:
            with open(self.hash_file, 'w', encoding='utf-8') as hf:
                json.dump(list(self.existing_hashes), hf, indent=4)
            self.logger.info(f"Updated '{self.hash_file}' with {len(self.existing_hashes)} hashes.")
        except Exception as e:
            self.logger.error(f"Failed to update '{self.hash_file}': {e}")
