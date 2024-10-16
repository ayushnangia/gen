# dialogue_generation.py

import os
import json
import random
import logging
from time import sleep
from typing import List, Dict, Tuple
from datasets import load_dataset, DatasetDict
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
import argparse
import argilla as rg
from argilla.client import Argilla



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
        self.argilla_api_key = None
        self.argilla_api_url = None

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,  # Change to DEBUG for more detailed logs
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler("dialogue_generation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        try:
            self.logger.info("Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.info("spaCy model not found. Downloading 'en_core_web_sm'...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Load environment variables
        load_dotenv('.env.local')

        # Initialize OpenAI client
        openai_api_key = os.getenv('OPENAI_KEY')
        self.argilla_api_key = os.getenv('AGAPI_KEY')
        self.argilla_api_url = os.getenv('AGAPI_URL')
        if not openai_api_key:
            self.logger.error("OPENAI_API_KEY environment variable not set.")
            raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=openai_api_key)

        # Initialize Sentence Transformer model for embeddings
        try:
            self.logger.info("Loading SentenceTransformer model...")
            model_path = './models/sentence_transformer'
            self.embedding_model = SentenceTransformer(model_path)
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
            "booking", "cancellation", "complaint", "feedback", "inquiry",
            "rescheduling", "issue reporting", "assistance seeking",
            "personal details update", "refund request", "payment issues", "review",
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
        self.existing_ids = self.load_existing_dialogues()
        self.existing_hashes = self.load_existing_hashes()
        self.existing_embeddings = self.load_existing_embeddings()
        self.argilla_client = Argilla(api_key=self.argilla_api_key, api_url=self.argilla_api_url)

        # Initialize Argilla
        self.argilla_dataset = self.init_argilla_dataset()
    def init_argilla_dataset(self):
        """
        Initialize Argilla dataset for logging dialogues.
        """
        dataset_name = "synthetic_dialogues"
        try:
            # Try to retrieve the existing dataset
            dataset = self.argilla_client.datasets(name=dataset_name)
            if dataset is None:
                # If the dataset doesn't exist, create it
                settings = rg.Settings(
                    guidelines="Synthetic dialogues for transport services.",
                    fields=[
                        rg.TextField(
                            name="dialogue",
                            title="Generated Dialogue",
                        ),
                    ],
                    questions=[
                        rg.LabelQuestion(
                            name="scenario_category",
                            title="Scenario Category",
                            labels=self.SCENARIO_CATEGORIES,
                        )
                    ],
                    metadata=[
                        rg.TermsMetadataProperty(name="services"),
                        rg.TermsMetadataProperty(name="user_emotions"),
                        rg.TermsMetadataProperty(name="assistant_emotions"),
                        rg.TermsMetadataProperty(name="regions"),
                    ],
                    allow_extra_metadata=True
                )
                
                dataset = rg.Dataset(
                    name=dataset_name,
                    workspace="argilla",  # Change this to your workspace if different
                    settings=settings,
                )
                dataset.create()
                self.logger.info(f"Argilla dataset '{dataset_name}' created successfully.")
            else:
                self.logger.info(f"Argilla dataset '{dataset_name}' already exists.")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to initialize Argilla dataset: {e}")
            return None
    def log_to_argilla(self, dialogue: Dict):
        """
        Log a generated dialogue to Argilla.
        """
        if self.argilla_dataset is None:
            self.logger.error("Argilla dataset is not initialized. Cannot log dialogue.")
            return

        try:
            dialogue_text = "\n".join([f"{turn['speaker']}: {turn['utterance']}" for turn in dialogue['turns']])
            # self.logger.info(f"{dialogue_text}")
            record = rg.Record(
                fields={
                    "dialogue": "dialogue_text",
                },
                metadata={
                    "dialogue_id": dialogue['dialogue_id'],
                    "services": dialogue['services'],
                    "num_lines": dialogue['num_lines'],
                    "user_emotions": dialogue['user_emotions'],
                    "assistant_emotions": dialogue['assistant_emotions'],
                    "generated_scenario": dialogue['generated_scenario'],
                    "regions": dialogue['regions'],
                    "scenario_category": dialogue['scenario_category']
                }
            )
            self.argilla_dataset.records.log([record])
            self.logger.info(f"Successfully logged dialogue {dialogue['dialogue_id']} to Argilla.")
        except Exception as e:
            self.logger.error(f"Failed to log dialogue to Argilla: {e}")

    def load_dataset(self) -> any:
        """
        Loads the dataset from Hugging Face once during initialization.
        """
        try:
            self.logger.info(f"Loading dataset from local path: ./local_datasets/transport_multiwoz_v22")
            dataset = DatasetDict.load_from_disk("./local_datasets/transport_multiwoz_v22")
            self.logger.info("Dataset loaded successfully.")
            self.logger.info(f"Number of dialogues in 'train' split: {len(dataset['train'])}")
            return dataset['train']
        except Exception as e:
            self.logger.error(f"Failed to load dataset from ./local_datasets/transport_multiwoz_v22: {e}")
            raise e

    def extract_dialogue(self, dialogue_json: Dict) -> List[Dict]:
        """
        Extracts turns from the dialogue JSON and processes the utterances.
        Returns a list of turns with processed utterances.
        """
        turns = []
        dialogue_id = dialogue_json.get("dialogue_id", "unknown")
        turns_data = dialogue_json.get("turns", {})
        
        speakers = turns_data.get("speaker", [])
        utterances = turns_data.get("utterance", [])
        turn_ids = turns_data.get("turn_id", [])

        self.logger.info(f"Processing dialogue_id: {dialogue_id}")
        self.logger.info(f"Number of speakers: {len(speakers)}, utterances: {len(utterances)}, turn_ids: {len(turn_ids)}")

        for turn_id, speaker, utterance in zip(turn_ids, speakers, utterances):
            if speaker == 0:
                speaker_label = "USER"
            elif speaker == 1:
                speaker_label = "ASSISTANT"
            else:
                speaker_label = "UNKNOWN"

            turns.append({
                "turn_id": turn_id,
                "speaker": speaker_label,
                "utterance": utterance
            })

        self.logger.info(f"Extracted {len(turns)} turns from dialogue_id '{dialogue_id}'.")
        return turns

    def generate_base_conversation(self, turns: List[Dict]) -> str:
        """
        Formats the list of turns into a base conversation string.
        """
        conversation_lines = []
        for turn in turns:
            speaker = turn.get('speaker', 'UNKNOWN')
            utterance = turn.get('utterance', '')
            conversation_lines.append(f"{speaker}: {utterance}")
        
        base_conversation = "\n".join(conversation_lines)
        return base_conversation

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
                return existing_ids
            except Exception as e:
                self.logger.warning(f"Could not load existing dialogues: {e}")
        return set()

    def load_existing_hashes(self) -> set:
        """
        Loads existing dialogue hashes from a hash file or the output JSON file.
        """
        if os.path.exists(self.hash_file):
            try:
                with open(self.hash_file, 'r', encoding='utf-8') as f:
                    hashes = set(json.load(f))
                return hashes
            except Exception as e:
                self.logger.warning(f"Could not load existing hashes: {e}")
        elif os.path.exists(self.output_file):
            # Fallback to existing output file
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
            user_prompt = f"Generate a concise (2-3 lines) scenario for the category: '{category}' and transport service: '{service}'.\nPlease ensure that the transport service is always kept in mind."

            response = self.client.chat.completions.create(
                model='gpt-4o-mini',  # Corrected model name
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
                         temperature: float = 0.9, top_p: float = 0.95, frequency_penalty: float = 0.5, presence_penalty: float = 0.5
    ) -> List[Dict]:
        """
        Generates a dialogue using OpenAI's API based on the given prompt and parameters.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI assistant generating a dialogue between a user and a customer service representative for a transport service. The dialogue should be natural, informative, and relevant to the given scenario."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=1,
                stop=None,
            )

            generated_text = response.choices[0].message.content.strip()
            turns = self.process_generated_dialogue(generated_text)

            if len(turns) < min_turns or len(turns) > max_turns:
                self.logger.warning(f"Generated dialogue has {len(turns)} turns, which is outside the specified range of {min_turns}-{max_turns}.")
                return None

            return turns

        except OpenAIError as e:
            self.logger.error(f"OpenAI API error during dialogue generation: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during dialogue generation: {e}")
            return None

    def process_generated_dialogue(self, generated_text: str) -> List[Dict]:
        """
        Processes the generated dialogue text into a list of turns.
        """
        lines = generated_text.split('\n')
        turns = []
        current_speaker = None
        current_utterance = []

        for line in lines:
            line = line.strip()
            if line.startswith("USER:") or line.startswith("ASSISTANT:"):
                if current_speaker and current_utterance:
                    turns.append({
                        "speaker": current_speaker,
                        "utterance": " ".join(current_utterance).strip()
                    })
                current_speaker = "USER" if line.startswith("USER:") else "ASSISTANT"
                current_utterance = [line.split(":", 1)[1].strip()]
            elif line:
                current_utterance.append(line)

        if current_speaker and current_utterance:
            turns.append({
                "speaker": current_speaker,
                "utterance": " ".join(current_utterance).strip()
            })

        return turns

    def generate_unique_dialogues(self, num_generations: int, min_turns: int, max_turns: int,
                                  temperature_options: List[float], top_p_options: List[float],
                                  frequency_penalty_options: List[float], presence_penalty_options: List[float]) -> List[Dict]:
        new_dialogues = []
        new_hashes = set()
        new_embeddings = []

        # Select random indices from the dataset
        dataset_size = len(self.dataset)
        selected_indices = random.sample(range(dataset_size), num_generations)

        for idx, dataset_index in enumerate(tqdm(selected_indices, desc="Generating dialogues")):
            dialogue_json = self.dataset[dataset_index]
            services = dialogue_json.get('services', [])
            
            if not services:
                self.logger.warning(f"No services found for dialogue at index {dataset_index}. Skipping.")
                continue

            service = random.choice(services)
            turns = self.extract_dialogue(dialogue_json)
            base_conversation = self.generate_base_conversation(turns)

            # Select random emotions for user and assistant
            selected_user_emotions = random.sample(self.USER_EMOTION_LIST, k=random.randint(2, 4))
            selected_assistant_emotions = random.sample(self.ASSISTANT_EMOTION_LIST, k=random.randint(2, 4))

            # Select a random scenario category and generate a dynamic scenario
            selected_category = random.choice(self.SCENARIO_CATEGORIES)
            generated_scenario = self.generate_dynamic_scenario(selected_category, service)

            if not generated_scenario:
                self.logger.warning(f"Failed to generate scenario for dialogue {idx}. Skipping.")
                continue

            # Select random regions
            regions = random.sample(self.PREDEFINED_REGIONS, k=random.randint(1, 3))

            prompt = f"""
            Service: {service}
            Scenario: {generated_scenario}
            Regions: {', '.join(regions)}
            User Emotions: {', '.join(selected_user_emotions)}
            Assistant Emotions: {', '.join(selected_assistant_emotions)}

            Based on the above information and the following base conversation, generate a new dialogue between a USER and an ASSISTANT. The dialogue should incorporate the given scenario, emotions, and be relevant to the specified service and regions. Ensure each turn starts with either "USER:" or "ASSISTANT:".

            Base Conversation:
            {base_conversation}

            Generated Dialogue:
            """

            temperature = random.choice(temperature_options)
            top_p = random.choice(top_p_options)
            frequency_penalty = random.choice(frequency_penalty_options)
            presence_penalty = random.choice(presence_penalty_options)

            generated_dialogue = self.generate_dialogue(
                service, prompt, min_turns, max_turns,
                temperature, top_p, frequency_penalty, presence_penalty
            )

            if generated_dialogue:
                generated_turns = self.assign_selected_emotions(generated_dialogue, selected_user_emotions, selected_assistant_emotions)
                num_lines = len(generated_turns)

                # Generate a new unique dialogue ID
                new_dialogue_id = str(uuid.uuid4())
                while new_dialogue_id in self.existing_ids:
                    new_dialogue_id = str(uuid.uuid4())

                new_dialogue = {
                    'services': services,
                    'dialogue_id': new_dialogue_id,
                    'turns': generated_turns,
                    'num_lines': num_lines,
                    'user_emotions': selected_user_emotions,
                    'assistant_emotions': selected_assistant_emotions,
                    'scenario_category': selected_category,
                    'generated_scenario': generated_scenario,
                    'regions': regions
                }

                new_dialogues.append(new_dialogue)

                # Log the generated dialogue to Argilla
                self.log_to_argilla(new_dialogue)

                # Update hashes and embeddings
                dialogue_text = " ".join([turn['utterance'] for turn in generated_turns])
                dialogue_hash = hashlib.md5(dialogue_text.encode()).hexdigest()
                new_hashes.add(dialogue_hash)

                dialogue_embedding = self.embedding_model.encode([dialogue_text])
                new_embeddings.append(dialogue_embedding)

                self.logger.info(f"Generated dialogue {idx + 1}/{num_generations} with {num_lines} turns.")
            else:
                self.logger.warning(f"Failed to generate dialogue {idx + 1}/{num_generations}. Skipping.")

        # Save new dialogues
        if new_dialogues:
            try:
                with open(self.output_file, 'r+', encoding='utf-8') as f:
                    existing_dialogues = json.load(f)
                    existing_dialogues.extend(new_dialogues)
                    f.seek(0)
                    json.dump(existing_dialogues, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Saved {len(new_dialogues)} new dialogues to '{self.output_file}'.")
            except FileNotFoundError:
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(new_dialogues, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Created new file '{self.output_file}' with {len(new_dialogues)} dialogues.")
            except Exception as e:
                self.logger.error(f"Failed to save dialogues to '{self.output_file}': {e}")

        # Update and save hashes
        self.existing_hashes.update(new_hashes)
        try:
            with open(self.hash_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.existing_hashes), f)
            self.logger.info(f"Updated hash file '{self.hash_file}' with {len(new_hashes)} new hashes.")
        except Exception as e:
            self.logger.error(f"Failed to save hashes to '{self.hash_file}': {e}")

        # Update and save embeddings
        if new_embeddings:
            new_embeddings_array = np.vstack(new_embeddings)
            if self.existing_embeddings.size == 0:
                self.existing_embeddings = new_embeddings_array
            else:
                self.existing_embeddings = np.vstack([self.existing_embeddings, new_embeddings_array])
            self.save_embeddings(self.existing_embeddings)

        return new_dialogues

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dialogue Generation Script")
    parser.add_argument('--total_generations', type=int, default=10, help='Total number of dialogues to generate')
    parser.add_argument('--min_turns', type=int, default=5, help='Minimum number of turns in a dialogue')
    parser.add_argument('--max_turns', type=int, default=15, help='Maximum number of turns in a dialogue')
    parser.add_argument('--temperature', type=float, nargs='+', default=[0.7, 0.8, 0.9], help='Temperature options for generation')
    parser.add_argument('--top_p', type=float, nargs='+', default=[0.9, 0.95, 1.0], help='Top-p options for generation')
    parser.add_argument('--frequency_penalty', type=float, nargs='+', default=[0.0, 0.3, 0.5], help='Frequency penalty options for generation')
    parser.add_argument('--presence_penalty', type=float, nargs='+', default=[0.0, 0.3, 0.5], help='Presence penalty options for generation')
    parser.add_argument('--output_file', type=str, default='generated_dialogues.json', help='Output file for generated dialogues')
    parser.add_argument('--hash_file', type=str, default='dialogue_hashes.json', help='File to store dialogue hashes')
    parser.add_argument('--embedding_file', type=str, default='dialogue_embeddings.npy', help='File to store dialogue embeddings')
    parser.add_argument('--similarity_threshold', type=float, default=0.9, help='Similarity threshold for uniqueness check')
    return parser.parse_args()

def main():
    args = parse_arguments()

    config = {
        'output_file': args.output_file,
        'hash_file': args.hash_file,
        'embedding_file': args.embedding_file,
        'similarity_threshold': args.similarity_threshold,
        'total_generations': args.total_generations,
    }

    generator = DialogueGenerator(config)

    new_dialogues = generator.generate_unique_dialogues(
        num_generations=args.total_generations,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        temperature_options=args.temperature,
        top_p_options=args.top_p,
        frequency_penalty_options=args.frequency_penalty,
        presence_penalty_options=args.presence_penalty
    )

    print(f"Generated {len(new_dialogues)} new unique dialogues.")

if __name__ == "__main__":
    main()