import os
import json
import random
import logging
from time import sleep
from typing import List, Dict, Tuple, Optional, Literal
from datasets import load_from_disk, DatasetDict
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
from collections import defaultdict
import ast
from pydantic import BaseModel, Field, validator
from datetime import datetime

# Pydantic Models
class TurnBase(BaseModel):
    """Base model for dialogue turns"""
    turn_number: int
    utterance: str = Field(..., min_length=1)
    intent: str = Field(..., min_length=1)
    assistant_response: str = Field(..., min_length=1)
    emotion: Optional[str] = None

    class Config:
        extra = "allow"

class DialogueBase(BaseModel):
    """Base model for dialogues"""
    dialogue_id: str = Field(..., min_length=1)
    services: List[str] = Field(..., min_items=1)
    turns: List[TurnBase]
    num_lines: int = Field(..., gt=0)
    user_emotions: List[str] = Field(default_factory=list)
    assistant_emotions: List[str] = Field(default_factory=list)
    scenario_category: str
    generated_scenario: str
    time_slot: Tuple[int, int, str]
    regions: List[str] = Field(default_factory=list)
    resolution_status: Literal["Resolved", "Failed", "Escalated"]
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        extra = "allow"

    @validator('services')
    def validate_services(cls, v):
        if not v:
            raise ValueError('At least one service must be specified')
        return v

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

        # Initialize logging first
        self._initialize_logging()
        
        # Then load persona dataset
        self.persona_dataset = self.load_persona_dataset()
        self.cluster_dict = defaultdict(list)
        self.summary_dict = defaultdict(list)
        self.populate_persona_dicts()

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
    def _initialize_logging(self):
        """Initialize logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler("dialogue_generation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize spaCy
        self._initialize_spacy()
        
        # Initialize OpenAI
        self._initialize_openai()
        
        # Initialize SentenceTransformer
        self._initialize_sentence_transformer()

        # Initialize constants
        self._initialize_constants()

        # Load dataset and existing data
        self.dataset = self.load_dataset()
        self.existing_ids = self.load_existing_dialogues()
        self.existing_hashes = self.load_existing_hashes()
        self.existing_embeddings = self.load_existing_embeddings()

    def _initialize_spacy(self):
        """Initialize spaCy model"""
        try:
            self.logger.info("Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.info("Downloading spaCy model 'en_core_web_sm'...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

    def _initialize_openai(self):
        """Initialize OpenAI client"""
        load_dotenv('.env.local')
        openai_api_key = os.getenv('OPENAI_KEY')
        if not openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=openai_api_key)

    def _initialize_sentence_transformer(self):
        """Initialize SentenceTransformer model"""
        try:
            self.logger.info("Loading SentenceTransformer model...")
            model_path = './models/sentence_transformer'
            self.embedding_model = SentenceTransformer(model_path)
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise e
    def _initialize_constants(self):
        """Initialize constant values and prompts"""
        self.SERVICES = ["train", "bus", "taxi", "hotel", "restaurant", "attraction"]
        self.TIME_SLOTS = [
            (0, 6, "Early Morning"),
            (6, 12, "Morning"),
            (12, 17, "Afternoon"),
            (17, 21, "Evening"),
            (21, 24, "Late Night")
        ]
        self.REGIONS = ["Tokyo", "Osaka", "Kyoto", "Nagoya", "Fukuoka"]
        self.EMOTIONS = ["Happy", "Frustrated", "Neutral", "Confused", "Satisfied"]
        self.SCENARIO_CATEGORIES = ["Booking", "Cancellation", "Information", "Complaint", "Emergency"]

    def load_persona_dataset(self) -> Optional[DatasetDict]:
        """Load the persona dataset"""
        try:
            dataset_path = "./data/persona_dataset"
            if os.path.exists(dataset_path):
                return load_from_disk(dataset_path)
            self.logger.warning("Persona dataset not found")
            return None
        except Exception as e:
            self.logger.error(f"Error loading persona dataset: {e}")
            return None

    def populate_persona_dicts(self):
        """Populate persona dictionaries from dataset"""
        if not self.persona_dataset:
            return

        try:
            for item in self.persona_dataset['train']:
                cluster = item.get('cluster', 'default')
                summary = item.get('summary', '')
                if cluster and summary:
                    self.cluster_dict[cluster].append(item)
                    self.summary_dict[cluster].append(summary)
        except Exception as e:
            self.logger.error(f"Error populating persona dicts: {e}")

    def load_dataset(self) -> Optional[DatasetDict]:
        """Load the main dialogue dataset"""
        try:
            return load_from_disk(self.dataset_name)
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return None

    def load_existing_dialogues(self) -> set:
        """Load existing dialogue IDs"""
        try:
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {d.get('dialogue_id') for d in data if 'dialogue_id' in d}
            return set()
        except Exception as e:
            self.logger.error(f"Error loading existing dialogues: {e}")
            return set()

    def load_existing_hashes(self) -> set:
        """Load existing dialogue hashes"""
        try:
            if os.path.exists(self.hash_file):
                with open(self.hash_file, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            return set()
        except Exception as e:
            self.logger.error(f"Error loading existing hashes: {e}")
            return set()

    def load_existing_embeddings(self) -> Optional[np.ndarray]:
        """Load existing dialogue embeddings"""
        try:
            if os.path.exists(self.embedding_file):
                return np.load(self.embedding_file)
            return None
        except Exception as e:
            self.logger.error(f"Error loading existing embeddings: {e}")
            return None

    def generate_dialogue(self, service: str, prompt: str, min_turns: int, max_turns: int,
                         temperature: float = 0.9, top_p: float = 0.95,
                         frequency_penalty: float = 0.5, presence_penalty: float = 0.5,
                         max_retries: int = 3) -> Optional[Dict]:
        """
        Generate a structured dialogue using OpenAI's function calling
        """
        function_schema = {
            "name": "create_dialogue",
            "description": "Create a structured dialogue between user and assistant",
            "parameters": {
                "type": "object",
                "properties": {
                    "turns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "utterance": {"type": "string"},
                                "intent": {"type": "string"},
                                "assistant_response": {"type": "string"},
                                "emotion": {"type": "string"}
                            },
                            "required": ["utterance", "intent", "assistant_response"]
                        }
                    }
                },
                "required": ["turns"]
            }
        }

        system_prompt = f"""You are an AI assistant helping with {service}-related inquiries.
        Generate a natural dialogue with {min_turns}-{max_turns} turns.
        Each turn should include user utterance, intent, and your response."""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    functions=[function_schema],
                    function_call={"name": "create_dialogue"},
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )

                if response.choices[0].message.function_call:
                    function_response = json.loads(
                        response.choices[0].message.function_call.arguments
                    )
                    
                    # Validate turns using Pydantic
                    validated_turns = []
                    for idx, turn in enumerate(function_response.get('turns', []), 1):
                        try:
                            validated_turn = TurnBase(
                                turn_number=idx,
                                **turn
                            )
                            validated_turns.append(validated_turn.dict())
                        except Exception as e:
                            self.logger.warning(f"Invalid turn structure: {e}")
                            continue
                    
                    if validated_turns:
                        return {"turns": validated_turns}

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    sleep(2 ** attempt)

        self.logger.error("Failed to generate valid dialogue after all retries")
        return None
    def validate_dialogue_data(self, dialogue_data: Dict) -> Optional[DialogueBase]:
        """Validates dialogue data using Pydantic model"""
        try:
            return DialogueBase(**dialogue_data)
        except Exception as e:
            self.logger.error(f"Dialogue validation failed: {e}")
            return None

    def compute_dialogue_hash(self, dialogue: Dict) -> str:
        """Compute a hash for the dialogue content"""
        try:
            # Create a string representation of the dialogue turns
            dialogue_str = json.dumps([
                {
                    'utterance': turn['utterance'],
                    'assistant_response': turn['assistant_response']
                }
                for turn in dialogue['turns']
            ], sort_keys=True)
            return hashlib.sha256(dialogue_str.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error computing dialogue hash: {e}")
            return ""

    def compute_dialogue_embedding(self, dialogue: Dict) -> Optional[np.ndarray]:
        """Compute embedding for dialogue content"""
        try:
            # Concatenate all turns into a single string
            dialogue_text = " ".join([
                f"{turn['utterance']} {turn['assistant_response']}"
                for turn in dialogue['turns']
            ])
            return self.embedding_model.encode([dialogue_text])[0]
        except Exception as e:
            self.logger.error(f"Error computing dialogue embedding: {e}")
            return None

    def is_similar_to_existing(self, dialogue: Dict, threshold: float = None) -> bool:
        """Check if dialogue is similar to existing ones"""
        if threshold is None:
            threshold = self.similarity_threshold

        try:
            # Compute hash and check for exact matches
            dialogue_hash = self.compute_dialogue_hash(dialogue)
            if dialogue_hash in self.existing_hashes:
                return True

            # Compute embedding and check for semantic similarity
            new_embedding = self.compute_dialogue_embedding(dialogue)
            if new_embedding is None or self.existing_embeddings is None:
                return False

            similarities = cosine_similarity([new_embedding], self.existing_embeddings)[0]
            return any(sim > threshold for sim in similarities)
        except Exception as e:
            self.logger.error(f"Error checking dialogue similarity: {e}")
            return False

    def save_embeddings(self, embeddings: Optional[np.ndarray]):
        """Save dialogue embeddings to file"""
        try:
            if embeddings is not None:
                np.save(self.embedding_file, embeddings)
                self.logger.info(f"Saved embeddings to {self.embedding_file}")
        except Exception as e:
            self.logger.error(f"Failed to save embeddings: {e}")

    def save_new_dialogues(self, new_dialogues: List[Dict]):
        """Save new dialogues with validation"""
        try:
            # Load existing dialogues
            all_dialogues = []
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    for d in existing_data:
                        validated = self.validate_dialogue_data(d)
                        if validated:
                            all_dialogues.append(validated.dict())
                        else:
                            all_dialogues.append(d)  # Keep original if validation fails

            # Process new dialogues
            new_embeddings = []
            for dialogue in new_dialogues:
                try:
                    # Validate dialogue
                    validated_dialogue = self.validate_dialogue_data(dialogue)
                    if not validated_dialogue:
                        continue

                    # Compute hash and embedding
                    dialogue_hash = self.compute_dialogue_hash(dialogue)
                    dialogue_embedding = self.compute_dialogue_embedding(dialogue)

                    if dialogue_hash and dialogue_embedding is not None:
                        self.existing_hashes.add(dialogue_hash)
                        new_embeddings.append(dialogue_embedding)
                        all_dialogues.append(validated_dialogue.dict())
                except Exception as e:
                    self.logger.error(f"Error processing dialogue: {e}")
                    continue

            # Update embeddings
            if new_embeddings:
                if self.existing_embeddings is None:
                    self.existing_embeddings = np.array(new_embeddings)
                else:
                    self.existing_embeddings = np.vstack([self.existing_embeddings, new_embeddings])

            # Save everything
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(all_dialogues, f, indent=4, ensure_ascii=False, default=str)

            self.save_embeddings(self.existing_embeddings)
            
            with open(self.hash_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.existing_hashes), f, indent=4)

            self.logger.info(f"Saved {len(new_dialogues)} new dialogues")

        except Exception as e:
            self.logger.error(f"Failed to save dialogues: {e}")

    def generate_unique_dialogues(self, num_generations: int) -> List[DialogueBase]:
        """Generate unique dialogues with varying parameters"""
        new_dialogues = []
        pbar = tqdm(total=num_generations, desc="Generating dialogues")

        while len(new_dialogues) < num_generations:
            try:
                # Select random parameters
                service = random.choice(self.SERVICES)
                min_turns = random.randint(*self.min_turns_range)
                max_turns = random.randint(min_turns, max(min_turns + 2, self.max_turns_range[1]))
                temperature = random.uniform(*self.temperature_range)
                top_p = random.uniform(*self.top_p_range)
                frequency_penalty = random.uniform(*self.frequency_penalty_range)
                presence_penalty = random.uniform(*self.presence_penalty_range)

                # Generate scenario and prompt
                scenario_category = random.choice(self.SCENARIO_CATEGORIES)
                time_slot = random.choice(self.TIME_SLOTS)
                regions = random.sample(self.REGIONS, k=random.randint(1, 3))
                
                prompt = self.generate_scenario_prompt(
                    service, scenario_category, time_slot, regions
                )

                # Generate dialogue
                generated = self.generate_dialogue(
                    service, prompt, min_turns, max_turns,
                    temperature, top_p, frequency_penalty, presence_penalty
                )

                if not generated:
                    continue

                # Create dialogue object
                dialogue_data = {
                    "dialogue_id": str(uuid.uuid4()),
                    "services": [service],
                    "turns": generated["turns"],
                    "num_lines": len(generated["turns"]),
                    "scenario_category": scenario_category,
                    "generated_scenario": prompt,
                    "time_slot": time_slot,
                    "regions": regions,
                    "resolution_status": random.choice(["Resolved", "Failed", "Escalated"])
                }

                # Validate and check similarity
                validated_dialogue = self.validate_dialogue_data(dialogue_data)
                if validated_dialogue and not self.is_similar_to_existing(dialogue_data):
                    new_dialogues.append(validated_dialogue)
                    pbar.update(1)

            except Exception as e:
                self.logger.error(f"Error generating dialogue: {e}")
                continue

        pbar.close()
        return new_dialogues
    def generate_scenario_prompt(self, service: str, category: str, time_slot: Tuple[int, int, str], regions: List[str]) -> str:
        """Generate a scenario prompt for dialogue generation"""
        try:
            time_desc = time_slot[2]
            region_str = ", ".join(regions)
            
            scenario_templates = {
                "Booking": [
                    f"Generate a dialogue where a user is trying to book a {service} service in {region_str} during {time_desc}.",
                    f"Create a conversation about making a {service} reservation in {region_str} for {time_desc}."
                ],
                "Cancellation": [
                    f"Generate a dialogue about cancelling a {service} booking in {region_str} scheduled for {time_desc}.",
                    f"Create a conversation where a user needs to cancel their {service} reservation in {region_str}."
                ],
                "Information": [
                    f"Generate a dialogue where a user is asking for information about {service} services in {region_str} during {time_desc}.",
                    f"Create a conversation about {service} service details and availability in {region_str}."
                ],
                "Complaint": [
                    f"Generate a dialogue where a user is complaining about a {service} service issue in {region_str}.",
                    f"Create a conversation about a problem with {service} service in {region_str} during {time_desc}."
                ],
                "Emergency": [
                    f"Generate a dialogue about an urgent {service}-related situation in {region_str} during {time_desc}.",
                    f"Create a conversation about an emergency involving {service} service in {region_str}."
                ]
            }

            templates = scenario_templates.get(category, scenario_templates["Information"])
            base_prompt = random.choice(templates)

            # Add specific instructions for the AI
            additional_instructions = [
                "Include realistic details and natural conversation flow.",
                "Make sure to handle any user concerns professionally.",
                "Include appropriate emotional responses where relevant.",
                f"Consider the time of day ({time_desc}) in the interaction."
            ]

            full_prompt = f"{base_prompt}\n\nInstructions:\n" + "\n".join(additional_instructions)
            return full_prompt

        except Exception as e:
            self.logger.error(f"Error generating scenario prompt: {e}")
            return f"Generate a basic {service} service dialogue."

    def process_generated_dialogue(self, generated_dialogue: Dict) -> List[TurnBase]:
        """Process and validate generated dialogue turns"""
        if not generated_dialogue or 'turns' not in generated_dialogue:
            return []

        processed_turns = []
        for idx, turn_data in enumerate(generated_dialogue['turns'], 1):
            try:
                turn = TurnBase(
                    turn_number=idx,
                    utterance=turn_data['utterance'],
                    intent=turn_data['intent'],
                    assistant_response=turn_data['assistant_response'],
                    emotion=turn_data.get('emotion', 'Neutral')
                )
                processed_turns.append(turn)
            except Exception as e:
                self.logger.error(f"Error processing turn {idx}: {e}")
                continue
        
        return processed_turns

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Remove special characters but keep basic punctuation
            text = re.sub(r'[^\w\s.,!?-]', '', text)
            
            # Normalize quotes
            text = re.sub(r'["""\']', '"', text)  # Fixed quote characters in regex
            
            return text
        except Exception as e:
            self.logger.error(f"Error cleaning text: {e}")
            return text

    def run_generation(self, num_dialogues: int = None):
        """Main method to run dialogue generation"""
        try:
            if num_dialogues is None:
                num_dialogues = self.total_generations

            self.logger.info(f"Starting generation of {num_dialogues} dialogues...")
            
            # Generate dialogues
            generated_dialogues = self.generate_unique_dialogues(num_dialogues)
            
            if generated_dialogues:
                # Save the new dialogues
                self.save_new_dialogues([d.dict() for d in generated_dialogues])
                self.logger.info(f"Successfully generated and saved {len(generated_dialogues)} dialogues")
            else:
                self.logger.warning("No dialogues were generated")

        except Exception as e:
            self.logger.error(f"Error in run_generation: {e}")
            raise

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate dialogue datasets')
    parser.add_argument('--config', type=str, default='config.json',
                      help='Path to configuration file')
    parser.add_argument('--num_dialogues', type=int, default=None,
                      help='Number of dialogues to generate')
    parser.add_argument('--output', type=str, default=None,
                      help='Output file path')
    
    args = parser.parse_args()

    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = json.load(f)

        # Override config with command line arguments
        if args.output:
            config['output_file'] = args.output

        # Initialize generator
        generator = DialogueGenerator(config)
        
        # Run generation
        generator.run_generation(args.num_dialogues)

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()