# dialogue_generation.py

import os
import json
import random
import logging
from time import sleep
from typing import List, Dict, Tuple
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


class DialogueGenerator:
    def __init__(self, config: Dict):
        self.output_file = config.get('output_file', 'generated_dialogues.json')
        self.hash_file = config.get('hash_file', 'dialogue_hashes.json')
        self.embedding_file = config.get('embedding_file', 'dialogue_embeddings.npy')
        self.similarity_threshold = config.get('similarity_threshold', 0.9)
        self.dataset_name = config.get('dataset_name', 'pfb30/multi_woz_v22')
        self.min_turns_range = config.get('min_turns_range', (5, 10))
        self.max_turns_range = config.get('max_turns_range', (7, 20))
        self.temperature_range = config.get('temperature_range', (0.7, 1.0))
        self.top_p_range = config.get('top_p_range', (0.8, 1.0))
        self.frequency_penalty_range = config.get('frequency_penalty_range', (0.0, 0.7))
        self.presence_penalty_range = config.get('presence_penalty_range', (0.0, 0.7))
        self.total_generations = config.get('total_generations', 10)
        self.persona_dataset = self.load_persona_dataset()
        self.cluster_dict = defaultdict(list)
        self.summary_dict = defaultdict(list)
        self.populate_persona_dicts()



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
        self.travel_time_slots = [
            (5, 9, "Early Morning"),    # 5 AM - 8:55 AM
            (9, 12, "Late Morning"),    # 9 AM - 11:55 AM
            (12, 17, "Afternoon"),      # 12 PM - 4:55 PM
            (17, 21, "Evening"),        # 5 PM - 8:55 PM
            (21, 23, "Late Night"),     # 9 PM - 11:55 PM
            (23, 5, "Overnight")         # 12 AM - 4:55 AM
        ]
        self.RESOLUTION_STATUSES = {
            "Resolved": 0.4,
            "Failed": 0.3,
            "Escalated": 0.3
        }

        # Define emotion lists
        self.USER_EMOTION_LIST = [
            "Frustrated", "Angry", "Confused", "Worried", "Disappointed",
            "Happy", "Anxious", "Impatient", "Skeptical", "Desperate",
            "Overwhelmed", "Hopeful", "Satisfied", "Stressed", "Suspicious",
            "Tired", "Excited", "Indifferent", "Grateful", "Demanding"
        ]

        self.ASSISTANT_EMOTION_LIST = [
            "Professional", "Informative", "Reassuring", "Diplomatic", "Patient",
            "Efficient", "Accommodating", "Solution-focused", "Methodical", "Proactive",
            "Analytical", "Composed", "Detail-oriented", "Responsive", "Thorough",
            "Systematic", "Precise", "Objective", "Resourceful", "Knowledgeable"
        ]
        # Scenario categories
        self.SCENARIO_CATEGORIES = {
            # General categories (applicable across services)
            "general": [
            "account_management",      # Managing user accounts, passwords, etc.
            "cancellation_general",    # General cancellation requests not tied to a specific service
            "complaint",               # General complaints about services or experiences
            "refund_request_general",  # Refunds not specific to a service
            "payment_issues",          # Issues related to payments across services
            "general_inquiry",         # General questions not specific to any service
            "feedback",                # Providing feedback about services or experiences
            "technical_support",       # Technical assistance for app or website issues
            "lost_and_found_general"   # Reporting lost items not tied to a specific service
            ],
                        # Restaurant-specific
            "restaurant": [
            "dining_reservation",
            "dietary_requirements",
            "table_modification",
            "special_occasion",
            "menu_inquiry",            # Inquiring about menu items or specials
            "order_status",            # Checking the status of an order
            "reservation_cancellation" # Specific cancellation for reservations
            ],
            
            # Hotel-specific
            "hotel": [
                "room_reservation",
                "check_in_out",
                "amenity_inquiry",
                "room_service",
                "booking_modification",    # Modifying existing bookings
                "housekeeping_request",    # Requests related to room cleaning and maintenance
                "reservation_cancellation" # Specific cancellation for hotel bookings
            ],
            # Train-specific
            "train": [
            "journey_planning",
            "schedule_inquiry",
            "ticket_booking",
            "platform_information",
            "ticket_change",           # Changing ticket details
            "ticket_cancellation",     # Specific cancellation for train tickets
            "seat_selection"           # Selecting or changing seats
            ],
            
            # Attraction-specific
            "attraction": [
                "ticket_availability",
                "opening_hours",
                "guided_tour",
                "venue_information",
                "group_booking",           # Booking for groups
                "ticket_cancellation",     # Specific cancellation for attraction tickets
                "accessibility_inquiry"    # Inquiries about accessibility features
            ],
            # Taxi-specific
            "taxi": [
                "ride_booking",
                "pickup_location",
                "fare_inquiry",
                "driver_tracking",
                "ride_cancellation",       # Specific cancellation for taxi rides
                "ride_feedback",           # Providing feedback on taxi rides
                "service_type_inquiry"    # Inquiring about different types of taxi services
            ],
            # Hospital-specific
            "hospital": [
                "appointment_booking",
                "department_inquiry",
                "medical_information",
                "emergency_services",
                "appointment_cancellation",# Specific cancellation for appointments
                "insurance_inquiry",       # Questions about insurance coverage
                "medical_record_request"   # Requesting medical records
            ],
            

            # Bus-specific
            "bus": [
                "route_information",
                "schedule_inquiry",
                "ticket_booking",
                "stop_location",
                "ticket_change",           # Changing bus ticket details
                "ticket_cancellation",     # Specific cancellation for bus tickets
                "seat_selection"           # Selecting or changing bus seats
            ],
            "flight": [
                "flight_booking",
                "cancellation_flight",     # Specific cancellation for flights
                "ticket_change",
                "baggage_inquiry",
                "check_in",
                "seat_selection",
                "flight_status",
                "upgrade_request",
                "refund_request_flight",   # Refund requests specific to flights
                "lounge_access",
                "boarding_pass_issue",     # Issues related to boarding passes
                "special_meals",           # Requests for special meals on flights
                "pet_transportation"      # Inquiries about transporting pets
            ]

        }


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
    def get_categories_for_service(self, service: str) -> List[str]:
        """
        Returns a list of categories applicable to the given service.
        Combines service-specific categories with general categories.
        """
        # Get general categories that apply to all services
        categories = self.SCENARIO_CATEGORIES["general"].copy()
        
        # Add service-specific categories if they exist
        if service.lower() in self.SCENARIO_CATEGORIES:
            categories.extend(self.SCENARIO_CATEGORIES[service.lower()])
        else:
            self.logger.warning(f"Unknown service '{service}', using only general categories")
            
        return categories
    def load_persona_dataset(self):
        dataset_path = './local_datasets/FinePersonas-v0.1-clustering-100k'
        dataset = load_from_disk(dataset_path)
        return dataset[list(dataset.keys())[0]]  
    def safe_eval(self, s):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []

    def get_summary_labels(self, label):
        labels = self.safe_eval(label)
        if isinstance(labels, list):
            return tuple(sorted(labels))
        elif isinstance(labels, str):
            return (label,)
        return tuple()

    def populate_persona_dicts(self):
        for i, (cluster, summary) in enumerate(zip(self.persona_dataset['cluster_label'], self.persona_dataset['summary_label'])):
            self.cluster_dict[cluster].append(i)
            summary_labels = self.get_summary_labels(summary)
            self.summary_dict[summary_labels].append(i)

    def select_random_persona(self):
        if random.choice([True, False]):
            # Cluster-based selection
            cluster = random.choice(list(self.cluster_dict.keys()))
            index = random.choice(self.cluster_dict[cluster])
        else:
            # Summary-based selection
            summary = random.choice(list(self.summary_dict.keys()))
            index = random.choice(self.summary_dict[summary])

        return self.persona_dataset[index]['persona']
    def generate_random_time(self, time_slot: Tuple[int, int, str]) -> str:
        """
        Generates a random time within a given time slot in 5-minute intervals.
        Handles overnight slots correctly and ensures uniform distribution.
        
        Args:
            time_slot: Tuple of (start_hour, end_hour, slot_name)
        
        Returns:
            String representation of the random time (HH:MM format)
        """
        try:
            start_hour, end_hour = time_slot[0], time_slot[1]
            
            # Validate input hours
            if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
                self.logger.error(f"Invalid hours in time slot: {time_slot}. Using default time.")
                return "00:00"
            
            # Handle overnight slots (e.g., 23:00 to 05:00)
            if end_hour < start_hour:
                if random.random() < 0.5:
                    # Night portion (start_hour to midnight)
                    start_minutes = start_hour * 60
                    end_minutes = 24 * 60  # midnight
                else:
                    # Morning portion (midnight to end_hour)
                    start_minutes = 0
                    end_minutes = end_hour * 60
            else:
                # Normal daytime slot
                start_minutes = start_hour * 60
                end_minutes = end_hour * 60

            # Generate random minutes within the range
            total_minutes = random.randint(start_minutes, end_minutes - 1)
            
            # Convert to hours and minutes
            hours = (total_minutes // 60) % 24
            mins = total_minutes % 60
            
            # Round to nearest 5 minutes
            mins = round(mins / 5) * 5
            if mins == 60:
                hours = (hours + 1) % 24
                mins = 0
                
            generated_time = f"{hours:02d}:{mins:02d}"
            self.logger.debug(f"Generated time {generated_time} for slot {time_slot[2]}")
            return generated_time

        except Exception as e:
            self.logger.error(f"Error generating random time for slot {time_slot}: {str(e)}")
            return "00:00"  # Return default time in case of any error
    def load_dataset(self) -> any:
        """
        Loads the dataset from Hugging Face once during initialization.
        """
        try:
            self.logger.info(f"Loading dataset from local path: ./local_datasets/multi_woz_v22")
            dataset = DatasetDict.load_from_disk("./local_datasets/multi_woz_v22")
            self.logger.info("Dataset loaded successfully.")
            self.logger.info(f"Number of dialogues in 'train' split: {len(dataset['train'])}")
            return dataset['train']
        except Exception as e:
            self.logger.error(f"Failed to load dataset from ./local_datasets/multi_woz_v22: {e}")
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
        
        # Log a preview of the conversation (first 3 turns and last 3 turns)
        # preview_lines = conversation_lines[:3] + ['...'] + conversation_lines[-3:] if len(conversation_lines) > 6 else conversation_lines
        # preview = "\n".join(preview_lines)
        # self.logger.info(f"Generated base conversation preview:\n{preview}")
        
        # self.logger.info(f"Total turns in conversation: {len(turns)}")
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
                # self.logger.info(f"Loaded {len(hashes)} existing dialogue hashes from '{self.hash_file}'.")
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

    # def assign_selected_emotions(self, turns: List[Dict], user_emotions: List[str], assistant_emotions: List[str]) -> List[Dict]:
    #     """
    #     Assigns emotions to each turn in the dialogue based on the speaker.
    #     """
    #     return turns
    def generate_dynamic_scenario(self, category: str, listservice: list, region: str) -> str:
        """
        Generates a specific scenario based on the given category and service using OpenAI's API.
        Ensures the scenario is relevant to the service and limited to 2-3 lines.
        """
        try:
            valid_categories = self.get_categories_for_service(listservice[0])
        
        # If the provided category isn't valid for this service, select a random valid one
            if category not in valid_categories:
                category = random.choice(valid_categories)
                self.logger.info(f"Selected new category '{category}' for service '{listservice}'")


            user_persona = self.select_random_persona()
            selected_slot = random.choice(self.travel_time_slots)
            specific_time = self.generate_random_time(selected_slot)

            self.logger.info(f"Selected persona: {user_persona}")
            self.logger.info(f"Selected time: {specific_time} ({selected_slot[2]})")
            self.logger.info(f"Selected region: {region}")

            system_prompt = (
                "You are a creative assistant tasked with generating specific scenarios relevant to the given category and service. "
                "Each scenario should be detailed, pertinent to the provided service, and confined to 2-3 lines. "
                f"Provide one unique scenario for the category and service from the perspective of the persona: {user_persona}. "
                f"The scenario should occur specifically at {specific_time} during the {selected_slot[2]} period"
                f"and be set in {region}. Include local context and regional specifics where appropriate."

            )

            user_prompt = (
                f"Generate a concise (2-3 lines) scenario for the category: '{category}' and service: "
                f"'{', '.join(listservice)}'. The scenario should occur during the {specific_time} time slot set in {region} "
                "Please ensure that the service, location, and time slot are always kept in mind."
            )

            response = self.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,
                temperature=0.7,
                top_p=0.9,
                n=1,
                stop=None,
            )
            scenario = response.choices[0].message.content.strip()
            self.logger.info(f"Generated scenario for category '{category}' and service '{listservice}': {scenario}")
            return scenario, selected_slot

        except OpenAIError as e:
            self.logger.error(f"OpenAI API error during scenario generation for category '{category}' and service '{listservice}': {e}")
            return None, None
        except Exception as e:
            self.logger.error(f"Unexpected error during scenario generation for category '{category}' and service '{listservice}': {e}")
            return None, None


    def generate_dialogue(self, listservice: list, prompt: str, min_turns: int, max_turns: int, 
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
                f"You are an expert dialogue generator for the '{', '.join(listservice)}' services. "
                f"{scenario_str} "
                f"{regions_str} "
                f"Create a high-quality, coherent, and emotionally rich dialogue between a user and an assistant. "
                f"The user should express the following emotions: {user_emotions_str}. "
                f"The assistant should express the following emotions: {assistant_emotions_str}. "
                f"The dialogue should have between {min_turns} and {max_turns} turns (a turn is one user message and one assistant response). "
                f"The dialogue should not be the same as any existing dialogues and should be better and more engaging. "
                f"Make sure that these dialogues are representative of conversations between a text/voice interface hence the assistant might not be present. "
                f"These dialogues should encompass and showcase different scenarios and outcomes in a strictly service setting. "
                f"Encourage diverse linguistic expressions and response styles to mimic real human interactions.\n\n"
                f"For each turn, provide an intent classification for the user's message. "
                f"Use the following format for each turn:\n"
                f"<User> user's message </User>\n"
                f"<Intent> intent classification </Intent>\n"
                f"<Assistant> assistant's response </Assistant>\n"
            )

            # self.logger.info(f"System prompt: {system_prompt}")

            # self.logger.info(f" prompt: {prompt}")
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model='gpt-4o-mini',  # Corrected model name
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
                        # Check if the dialogue contains the expected XML-like tags
                        if re.search(r'<User>.*?</User>.*?<Intent>.*?</Intent>.*?<Assistant>.*?</Assistant>', gen_dialogue, re.DOTALL):
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
        current_turn = {}
        
        # Use regex to split the dialogue into turns
        turn_pattern = re.compile(r'<User>(.*?)</User>\s*<Intent>(.*?)</Intent>\s*<Assistant>(.*?)</Assistant>', re.DOTALL)
        turns = turn_pattern.findall(generated_dialogue)
        
        for turn_number, (user_message, intent, assistant_response) in enumerate(turns, start=1):
            current_turn = {
                'turn_number': turn_number,
                'utterance': user_message.strip(),
                'intent': intent.strip(),
                'assistant_response': assistant_response.strip()
            }
            generated_turns.append(current_turn)

        # Ensure each turn has both 'utterance' and 'assistant_response'
        for turn in generated_turns:
            if 'utterance' not in turn or not turn['utterance']:
                turn['utterance'] = "N/A"
                self.logger.warning("Added missing utterance in turn")
            if 'assistant_response' not in turn or not turn['assistant_response']:
                turn['assistant_response'] = "N/A"
                self.logger.warning("Added missing assistant_response in turn")
            if 'intent' not in turn or not turn['intent']:
                turn['intent'] = "N/A"
                self.logger.warning("Added missing intent in turn")

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
        service_to_indices = defaultdict(list)
        for idx in range(len(self.dataset)):
            services = self.dataset[idx].get('services', [])
            if services:  # Only consider examples with defined services
                primary_service = services[0]  # Take the primary service
                service_to_indices[primary_service].append(idx)

        # NEW CODE: Calculate target number of dialogues per service
        total_services = len(service_to_indices)
        dialogues_per_service = num_generations // total_services
        remainder = num_generations % total_services

        # NEW CODE: Distribute dialogues across services
        selected_indices = []
        for service, indices in service_to_indices.items():
            # Add extra dialogue from remainder if available
            service_quota = dialogues_per_service + (1 if remainder > 0 else 0)
            remainder = max(0, remainder - 1)
            
            # Sample indices for this service (with replacement if necessary)
            service_indices = random.choices(indices, k=service_quota)
            selected_indices.extend(service_indices)
            
            self.logger.info(f"Allocated {service_quota} dialogues for service '{service}'")

        # NEW CODE: Shuffle the selected indices to mix services
        random.shuffle(selected_indices)
        CORE_SERVICES = ['hotel', 'restaurant', 'train', 'attraction', 'taxi', 'bus', 'hospital','flight']
        LOGICAL_COMBINATIONS = {
        'double': [
            # Hospital-related combinations
            ['hospital', 'taxi'],
            ['hospital', 'hotel'],
            # **Added flight-related combinations**
            ['flight', 'taxi'],
            ['flight', 'hotel'],
            ['flight', 'train'],
            ['flight', 'bus'],
            ['flight', 'restaurant'],
            # Travel & Accommodation
            ['hotel', 'taxi'],
            ['hotel', 'train'],
            ['hotel', 'bus'],
            # Dining & Entertainment
            ['restaurant', 'taxi'],
            ['restaurant', 'attraction'],
            ['attraction', 'taxi'],
            # Transport Connections
            ['train', 'taxi'],
            ['bus', 'taxi'],
            ['train', 'bus'],
            # Common Pairings
            ['hotel', 'restaurant'],
        ],
        'triple': [
            # Hospital-related combinations
            ['hospital', 'taxi', 'hotel'],
            ['hospital', 'hotel', 'restaurant'],
            # Travel & Stay Combinations
            ['hotel', 'restaurant', 'taxi'],
            ['hotel', 'train', 'taxi'],
            ['hotel', 'bus', 'taxi'],
            # Tourism Combinations
            ['attraction', 'restaurant', 'taxi'],
            ['attraction', 'hotel', 'taxi'],
            ['attraction', 'train', 'taxi'],
            # Extended Travel Plans
            ['train', 'hotel', 'restaurant'],
            ['bus', 'hotel', 'restaurant'],
            ['train', 'restaurant', 'taxi'],
            # **Added flight-related triples**
            ['flight', 'hotel', 'taxi'],
            ['flight', 'train', 'taxi'],
            ['flight', 'bus', 'taxi'],
            ['flight', 'restaurant', 'taxi'],

        ],
        'quadruple': [
            # Hospital-related combinations
            ['hospital', 'hotel', 'restaurant', 'taxi'],
            # Full Tourism Package
            ['hotel', 'restaurant', 'attraction', 'taxi'],
            ['train', 'hotel', 'restaurant', 'taxi'],
            ['bus', 'hotel', 'restaurant', 'taxi'],
            # Extended Tourism
            ['train', 'hotel', 'attraction', 'taxi'],
            ['bus', 'hotel', 'attraction', 'taxi'],
            ['flight', 'hotel', 'restaurant', 'taxi'],
            ['flight', 'train', 'hotel', 'taxi'],
            ['flight', 'bus', 'hotel', 'taxi'],

        ]}
        SERVICE_WEIGHTS = {
        'single': 0.30,    # 30% single service
        'double': 0.35,    # 35% two services
        'triple': 0.25,    # 25% three services
        'quadruple': 0.10  # 10% four services
    }

    # Track usage of combinations to ensure balanced distribution
        combination_usage = {
            'single': {service: 0 for service in CORE_SERVICES},
            'double': {str(tuple(sorted(combo))): 0 for combo in LOGICAL_COMBINATIONS['double']},
            'triple': {str(tuple(sorted(combo))): 0 for combo in LOGICAL_COMBINATIONS['triple']},
            'quadruple': {str(tuple(sorted(combo))): 0 for combo in LOGICAL_COMBINATIONS['quadruple']}
        }
        new_dialogues = []
        batch_size = 10000  # Define a batch size suitable for your system

        for idx, dataset_index in enumerate(tqdm(selected_indices, desc="Generating dialogues")):
            example = self.dataset[dataset_index]
            services = example.get('services', [])
            original_dialogue_id = example.get('dialogue_id', f"dialogue_{dataset_index}")
                    # REPLACE THIS SECTION - Start
        # Determine number of services based on weights
            service_count = random.choices(
                ['single', 'double', 'triple', 'quadruple'],
                weights=[SERVICE_WEIGHTS['single'], 
                        SERVICE_WEIGHTS['double'], 
                        SERVICE_WEIGHTS['triple'],
                        SERVICE_WEIGHTS['quadruple']]
            )[0]

            if service_count == 'single':
                # Find least used single service (all services have equal chance)
                least_used_service = min(
                    CORE_SERVICES,
                    key=lambda x: combination_usage['single'][x]
                )
                services_to_use = [least_used_service]
                combination_usage['single'][least_used_service] += 1
            else:
                # Find least used logical combination
                available_combinations = LOGICAL_COMBINATIONS[service_count]
                usage_counts = {
                    str(tuple(sorted(combo))): combination_usage[service_count][str(tuple(sorted(combo)))] 
                    for combo in available_combinations
                }
                least_used_count = min(usage_counts.values())
                least_used_combinations = [
                    combo for combo in available_combinations 
                    if combination_usage[service_count][str(tuple(sorted(combo)))] == least_used_count
                ]
                services_to_use = random.choice(least_used_combinations)
                combination_usage[service_count][str(tuple(sorted(services_to_use)))] += 1

            random.shuffle(services_to_use)

            primary_service = services_to_use[0]
            self.logger.info(f"Selected services for dialogue_id '{original_dialogue_id}': {services_to_use}")
            if idx % 100 == 0:
                self.logger.info("Current combination usage statistics:")
                for service_type, usage in combination_usage.items():
                    min_usage = min(usage.values())
                    max_usage = max(usage.values())
                    self.logger.info(f"{service_type}: min={min_usage}, max={max_usage}")


            # Assign a Region from the Predefined List
            assigned_region = random.choice(self.PREDEFINED_REGIONS)
            self.logger.info(f"Assigned region for dialogue_id '{original_dialogue_id}': {assigned_region}")
            regions = [assigned_region]

            # Extract and anonymize existing dialogue
            processed_dialogue = self.extract_dialogue(example)
            base_conversation = self.generate_base_conversation(processed_dialogue)
            # self.logger.debug(f"Base conversation for dialogue_id '{original_dialogue_id}':\n{base_conversation}")

            # Create hash of the base conversation to check for duplicates
            dialogue_hash = hashlib.sha256(base_conversation.encode('utf-8')).hexdigest()
            if dialogue_hash in self.existing_hashes:
                self.logger.info(f"Duplicate dialogue detected for dialogue_id '{original_dialogue_id}'. Skipping.")
                continue

            # Scenario Diversification: Dynamically generate a scenario for this dialogue

            valid_categories = self.get_categories_for_service(primary_service)
            selected_category = random.choice(valid_categories)
            
            generated_scenario, selected_time_slot = self.generate_dynamic_scenario(
                selected_category, 
                services_to_use,
                assigned_region
            )
            if not generated_scenario:
                self.logger.warning(f"Could not generate scenario for category '{selected_category}' and service '{primary_service}'. Skipping dialogue_id '{original_dialogue_id}'.")
                continue
            self.logger.debug(f"Selected category for dialogue_id '{original_dialogue_id}': {selected_category}")
            self.logger.debug(f"Generated scenario for dialogue_id '{original_dialogue_id}': {generated_scenario}")
            self.logger.debug(f"Selected time slot for dialogue_id '{original_dialogue_id}': {selected_time_slot}")

            # Randomly select emotions from the respective emotion lists
            selected_user_emotions = random.sample(self.USER_EMOTION_LIST, 1)
            selected_assistant_emotions = random.sample(self.ASSISTANT_EMOTION_LIST, 1)
            self.logger.info(f"Selected user emotions for dialogue_id '{original_dialogue_id}': {selected_user_emotions}")
            self.logger.info(f"Selected assistant emotions for dialogue_id '{original_dialogue_id}': {selected_assistant_emotions}")
            resolution_status = random.choices(list(self.RESOLUTION_STATUSES.keys()), 
                                               weights=list(self.RESOLUTION_STATUSES.values()))[0]
            self.logger.info(f"Selected resolution status for dialogue_id '{original_dialogue_id}': {resolution_status}")
            prompt = (
                f"Using the following base conversation as a reference, create a new dialogue for the service(s): {', '.join(services_to_use)}. "
                f"The dialogue should be completely new and more relevant than any existing dialogue. Do not copy any part of existing dialogues. "
                f"The dialogue should be between a user and an assistant.\n\n"
                f"Base Conversation:\n{base_conversation}"
            )

            if resolution_status == "Escalated":
                prompt += (
                    f"\nEnsure that the conversation ends with an escalated status. "
                    f"This means the user's issue or request should be forwarded to a higher authority or another department for resolution. "
                    f"The dialogue should naturally lead to this outcome, showing that the issue requires specialized attention "
                    f"or cannot be resolved within the current support level. "
                    f"The final exchange should clearly reflect the escalated status, with the assistant explaining "
                    f"why the issue needs to be escalated and what the next steps will be."
                )
            else:
                prompt += (
                    f"\nEnsure that the conversation ends with a {resolution_status.lower()} status. "
                    f"The dialogue should naturally lead to this outcome, and the final exchange should clearly reflect the {resolution_status.lower()} status."
                )



            # Randomly select dynamic parameters for this generation
            temperature = random.choice(temperature_options)
            top_p = random.choice(top_p_options)
            frequency_penalty = random.choice(frequency_penalty_options)
            presence_penalty = random.choice(presence_penalty_options)

            generated_dialogue = self.generate_dialogue(
                services_to_use,
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

            try:

                generated_turns = self.process_generated_dialogue(generated_dialogue)
                if not generated_turns:
                    self.logger.warning(f"No valid turns extracted from generated dialogue for dialogue_id '{original_dialogue_id}'. Skipping.")
                    continue
            except Exception as e:
                self.logger.error(f"Error processing generated dialogue for dialogue_id '{original_dialogue_id}': {e}")
                continue
                # Assign emotions to each turn based on the speaker
            # generated_turns = self.assign_selected_emotions(generated_turns, selected_user_emotions, selected_assistant_emotions)

            # Generate conversation text without including it in the JSON
            generated_conversation_text = "\n".join([
                f"User: {turn['utterance']}\n"
                f"Intent: {turn.get('intent', 'N/A')}\n"
                f"Emotion: {turn.get('emotion', 'N/A')}\n"
                f"Assistant: {turn.get('assistant_response', 'N/A')}"
                for turn in generated_turns
            ])
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
                'services': services_to_use,
                'dialogue_id': new_dialogue_id,
                'turns': generated_turns,
                'num_lines': num_lines,
                'user_emotions': selected_user_emotions,
                'assistant_emotions': selected_assistant_emotions,
                'scenario_category': selected_category,
                'generated_scenario': generated_scenario,
                'time_slot': selected_time_slot,
                'regions': regions,
                'resolution_status': resolution_status  # Add the resolution status to the dialogue data
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
        return new_dialogues  # Changed to return the generated dialogues

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

    def test_single_dialogue_extraction(self, index: int = 0):
        """
        Tests the extraction of a single dialogue from the dataset.
        """
        if index >= len(self.dataset):
            self.logger.error(f"Index {index} is out of bounds for the dataset.")
            return

        example = self.dataset[index]
        services = example.get('services', [])
        original_dialogue_id = example.get('dialogue_id', f"dialogue_{index}")

        self.logger.info(f"Testing extraction for dialogue_id '{original_dialogue_id}'")

        processed_dialogue = self.extract_dialogue(example)
        if not processed_dialogue:
            self.logger.warning(f"No turns extracted for dialogue_id '{original_dialogue_id}'.")
            return

        base_conversation = self.generate_base_conversation(processed_dialogue)
        if not base_conversation:
            self.logger.warning(f"Base conversation is empty for dialogue_id '{original_dialogue_id}'.")
            return

        self.logger.info(f"Base conversation for dialogue_id '{original_dialogue_id}':\n{base_conversation}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dialogue Generation Script")
    
    # Required arguments
    parser.add_argument('--total_generations', type=int, required=True, help='Total number of dialogues to generate.')
    
    # Optional arguments with defaults
    parser.add_argument('--output_file', type=str, default='generated_dialogues.json', help='Output JSON file to save generated dialogues.')
    parser.add_argument('--hash_file', type=str, default='dialogue_hashes.json', help='Hash file to save dialogue hashes.')
    parser.add_argument('--embedding_file', type=str, default='dialogue_embeddings.npy', help='File to save dialogue embeddings.')
    parser.add_argument('--similarity_threshold', type=float, default=0.9, help='Similarity threshold for uniqueness (e.g., 0.9).')
    parser.add_argument('--dataset_name', type=str, default='pfb30/multi_woz_v22', help='Name of the dataset to use.')
    parser.add_argument('--min_turns', type=int, nargs=2, default=[5, 10], help='Minimum turns range (e.g., 5 10).')
    parser.add_argument('--max_turns', type=int, nargs=2, default=[7, 20], help='Maximum turns range (e.g., 7 20).')
    parser.add_argument('--temperature', type=float, nargs='+', default=[0.7, 0.8, 0.9, 1.0], help='Temperature options for OpenAI API.')
    parser.add_argument('--top_p', type=float, nargs='+', default=[0.8, 0.85, 0.9, 0.95, 1.0], help='Top-p options for OpenAI API.')
    parser.add_argument('--frequency_penalty', type=float, nargs='+', default=[0.0, 0.2, 0.4, 0.6, 0.7], help='Frequency penalty options for OpenAI API.')
    parser.add_argument('--presence_penalty', type=float, nargs='+', default=[0.0, 0.2, 0.4, 0.6, 0.7], help='Presence penalty options for OpenAI API.')
    
    # Additional options
    parser.add_argument('--test_extraction', action='store_true', help='Test extraction of a single dialogue.')
    parser.add_argument('--extraction_index', type=int, default=0, help='Index of the dialogue to extract for testing.')

    return parser.parse_args()

def main():
    args = parse_arguments()

    config = {
        'output_file': args.output_file,
        'hash_file': args.hash_file,
        'embedding_file': args.embedding_file,
        'similarity_threshold': args.similarity_threshold,
        'dataset_name': args.dataset_name,
        'min_turns_range': tuple(args.min_turns),
        'max_turns_range': tuple(args.max_turns),
        'temperature_range': (min(args.temperature), max(args.temperature)),
        'top_p_range': (min(args.top_p), max(args.top_p)),
        'frequency_penalty_range': (min(args.frequency_penalty), max(args.frequency_penalty)),
        'presence_penalty_range': (min(args.presence_penalty), max(args.presence_penalty)),
        'total_generations': args.total_generations
    }

    generator = DialogueGenerator(config)

    if args.test_extraction:
        generator.test_single_dialogue_extraction(index=args.extraction_index)
    else:
        generated_dialogues = generator.generate_unique_dialogues(
            num_generations=args.total_generations,
            min_turns=args.min_turns[0],
            max_turns=args.max_turns[1],
            temperature_options=args.temperature,
            top_p_options=args.top_p,
            frequency_penalty_options=args.frequency_penalty,
            presence_penalty_options=args.presence_penalty
        )

if __name__ == "__main__":
    main()