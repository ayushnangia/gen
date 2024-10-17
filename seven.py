import json
from openai import OpenAI,OpenAIError
import os
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import time
import re

# ----------------------------
# Configuration
# ----------------------------

# Load environment variables from .env.local file
load_dotenv('.env.local')

# Set OpenAI API key
openai_api_key = os.getenv('OPENAI_KEY')
client=OpenAI(api_key=openai_api_key)
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set OPENAI_KEY in your .env.local file.")

# Define constants
MODEL = "gpt-4o-mini"  # Change to "gpt-3.5-turbo" if needed
MAX_RETRIES = 5
SLEEP_TIME = 10  # Seconds to wait before retrying after a failure

# Define entity types relevant to transportation, including Misc
ENTITY_TYPES = [
    "Date",
    "Time",
    "Location",
    "Address",
    "Person",          # Replaces "Name" for individuals
    "Organization",
    "Service",
    "PhoneNumber",
    "Email",
    "ReferenceNumber",
    "PaymentMethod",
    "VehicleType",
    "TicketNumber",
    "Fare",
    "Route",
    "PickupPoint",
    "DropoffPoint",
    "BookingStatus",
    "SeatNumber",
    "PaymentStatus",
    "Duration",
    "Event",           # New category for events
    "Misc"             # Captures any other entities
]

# Define standardized delimiters
START_DELIM = "[["
END_DELIM = "]]"

# ----------------------------
# Utility Functions
# ----------------------------

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Extract and append entities to a JSON dialogue dataset using OpenAI API.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output JSON file with appended entities.')
    return parser.parse_args()

def load_json(file_path):
    """
    Load JSON data from a file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    """
    Save JSON data to a file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def standardize_entities(extracted_entities):
    """
    Standardize the extracted entities with delimiters, including Misc.
    """
    standardized = {}
    for entity_type in ENTITY_TYPES:
        values = extracted_entities.get(entity_type, [])
        # Remove duplicates and sort the values
        unique_values = sorted(list(set(values)))
        standardized[entity_type] = [f"[[{entity_type.upper()}: {value}]]" for value in unique_values]
    return standardized

def create_prompt(text):
    """
    Create a prompt for the OpenAI API to extract entities, including Misc.
    """
    prompt = (
        "Extract the following entities from the given transportation-related text and format them as a JSON object.\n\n"
        "Entities to extract:\n"
        "- Date: Specific dates mentioned.\n"
        "- Time: Specific times mentioned.\n"
        "- Location: General locations or landmarks.\n"
        "- Address: Specific street addresses.\n"
        "- Person: Names of individuals.\n"
        "- Organization: Names of organizations or companies.\n"
        "- Service: Types of services mentioned (e.g., taxi, train).\n"
        "- PhoneNumber: Contact phone numbers.\n"
        "- Email: Contact email addresses.\n"
        "- ReferenceNumber: Any reference or booking numbers.\n"
        "- PaymentMethod: Methods of payment mentioned.\n"
        "- VehicleType: Types of vehicles mentioned (e.g., Toyota Corolla).\n"
        "- TicketNumber: Specific ticket numbers.\n"
        "- Fare: Monetary amounts related to fares.\n"
        "- Route: Specific routes or paths mentioned.\n"
        "- PickupPoint: Specific pickup locations.\n"
        "- DropoffPoint: Specific drop-off locations.\n"
        "- BookingStatus: Status of bookings (e.g., Confirmed, Pending).\n"
        "- SeatNumber: Specific seat numbers mentioned.\n"
        "- PaymentStatus: Status of payments (e.g., Completed, Failed).\n"
        "- Duration: Time durations mentioned (e.g., \"30 minutes\").\n"
        "- Event: Names of events mentioned.\n"
        "- Misc: Any other relevant entities not covered above.\n\n"
        "Use the following delimiters for each entity: [[ENTITY_TYPE: value]].\n"
        "Ensure that each entity type has its own list and avoid duplicates.\n"
        "If an entity does not fit into any of the predefined categories, categorize it under 'Misc'.\n"
        "Do not assign an entity to any category unless you are certain it fits.\n\n"
        "Text:\n"
        f"{text}\n\n"
        "Extracted Entities:"
    )
    return prompt

def extract_entities_from_text(text, retries=0):
    """
    Use OpenAI API to extract entities from the given text.
    Implements retry logic for robustness.
    """
    prompt = create_prompt(text)
    try:
        response = client.chat.completions.create(
            model=MODEL,        
            messages=[
                {"role": "system", "content": "You are an assistant that extracts specific entities from transportation-related texts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500,
            n=1,
            stop=None
        )
        content = response.choices[0].message.content.strip()
        # Parse the response to extract entities
        extracted = parse_openai_response(content)
        return extracted
    except OpenAIError.RateLimitError as e:
        if retries < MAX_RETRIES:
            print(f"RateLimitError encountered. Retrying in {SLEEP_TIME} seconds... (Attempt {retries + 1}/{MAX_RETRIES})")
            time.sleep(SLEEP_TIME)
            return extract_entities_from_text(text, retries + 1)
        else:
            print("Max retries exceeded for RateLimitError.")
            return {}
    except OpenAIError.APIError as e:
        if retries < MAX_RETRIES:
            print(f"APIError encountered: {e}. Retrying in {SLEEP_TIME} seconds... (Attempt {retries + 1}/{MAX_RETRIES})")
            time.sleep(SLEEP_TIME)
            return extract_entities_from_text(text, retries + 1)
        else:
            print("Max retries exceeded for APIError.")
            return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}

def parse_openai_response(content):
    """
    Parse the OpenAI API response to extract entities, including Misc.
    Assumes the response is a JSON-like structure.
    """
    # Attempt to extract JSON from the response
    try:
        # Remove any leading/trailing text outside the JSON
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        json_str = content[json_start:json_end]
        extracted = json.loads(json_str)
        # Ensure all ENTITY_TYPES are present
        for etype in ENTITY_TYPES:
            if etype not in extracted:
                extracted[etype] = []
        return extracted
    except json.JSONDecodeError:
        # If parsing fails, attempt to extract using regex
        extracted = {etype: [] for etype in ENTITY_TYPES}
        for etype in ENTITY_TYPES:
            pattern = re.escape(START_DELIM) + rf"{etype.upper()}:\s*(.+?){re.escape(END_DELIM)}"
            matches = re.findall(pattern, content, re.IGNORECASE)
            # Clean and strip matches
            matches = [match.strip() for match in matches]
            extracted[etype] = matches
        return extracted

def process_entry(entry):
    """
    Process a single JSON entry to extract and append entities.
    """
    # Concatenate all relevant text fields
    texts = []
    turns = entry.get("turns", [])
    for turn in turns:
        utterance = turn.get("utterance", "")
        assistant_response = turn.get("assistant_response", "")
        texts.append(utterance)
        texts.append(assistant_response)
    generated_scenario = entry.get("generated_scenario", "")
    texts.append(generated_scenario)
    combined_text = "\n".join(texts)
    
    # Extract entities
    extracted_entities = extract_entities_from_text(combined_text)
    standardized_entities = standardize_entities(extracted_entities)
    
    # Append to the entry
    entry["extracted_entities"] = standardized_entities
    return entry

# ----------------------------
# Main Processing Function
# ----------------------------

def main():
    args = parse_arguments()
    
    input_file = args.input_file
    output_file = args.output_file
    
    # Load data
    print(f"Loading data from {input_file}...")
    data = load_json(input_file)
    
    # Check if data is a list
    if not isinstance(data, list):
        raise ValueError("Input JSON file must contain a list of entries.")
    
    # Process each entry with a progress bar
    print("Processing entries...")
    for entry in tqdm(data, desc="Extracting Entities"):
        process_entry(entry)
    
    # Save the updated data
    print(f"Saving updated data to {output_file}...")
    save_json(data, output_file)
    
    print("Entity extraction and appending completed successfully!")

# ----------------------------
# Execute the Script
# ----------------------------

if __name__ == "__main__":
    main()
