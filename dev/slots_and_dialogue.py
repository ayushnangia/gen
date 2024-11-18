import json
import os
from openai import OpenAI, OpenAIError
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import time
import re
# Load environment variables from .env.local file
load_dotenv('.env.local')

# Set OpenAI API key
openai_api_key = os.getenv('OPENAI_KEY')
client = OpenAI(api_key=openai_api_key)

# Define constants
MODEL = "gpt-4o"
MAX_RETRIES = 5
SLEEP_TIME = 10

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process dialogues using OpenAI API.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output JSON file.')
    return parser.parse_args()

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def validate_annotations(annotations):
    required_keys = ["frames", "dialogue_acts"]
    if not all(key in annotations for key in required_keys):
        return False

    return True


def get_annotations_from_openai(dialogue_text, retries=0):
    prompt = f"""
    Given the following dialogue between a user and an assistant:

    {dialogue_text}

    Analyze the dialogue and provide annotations for each turn in the following JSON format. 
    Enclose the JSON output within <json></json> tags:

    <json>
    {{
        "frames": [
            {{
                "service": "service_name",
                "state": {{
                    "active_intent": "intent_name",
                    "requested_slots": ["slot1", "slot2"],
                    "slots_values": {{
                        "slots_values_name": ["slot1", "slot2"],
                        "slots_values_list": [["value1"], ["value2"]]
                    }}
                }},
                "slots": [
                    {{
                        "slot": "slot_name",
                        "value": "slot_value",
                        "start": start_index,
                        "exclusive_end": end_index
                    }}
                ]
            }}
        ],
        "dialogue_acts": [
            {{
                "dialog_act": {{
                    "act_type": ["act_type"],
                    "act_slots": [
                        {{
                            "slot_name": ["slot_name"],
                            "slot_value": ["slot_value"]
                        }}
                    ]
                }},
                "span_info": {{
                    "act_type": ["act_type"],
                    "act_slot_name": ["slot_name"],
                    "act_slot_value": ["slot_value"],
                    "span_start": [start_index],
                    "span_end": [end_index]
                }}
            }}
        ]
    }}
    </json>

    Provide the annotations for each turn in the dialogue.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an assistant that processes dialogues and outputs annotations in the specified JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=2000,
            n=1,
            stop=None
        )
        content = response.choices[0].message.content.strip()
        
        # Extract JSON from between <json></json> tags
        json_match = re.search(r'<json>(.*?)</json>', content, re.DOTALL)
        if not json_match:
            raise ValueError("JSON output not found in the expected format")
        
        json_content = json_match.group(1)
        
        # Parse the JSON response
        annotations = json.loads(json_content)
        
        # Validate the structure of the annotations
        if not validate_annotations(annotations):
            raise ValueError("Invalid annotation structure")
        
        return annotations

    except (OpenAIError, json.JSONDecodeError, ValueError) as e:
        if retries < MAX_RETRIES:
            print(f"Error occurred: {str(e)}. Retrying in {SLEEP_TIME} seconds...")
            time.sleep(SLEEP_TIME)
            return get_annotations_from_openai(dialogue_text, retries + 1)
        else:
            print(f"Max retries reached. Failed to get annotations for dialogue.")
            return None
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")
        return None

def process_dialogue(dialogue):
    dialogue_text = ''
    for turn in dialogue['turns']['utterance']:
        dialogue_text += f"{turn}\n"

    print(f"Dialogue text being sent to OpenAI:\n{dialogue_text}")  # Debug print

    annotations = get_annotations_from_openai(dialogue_text)
    
    if annotations:
        dialogue['annotations'] = annotations
    else:
        dialogue['annotations'] = {"error": "Failed to generate annotations"}
    
    return dialogue

def main():
    args = parse_arguments()
    input_file = args.input_file
    output_file = args.output_file

    print(f"Loading data from {input_file}...")
    data = load_json(input_file)

    print("Processing dialogues...")
    for dialogue in tqdm(data, desc="Processing Dialogues"):
        processed_dialogue = process_dialogue(dialogue)
        dialogue.update(processed_dialogue)

    print(f"Saving updated data to {output_file}...")
    save_json(data, output_file)

    print("Dataset processing completed successfully!")

if __name__ == "__main__":
    main()