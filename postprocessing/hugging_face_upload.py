
import json
import os
import subprocess
import sys
import textwrap
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm

# ==============================
# Configuration
# ==============================

# Path to your dataset file
DATASET_PATH = "deduplicated_dataset.jsonl"  # Ensure this is the path to your JSONL file

# Desired name for your dataset on Hugging Face
DATASET_NAME = "SynWOZ"  # Replace with your desired dataset name

# Description and other metadata
DATASET_DESCRIPTION = (
    "A dataset containing 50k dialogues with various intents and emotions, generated using an advanced dialogue generation pipeline."
)
DATASET_TAGS = ["dialogue", "intent", "emotion", "conversation", "large-scale"]

# Your Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")  # Ensure this environment variable is set

# Your Hugging Face username
HF_USERNAME = "Ayushnangia"  # Replace with your Hugging Face username

# Directory to clone the repository
REPO_DIR = "./temp_repo"

# Git LFS threshold (files larger than this size will be tracked with Git LFS)
GIT_LFS_THRESHOLD = 50 * 1024 * 1024  # 50 MB

# ==============================
# Function Definitions
# ==============================
def create_hf_repository(repo_name, token, private=False):
    """
    Creates a new repository on Hugging Face Hub.
    """
    try:
        create_repo(
            repo_id=repo_name,  # Changed from 'name' to 'repo_id'
            token=token,
            private=private,
            exist_ok=True,
            repo_type="dataset"
        )
        print(f"Repository '{repo_name}' created or already exists.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        sys.exit(1)

def clone_repository(repo_url, repo_dir, token):
    """
    Clones the Hugging Face repository to a local directory.
    """
    try:
        if os.path.exists(repo_dir):
            print(f"Directory '{repo_dir}' already exists. Skipping clone.")
            return
        # Clone with authentication
        repo_url_auth = repo_url.replace("https://", f"https://user:{token}@")
        subprocess.run(["git", "clone", repo_url_auth, repo_dir], check=True)
        print(f"Cloned repository to '{repo_dir}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        sys.exit(1)

def prepare_dataset(dataset_path):
    """
    Loads and prepares the dataset using the datasets library.
    """
    try:
        # Read the JSONL file line by line and normalize the data types
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                item = json.loads(line)
                
                # Convert ALL time_slot values to strings to ensure consistency
                if "time_slot" in item:
                    item["time_slot"] = [str(x) if x is not None else "" for x in item["time_slot"]]
                
                # Ensure other fields that might cause issues are consistent
                if "num_lines" in item:
                    item["num_lines"] = int(item["num_lines"])
                if "turn_number" in item.get("turns", [{}])[0]:
                    for turn in item["turns"]:
                        turn["turn_number"] = int(turn["turn_number"])
                
                data.append(item)
                
        print(f"Loaded dataset from '{dataset_path}'.")
        return data
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        sys.exit(1)


def save_dataset_to_repo(dataset, repo_dir, dataset_filename="dataset.jsonl"):
    """
    Saves the dataset to the repository directory.
    """
    try:
        output_path = os.path.join(repo_dir, dataset_filename)

        if os.path.exists(output_path):
            os.remove(output_path)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc="Writing dataset to repo"):
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')

        file_size = os.path.getsize(output_path)
        if file_size >= GIT_LFS_THRESHOLD:
            print(f"File size is {file_size} bytes, which exceeds the threshold. Using Git LFS.")
            subprocess.run(["git", "lfs", "install"], cwd=repo_dir, check=True)
            subprocess.run(["git", "lfs", "track", dataset_filename], cwd=repo_dir, check=True)
            subprocess.run(["git", "add", ".gitattributes"], cwd=repo_dir, check=True)

        print(f"Dataset saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving dataset to repo: {e}")
        sys.exit(1)
def create_readme(repo_dir, dataset_name, description, tags):
    """
    Creates a README.md file in the repository directory.
    """
    readme_content = f"""---
annotations_creators:
- machine-generated
language_creators:
- machine-generated
language:
- en
license:
- mit
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- text-generation
- fill-mask
- token-classification
- text-classification
task_ids:
- dialogue-modeling
- multi-class-classification
- parsing
---

# {dataset_name}

{description}

## Dataset Summary

This dataset consists of 50k dialogues generated by an advanced dialogue generation pipeline. The dialogues simulate realistic interactions across various services such as restaurants, hotels, taxis, and more, incorporating diverse scenarios, emotions, and resolution statuses.

## Supported Tasks and Leaderboards

- **Dialogue Modeling**
- **Emotion Recognition**
- **Intent Classification**
- **Conversational AI Research**

## Languages

The dataset is primarily in English (`en`).

## Dataset Structure

### Data Instances

An example from the dataset:

```json
{{
  "services": ["restaurant", "taxi", "attraction"],
  "dialogue_id": "MUL1835.json_generated_f5c8b86d-92a6-4108-8a6a-4822609b44fe",
  "turns": [
    {{
      "turn_number": 1,
      "utterance": "Excuse me, we need to talk about our taxi fare from earlier this evening.",
      "intent": "Request for Assistance",
      "assistant_response": "Of course! What seems to be the issue with the taxi fare?"
    }},
    ...
  ],
  "num_lines": 5,
  "user_emotions": ["Suspicious"],
  "assistant_emotions": ["Precise"],
  "scenario_category": "refund_request_general",
  "generated_scenario": "After enjoying a lovely dinner at a local trattoria in Trastevere...",
  "time_slot": ["17", "21", "Evening"],
  "regions": ["Rome"],
  "resolution_status": "Resolved"
}}
```

### Data Fields

- **services** (`List[str]`): Services involved in the dialogue.
- **dialogue_id** (`str`): Unique identifier for the dialogue.
- **turns** (`List[Dict]`): List of dialogue turns containing:
  - **turn_number** (`int`): The turn number in the dialogue.
  - **utterance** (`str`): The user's utterance.
  - **intent** (`str`): The intent behind the user's utterance.
  - **assistant_response** (`str`): The assistant's response.
- **num_lines** (`int`): Total number of turns in the dialogue.
- **user_emotions** (`List[str]`): Emotions expressed by the user.
- **assistant_emotions** (`List[str]`): Emotions expressed by the assistant.
- **scenario_category** (`str`): Category of the scenario.
- **generated_scenario** (`str`): Description of the scenario.
- **time_slot** (`List[str]`): Time information `["start_hour", "end_hour", "Period"]` (all values stored as strings).
- **regions** (`List[str]`): Geographic regions involved.
- **resolution_status** (`str`): Status of the dialogue resolution (e.g., Resolved, Failed, Escalated).

### Data Splits

The dataset is provided as a single file without predefined splits.

## Dataset Creation

The dialogues were generated using a dialogue generation pipeline that involves:

- **Persona Management**: Incorporating diverse user personas to enhance realism.
- **Scenario Generation**: Crafting specific scenarios based on service categories, regions, and time slots.
- **Dialogue Generation**: Utilizing OpenAI's GPT models to produce dialogues.
- **Uniqueness Verification**: Ensuring dialogues are unique using hashing and semantic embedding comparisons.
- **Emotion Assignment**: Assigning emotions to users and assistants to add depth to conversations.

### Source Data

- **Primary Dataset**: Derived from `multi_woz_v22` dataset.
- **Persona Dataset**: Utilized `FinePersonas-v0.1-clustering-100k` for persona diversity.

### Annotations

- **Annotation Process**: The dataset is machine-generated, and annotations are produced programmatically.
- **Annotation Fields**: Intents, emotions, and scenario details.


## Usage

To use this dataset, you can load it using the `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("{HF_USERNAME}/{DATASET_NAME}")
```

## License

This dataset is released under the [MIT License](LICENSE).

## Citation

If you use this dataset in your work, please cite it as:

```
@dataset{{{DATASET_NAME}_2024,
  author = {{Ayush Nangia}},
  title = {{{DATASET_NAME}}},
  year = {{2024}},
  url = {{https://huggingface.co/datasets/{HF_USERNAME}/{DATASET_NAME}}},
}}
```
"""

    # Remove any leading whitespace from the markdown content
    readme_content = textwrap.dedent(readme_content)

    readme_path = os.path.join(repo_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"README.md created at '{readme_path}'.")

def push_to_hf(repo_dir):
    """
    Pushes the local repository to Hugging Face Hub.
    """
    try:
        # Navigate to the repository directory
        current_dir = os.getcwd()
        os.chdir(repo_dir)

        # Configure Git (if not already configured)
        subprocess.run(["git", "config", "user.name", "Your Name"], check=True)
        subprocess.run(["git", "config", "user.email", "your_email@example.com"], check=True)

        # Add all files, commit, and push
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Add dataset and README"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("Dataset pushed to Hugging Face Hub successfully.")

        # Return to the original directory
        os.chdir(current_dir)
    except subprocess.CalledProcessError as e:
        print(f"Error pushing to Hugging Face Hub: {e}")
        sys.exit(1)

# ==============================
# Main Execution
# ==============================

def main():
    # Check for Hugging Face token
    if not HF_TOKEN:
        print("Error: Hugging Face token not found. Please set the HF_TOKEN environment variable.")
        sys.exit(1)

    # Step 1: Create Hugging Face repository
    repo_name = f"{HF_USERNAME}/{DATASET_NAME}"
    create_hf_repository(repo_name, HF_TOKEN, private=False)

    # Step 2: Clone the repository locally
    repo_url = f"https://huggingface.co/datasets/{repo_name}"
    clone_repository(repo_url, REPO_DIR, HF_TOKEN)

    # Step 3: Prepare the dataset (using streaming to handle large data)
    dataset = prepare_dataset(DATASET_PATH)

    # Step 4: Save the dataset in the repository directory
    save_dataset_to_repo(dataset, REPO_DIR, dataset_filename="dataset.jsonl")

    # Step 5: Create a README file
    create_readme(REPO_DIR, DATASET_NAME, DATASET_DESCRIPTION, DATASET_TAGS)

    # Step 6: Push to Hugging Face
    push_to_hf(REPO_DIR)

    # Optional: Clean up the cloned repository
    # import shutil
    # shutil.rmtree(REPO_DIR)
    # print(f"Cleaned up repository directory '{REPO_DIR}'.")

if __name__ == "__main__":
    main()

