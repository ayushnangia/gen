import json
from datasets import Dataset, Features, Sequence, Value, load_dataset
from huggingface_hub import HfApi, create_repo
import os
import subprocess
from tqdm import tqdm

# ==============================
# Configuration
# ==============================

# Path to your dataset file
DATASET_PATH = "dataset.json"  # Consider converting to Arrow for efficiency
# DATASET_PATH = "dataset.arrow"

# Desired name for your dataset on Hugging Face
DATASET_NAME = "your_dataset_name"

# Description and other metadata
DATASET_DESCRIPTION = "A large dataset containing 100k dialogues with various intents and emotions."
DATASET_TAGS = ["dialogue", "intent", "emotion", "conversation", "large-scale"]

# Your Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")  # Ensure this environment variable is set

# Your Hugging Face username
HF_USERNAME = "your_hf_username"

# Directory to clone the repository
REPO_DIR = "./temp_repo"

# ==============================
# Function Definitions
# ==============================

def create_hf_repository(repo_name, token, private=False):
    """
    Creates a new repository on Hugging Face Hub.
    """
    api = HfApi()
    try:
        create_repo(
            name=repo_name,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="dataset"
        )
        print(f"Repository '{repo_name}' created or already exists.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        exit(1)

def clone_repository(repo_url, repo_dir, token):
    """
    Clones the Hugging Face repository to a local directory.
    """
    try:
        if os.path.exists(repo_dir):
            print(f"Directory '{repo_dir}' already exists. Skipping clone.")
            return
        # Clone with authentication
        repo_url_auth = repo_url.replace("https://", f"https://{token}@")
        subprocess.run(["git", "clone", repo_url_auth, repo_dir], check=True)
        print(f"Cloned repository to '{repo_dir}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        exit(1)

def prepare_dataset(dataset_path):
    """
    Loads and prepares the dataset using the datasets library.
    """
    try:
        # For very large datasets, using streaming is recommended
        dataset = load_dataset('json', data_files=dataset_path, split='train', streaming=True)
        print(f"Loaded dataset with {len(dataset)} records (streaming).")
        return dataset
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        exit(1)

def save_dataset_to_repo(dataset, repo_dir, dataset_filename="dataset.json"):
    """
    Saves the dataset to the repository directory in chunks.
    """
    try:
        output_path = os.path.join(repo_dir, dataset_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc="Writing dataset to repo"):
                json.dump(example, f)
                f.write('\n')  # Ensure newline-delimited JSON
        print(f"Dataset saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving dataset to repo: {e}")
        exit(1)

def create_readme(repo_dir, dataset_name, description, tags):
    """
    Creates a README.md file in the repository directory.
    """
    readme_content = f"""
# {dataset_name}

{description}

## Dataset Structure

The dataset contains dialogues with various intents and emotions. Each record includes:

- **services**: List of services involved.
- **dialogue_id**: Unique identifier for the dialogue.
- **turns**: List of dialogue turns with user and assistant interactions.
- **num_lines**: Number of lines in the dialogue.
- **user_emotions**: Emotions expressed by the user.
- **assistant_emotions**: Emotions expressed by the assistant.
- **scenario_category**: Category of the scenario.
- **generated_scenario**: Description of the scenario.
- **time_slot**: Time slot information.
- **regions**: Geographic regions involved.
- **resolution_status**: Status of the dialogue resolution.

## Tags

{", ".join(tags)}

## License

Specify your dataset's license here.

## Citation

Provide citation information if applicable.
"""
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
        subprocess.run(["git", "config", "user.name", "your_username"], check=True)
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
        exit(1)

# ==============================
# Main Execution
# ==============================

def main():
    # Check for Hugging Face token
    if not HF_TOKEN:
        print("Error: Hugging Face token not found. Please set the HF_TOKEN environment variable.")
        exit(1)

    # Step 1: Create Hugging Face repository
    repo_name = f"{HF_USERNAME}/{DATASET_NAME}"
    create_hf_repository(repo_name, HF_TOKEN, private=False)

    # Step 2: Clone the repository locally
    repo_url = f"https://huggingface.co/{repo_name}"
    clone_repository(repo_url, REPO_DIR, HF_TOKEN)

    # Step 3: Prepare the dataset (using streaming to handle large data)
    dataset = prepare_dataset(DATASET_PATH)

    # Step 4: Save the dataset in the repository directory
    save_dataset_to_repo(dataset, REPO_DIR, dataset_filename="dataset.json")

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
