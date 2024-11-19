import os
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import argparse

def download_sentence_transformer(model_name='all-MiniLM-L6-v2', save_path='./models/sentence_transformer'):
    """
    Downloads a Sentence Transformer model and saves it locally.
    """
    print(f"Preparing to download Sentence Transformer model: {model_name}")
    
    os.makedirs(save_path, exist_ok=True)
    
    if os.path.exists(os.path.join(save_path, 'config.json')):
        print(f"Model already exists at {save_path}. Skipping download.")
        return save_path
    
    print(f"Downloading model {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Saving model to {save_path}")
    model.save(save_path)
    
    print(f"Model successfully downloaded and saved to {save_path}")
    return save_path

def download_and_save_dataset(dataset_name, save_path):
    """
    Downloads a dataset from Hugging Face and saves it locally.
    If the dataset already exists at the specified path, it skips the download.
    """
    if os.path.exists(save_path):
        print(f"Dataset already exists at: {save_path}")
        print("Skipping download.")
        return

    print(f"Downloading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, download_mode="force_redownload", trust_remote_code=True)
    
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Saving dataset to: {save_path}")
    dataset.save_to_disk(save_path)
    
    print("Dataset downloaded and saved successfully.")

def setup_environment(model_name, model_path, dataset_name, dataset_path):
    """
    Sets up the environment by downloading the model and dataset.
    """
    print("Setting up environment for dialogue generation...")
    
    # Download and save the SentenceTransformer model
    model_save_path = download_sentence_transformer(model_name, model_path)
    
    # Verify the model can be loaded
    try:
        loaded_model = SentenceTransformer(model_save_path)
        print("SentenceTransformer model loaded successfully. Ready for use in your main project.")
    except Exception as e:
        print(f"Error loading the SentenceTransformer model: {e}")
    
    # Download and save the dataset
    download_and_save_dataset(dataset_name, dataset_path)
    
    print("\nSetup complete!")
    print(f"SentenceTransformer model is saved at: {model_save_path}")
    print(f"Dataset is saved at: {dataset_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Setup Script for Dialogue Generation")
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help='Name of the SentenceTransformer model to download')
    parser.add_argument('--model_path', type=str, default='./models/sentence_transformer', help='Path to save the SentenceTransformer model')
    parser.add_argument('--dataset_name', type=str, default='pfb30/multi_woz_v22', help='Name of the dataset to download')
    parser.add_argument('--dataset_path', type=str, default='./local_datasets/multi_woz_v22', help='Path to save the dataset')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    setup_environment(args.model_name, args.model_path, args.dataset_name, args.dataset_path)
    download_and_save_dataset("argilla/FinePersonas-v0.1-clustering-100k","./local_datasets/FinePersonas-v0.1-clustering-100k")
