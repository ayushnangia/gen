# Capstone Dialogue Generation Script

## Overview

The `capstone.py` script is a powerful tool designed to generate high-quality, unique dialogues using OpenAI's Chat Completion API. Leveraging a dataset from Hugging Face and advanced natural language processing techniques, this script ensures that each generated dialogue is both relevant and distinct. It incorporates sophisticated uniqueness checks and allows for dynamic parameter tuning to balance creativity and coherence in the outputs.

## Features

1. **Unique Dialogue Generation**:
    - **Exact Duplication Prevention**: Utilizes hashing to ensure that no two dialogues are identical in content.
    - **Semantic Similarity Checks**: Employs cosine similarity with embeddings to detect and prevent near-duplicate dialogues, ensuring semantic diversity.

2. **Dynamic Parameter Tuning**:
    - **Adjustable Settings**: Allows users to experiment with OpenAI API parameters such as `temperature`, `top_p`, `frequency_penalty`, and `presence_penalty` to fine-tune the creativity and coherence of the generated dialogues.
    - **Flexible Configuration**: Parameters can be specified via command-line arguments, enabling varied and tailored dialogue generation.

3. **Anonymization**:
    - **Privacy Preservation**: Automatically anonymizes sensitive information in dialogues, such as locations, times, numbers, and personal names, ensuring privacy and compliance with data protection standards.

4. **Comprehensive Logging**:
    - **Real-time Monitoring**: Logs detailed information about the generation process, including successes, warnings, and errors, to both console and log files for easy monitoring and troubleshooting.

5. **Enhanced Output Metadata**:
    - **Structured Data**: Each generated dialogue includes the number of lines (speaker turns), facilitating easier analysis and processing.
    - **Detailed Records**: Outputs are saved in a structured JSON format, capturing all relevant aspects of the dialogues.

## Prerequisites

Before using the `capstone.py` script, ensure that your environment meets the following requirements:

- **Python Version**: Python 3.7 or higher.
- **Python Packages**: Install the necessary Python packages using `pip`:

    ```bash
    pip install openai datasets spacy tqdm python-dotenv sentence-transformers scikit-learn numpy
    python -m spacy download en_core_web_sm
    ```

- **OpenAI API Key**: Obtain an API key from [OpenAI](https://platform.openai.com/account/api-keys) and set it in a `.env.local` file in the same directory as `capstone.py`:

    ```
    OPENAI_KEY=your_openai_api_key_here
    ```

- **Dataset Access**: Ensure access to the Hugging Face dataset `Ayushnangia/transport_multiwoz_v22`.

## Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/ayushnangia/gen.git
    cd gen
    ```

2. **Set Up the Environment**:

    - Create a virtual environment (optional but recommended):

        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```

    - Install the required packages:

        ```bash
        pip install -r requirements.txt
        ```

3. **Configure Environment Variables**:

    - Create a `.env.local` file in the project directory and add your OpenAI API key:

        ```
        OPENAI_KEY=your_openai_api_key_here
        ```

## Usage

The `capstone.py` script can be executed via the command line with various arguments to customize its behavior.

### Command-Line Arguments

- `--num_generations`: **(Required)** Number of unique dialogues to generate.

- `--min_turns`: **(Optional)** Minimum number of dialogue turns per conversation. Default is `3`.

- `--max_turns`: **(Optional)** Maximum number of dialogue turns per conversation. Default is `10`.

- `--temperature`: **(Optional)** Sampling temperature for OpenAI API. Controls creativity. Higher values (e.g., `0.9`) increase randomness, while lower values (e.g., `0.5`) make outputs more deterministic. Default is `0.9`.

- `--top_p`: **(Optional)** Nucleus sampling parameter for OpenAI API. Controls diversity by limiting token selection to a subset with cumulative probability `top_p`. Default is `0.95`.

- `--frequency_penalty`: **(Optional)** Penalizes new tokens based on their existing frequency in the text so far. Helps reduce repetition. Default is `0.5`.

- `--presence_penalty`: **(Optional)** Penalizes new tokens based on whether they appear in the text so far. Encourages the introduction of new topics. Default is `0.5`.

- `--output_file`: **(Optional)** Path to the output JSON file where generated dialogues will be saved. Default is `generated_dialogues.json`.

### Example Commands

1. **Basic Usage**:

    Generate 10 dialogues with default parameters.

    ```bash
    python capstone.py --num_generations 10
    ```

2. **Custom Turns and Output File**:

    Generate 20 dialogues with a minimum of 4 turns and a maximum of 8 turns, saving the output to `my_dialogues.json`.

    ```bash
    python capstone.py --num_generations 20 --min_turns 4 --max_turns 8 --output_file my_dialogues.json
    ```

3. **Dynamic Parameter Tuning**:

    Generate 15 dialogues with customized OpenAI API parameters to balance creativity and coherence.

    ```bash
    python capstone.py --num_generations 15 --temperature 0.7 --top_p 0.9 --frequency_penalty 0.3 --presence_penalty 0.4
    ```

## Logging

The script maintains detailed logs of its operations to assist with monitoring and troubleshooting.

- **Log File**: `dialogue_generation.log` - Contains detailed logs of the dialogue generation process, including information, warnings, and errors.

- **Console Output**: Real-time logging information is also displayed in the console during execution.

## Output Structure

Generated dialogues are saved in a structured JSON format, ensuring ease of access and analysis.

Each dialogue entry includes the following fields:

- `services`: List of services associated with the dialogue (e.g., `["taxi"]`).

- `dialogue_id`: Unique identifier for the dialogue, combining the original ID and a run index (e.g., `"dialogue_123_generated_456"`).

- `turns`: List of turns, each containing:
    - `speaker`: The speaker label (`"USER"` or `"ASSISTANT"`).
    - `utterance`: The anonymized text spoken by the speaker.

- `base_conversation`: The formatted conversation string, combining all turns.

- `num_lines`: **(Added)** The number of lines (speaker turns) in the conversation, providing a quick reference to the dialogue's length.

### Example Output Entry

```json
{
    "services": ["taxi"],
    "dialogue_id": "dialogue_123_generated_456",
    "turns": [
        {
            "speaker": "USER",
            "utterance": "I need to book a taxi to the airport."
        },
        {
            "speaker": "ASSISTANT",
            "utterance": "Sure, I can help with that. What time would you like the taxi to arrive?"
        }
        // Additional turns...
    ],
    "base_conversation": "USER: I need to book a taxi to the airport.\nASSISTANT: Sure, I can help with that. What time would you like the taxi to arrive?\n...",
    "num_lines": 6
}
```

## Ensuring Uniqueness

### Exact Duplication Prevention

- **Hashing Mechanism**: Each dialogue's `base_conversation` is hashed using SHA256 and stored in `dialogue_hashes.json`.
  
- **Duplicate Checks**: Before adding a new dialogue, the script checks if its hash already exists. If it does, the dialogue is skipped to prevent exact duplicates.

### Semantic Similarity Checks

- **Embeddings**: Utilizes the `sentence-transformers` library to generate embeddings for each dialogue's `base_conversation`.

- **Cosine Similarity**: Compares the embeddings of newly generated dialogues against existing ones. If the cosine similarity exceeds a predefined threshold (e.g., `0.8`), the dialogue is considered a near-duplicate and is skipped.

- **Threshold Configuration**: The similarity threshold can be adjusted within the script to fine-tune the sensitivity of duplication detection.

## Dynamic Parameter Tuning

The script allows users to experiment with various OpenAI API parameters to control the nature of the generated dialogues:

- **Temperature (`--temperature`)**:
    - **Purpose**: Controls the randomness of the output. Higher values (e.g., `0.9`) make the output more diverse, while lower values (e.g., `0.5`) make it more focused and deterministic.
  
- **Top-p (`--top_p`)**:
    - **Purpose**: Implements nucleus sampling by limiting token selection to a subset with a cumulative probability of `top_p`. Balances diversity and coherence.
  
- **Frequency Penalty (`--frequency_penalty`)**:
    - **Purpose**: Penalizes new tokens based on their existing frequency in the text, reducing repetition.
  
- **Presence Penalty (`--presence_penalty`)**:
    - **Purpose**: Penalizes new tokens based on whether they appear in the text so far, encouraging the introduction of new topics.

Users can specify these parameters via command-line arguments to achieve the desired balance between creativity and coherence in the generated dialogues.

## Additional Metadata

- **Number of Lines (`num_lines`)**: Each generated dialogue includes a `num_lines` field indicating the total number of speaker turns. This metadata provides a quick reference to the dialogue's length and complexity.

##Do not reduce any functionally or make any changes to the script i will give you as you are not aware of the changes in Tech due to your knowledge cutoff date

## Improvements:

Professionalism Score in the converstation
Emotion change whole approach 


Bias Mitigation Strategies
Evaluation Framework for Synthetic Data Quality
3. Contextual and Coherent Dialogue Flow
Description: Ensure that dialogues maintain contextual coherence throughout the conversation, reflecting natural progression and logical flow of information.

Improvements:

Context Tracking: Implement mechanisms to track and maintain context across multiple turns, ensuring responses are relevant to previous statements.
Coherence Models: Use models that prioritize coherent and contextually appropriate responses.
Relevant Citations:

Henderson et al., 2014 – The DSTC3 Evaluation Framework: Tracking and Evaluating Multi-Domain Dialogue State Tracking.
Vinyals & Le, 2015 – A Neural Conversational Model.

4. Incorporation of User Intent and Slot Filling
Description: Model user intents and relevant slots (e.g., pickup location, destination, time) to structure conversations and ensure all necessary information is exchanged.

Improvements:

Intent Recognition: Define and categorize various user intents related to transportation queries.
Slot Filling Mechanism: Implement slot-filling strategies to capture and utilize essential information within dialogues.
Relevant Citations:

Larson et al., 2019 – Don't Stop Pretraining: Adapt Language Models to Domains and Tasks.
Chen et al., 2019 – Latent Action Dialogue Policies.
