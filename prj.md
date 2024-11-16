# Dialogue Generation Script

## Overview

The **Dialogue Generation Script** is a sophisticated tool designed to create high-quality, unique dialogues tailored for various services using OpenAI's GPT-4 model. By integrating dataset processing, persona selection, dynamic scenario generation, emotion assignment, robust uniqueness verification, and advanced features like data balancing and bias mitigation, this script facilitates the generation of diverse and contextually rich conversational data. This data is ideal for training and enhancing conversational AI systems, ensuring they can handle a wide array of real-world interactions effectively.

## Features

- **Dataset Integration**
  - Utilizes the [MultiWOZ v2.2](https://github.com/budzianowski/multiwoz) dataset for foundational dialogues.
  - Incorporates personas from the [FinePersonas](https://github.com/PolyAI-LDN/FinePersonas) dataset to add depth and variety to conversations.
  - Supports multi-service dialogues, allowing the selection and combination of multiple services within a single conversation context.

- **Persona Selection**
  - Selects diverse personas based on clustering and summary labels to ensure varied conversational styles and backgrounds.
  - Randomly assigns personas to dialogues to enhance diversity and realism.

- **Dynamic Scenario Generation**
  - Generates specific scenarios tailored to service categories, regions, and time slots using OpenAI's API.
  - Incorporates user personas, selected time slots, and predefined regions to create contextually rich scenarios.
  - Supports the addition of new conversation domains, such as flight-related dialogues, expanding the versatility of the generated data.

- **Emotion Assignment**
  - Assigns distinct emotions to both user and assistant roles, enhancing the emotional dynamics of the dialogue.
  - Supports a wide range of emotions for users (e.g., Frustrated, Happy, Anxious) and assistants (e.g., Professional, Reassuring, Knowledgeable).
  - Ensures that emotions are consistently reflected throughout the conversation, contributing to more natural and engaging interactions.

- **Scenario Categories and Services**
  - Supports multiple service categories, including general services and specific ones like restaurant, hotel, train, attraction, taxi, hospital, bus, and flight.
  - Each service category has its own set of relevant scenario types (e.g., dining reservation for restaurants, room reservation for hotels).
  - Facilitates the creation of multi-service dialogues, allowing complex interactions that span multiple domains.

- **Predefined Regions and Time Slots**
  - Incorporates a comprehensive list of predefined regions (e.g., Tokyo, New York, London) to set the geographical context of dialogues.
  - Utilizes predefined time slots (e.g., Early Morning, Afternoon) to add temporal context to scenarios.
  - Enhances the realism of dialogues by embedding regional and temporal specifics.

- **Resolution Statuses**
  - Generates dialogues with varied resolution outcomes: Resolved, Failed, or Escalated.
  - Ensures dialogues naturally lead to the selected resolution status, enhancing realism.
  - Allows for the simulation of different conversational outcomes based on the resolution status.

- **Uniqueness Verification**
  - Ensures generated dialogues are unique by:
    - **Hash-Based Checking**: Uses SHA-256 hashing to prevent exact duplicates.
    - **Semantic Similarity Checks**: Employs Sentence Transformers and cosine similarity to ensure semantic uniqueness against existing embeddings.
  - Maintains a repository of existing dialogues and embeddings to continuously enforce uniqueness standards.

- **Data Balancer**
  - Implements data balancing techniques to ensure a uniform distribution of dialogues across various categories, services, and regions.
  - Prevents overrepresentation of certain dialogue types, promoting diversity and fairness in the generated dataset.

- **Bias Mitigation**
  - Integrates strategies to identify and reduce biases in generated dialogues.
  - Utilizes tools and methodologies to ensure that dialogues are fair, unbiased, and representative of diverse user interactions.
  - Continuously monitors and adjusts generation parameters to uphold ethical standards.

- **Omni Moderation**
  - Incorporates comprehensive moderation mechanisms to filter out inappropriate or sensitive content from generated dialogues.
  - Ensures that all dialogues adhere to predefined content policies and ethical guidelines.
  - Utilizes both automated checks and manual reviews to maintain content quality and safety.

- **Text Clustering and Analysis**
  - Employs text clustering techniques using repositories like [distilabel](https://github.com/yourusername/distilabel) and [text-clustering](https://github.com/yourusername/text-clustering) for in-depth analysis of generated dialogues.
  - Facilitates the categorization and summarization of dialogues, enhancing the ability to derive insights and optimize generation processes.
  - Supports the identification of prevalent themes, intents, and conversational patterns within the dataset.

- **Script Output Validation**
  - Implements mechanisms for stakeholders to review and validate the outputs of the dialogue generation script.
  - Ensures that generated dialogues meet quality standards and align with project objectives before integration into larger systems.

- **Testing Utilities**
  - Includes functionality to test the extraction and processing of single dialogues from the dataset, aiding in debugging and validation.
  - Facilitates the verification of individual components, ensuring the reliability and accuracy of the dialogue generation pipeline.

- **Configurable Parameters**
  - Offers extensive customization through command-line arguments, allowing users to define parameters like the number of dialogues, output filenames, similarity thresholds, turn ranges, and OpenAI API settings (temperature, top_p, frequency_penalty, presence_penalty).
  - Enables fine-tuning of generation processes to meet specific project requirements and preferences.

- **Logging**
  - Implements detailed logging to both console and log files (`dialogue_generation.log`), facilitating monitoring and debugging.
  - Logs include information on dataset loading, dialogue generation progress, scenario selections, emotion assignments, uniqueness checks, bias mitigation activities, and error handling.

- **Incremental Saving**
  - Supports batch-wise saving of generated dialogues to manage large-scale data generation efficiently.
  - Saves embeddings and hashes incrementally to maintain performance and prevent data loss.
  - Ensures data integrity by regularly persisting progress, allowing for seamless continuation in case of interruptions.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Generating Dialogues](#generating-dialogues)
  - [Testing Dialogue Extraction](#testing-dialogue-extraction)
- [Output Files](#output-files)
- [Logging](#logging)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Requirements

Before running the script, ensure that the following dependencies are installed:

- **Python**: Version 3.8 or higher.
- **Python Libraries**:
  - [OpenAI Python SDK](https://pypi.org/project/openai/)
  - [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
  - [Sentence Transformers](https://www.sbert.net/)
  - [scikit-learn](https://scikit-learn.org/)
  - [spaCy](https://spacy.io/)
  - [tqdm](https://tqdm.github.io/)
  - [dotenv](https://pypi.org/project/python-dotenv/)
  - [uuid](https://docs.python.org/3/library/uuid.html)
  - [distilabel](https://github.com/yourusername/distilabel) (for text clustering and analysis)
  - [text-clustering](https://github.com/yourusername/text-clustering) (for text clustering and analysis)
  - Additional Python libraries as specified in `requirements.txt`.

Ensure all dependencies are listed in the `requirements.txt` file for easy installation.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/dialogue-generation.git
   cd dialogue-generation
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Necessary Models and Datasets**

   - **spaCy Model**

     The script attempts to download the `en_core_web_sm` model automatically. If it fails, manually install it:

     ```bash
     python -m spacy download en_core_web_sm
     ```

   - **Sentence Transformer Model**

     Ensure that the SentenceTransformer model is available at `./models/sentence_transformer`. If not, download or train a suitable model and place it in the specified directory.

   - **Datasets**

     Download and place the required datasets in the `./local_datasets/` directory:

     - **MultiWOZ v2.2**: `./local_datasets/multi_woz_v22`
       - Download from [MultiWOZ GitHub](https://github.com/budzianowski/multiwoz).
     - **FinePersonas**: `./local_datasets/FinePersonas-v0.1-clustering-100k`
       - Download from [FinePersonas GitHub](https://github.com/PolyAI-LDN/FinePersonas).

## Configuration

1. **Environment Variables**

   Create a `.env.local` file in the root directory with the following content:

   ```env
   OPENAI_KEY=your_openai_api_key_here
   ```

   Replace `your_openai_api_key_here` with your actual OpenAI API key.

2. **Script Configuration**

   The script allows extensive configuration through command-line arguments (detailed in the [Usage](#usage) section). Default values are provided for optional parameters but can be overridden as needed.

## Usage

The script can be executed via the command line, offering various options to customize dialogue generation.

### Command-Line Arguments

Below are the available command-line arguments:

| Argument                 | Type           | Required | Default                                | Description                                                                                       |
|--------------------------|----------------|----------|----------------------------------------|---------------------------------------------------------------------------------------------------|
| `--total_generations`    | `int`          | Yes      | N/A                                    | **Total number of dialogues** to generate.                                                       |
| `--output_file`          | `str`          | No       | `generated_dialogues.json`             | Output JSON file to save generated dialogues.                                                     |
| `--hash_file`            | `str`          | No       | `dialogue_hashes.json`                 | Hash file to save dialogue hashes for uniqueness checks.                                         |
| `--embedding_file`       | `str`          | No       | `dialogue_embeddings.npy`              | File to save dialogue embeddings used for semantic similarity checks.                            |
| `--similarity_threshold` | `float`        | No       | `0.9`                                  | Similarity threshold for ensuring uniqueness (e.g., `0.9`).                                     |
| `--dataset_name`         | `str`          | No       | `pfb30/multi_woz_v22`                  | Name of the dataset to use for base dialogues.                                                   |
| `--min_turns`            | `int` `int`    | No       | `5 10`                                 | Minimum turns range for generated dialogues (e.g., `5 10`).                                     |
| `--max_turns`            | `int` `int`    | No       | `7 20`                                 | Maximum turns range for generated dialogues (e.g., `7 20`).                                     |
| `--temperature`          | `float`...     | No       | `0.7 0.8 0.9 1.0`                      | Temperature options for OpenAI API to introduce variability.                                    |
| `--top_p`                | `float`...     | No       | `0.8 0.85 0.9 0.95 1.0`                | Top-p options for OpenAI API to control nucleus sampling.                                       |
| `--frequency_penalty`    | `float`...     | No       | `0.0 0.2 0.4 0.6 0.7`                   | Frequency penalty options for OpenAI API to reduce repetition.                                  |
| `--presence_penalty`     | `float`...     | No       | `0.0 0.2 0.4 0.6 0.7`                   | Presence penalty options for OpenAI API to encourage new topic introduction.                     |
| `--test_extraction`      | `flag`         | No       | `False`                                | Flag to test the extraction of a single dialogue from the dataset.                              |
| `--extraction_index`     | `int`          | No       | `0`                                    | Index of the dialogue to extract for testing purposes.                                          |

### Generating Dialogues

To generate dialogues, execute the script with the required `--total_generations` argument. You can also customize optional parameters as needed.

**Examples:**

1. **Generate 100 Dialogues with Default Settings**

   ```bash
   python dialogue_generation.py --total_generations 100
   ```

2. **Generate 500 Dialogues with Custom Output File and Similarity Threshold**

   ```bash
   python dialogue_generation.py --total_generations 500 --output_file my_dialogues.json --similarity_threshold 0.85
   ```

3. **Generate 200 Dialogues with Specific Temperature and Top-p Options**

   ```bash
   python dialogue_generation.py --total_generations 200 --temperature 0.7 0.9 --top_p 0.8 0.95
   ```

4. **Generate Dialogues with Specific Frequency and Presence Penalties**

   ```bash
   python dialogue_generation.py --total_generations 300 --frequency_penalty 0.2 0.4 --presence_penalty 0.2 0.4
   ```

**Process Overview:**

1. **Dataset Sampling**: Randomly selects dialogues from the MultiWOZ v2.2 dataset based on the specified number of generations.
2. **Region Assignment**: Assigns a predefined region to each dialogue to set geographical context.
3. **Scenario Generation**: Creates a unique scenario for each dialogue based on service category, region, and time slot.
4. **Emotion Assignment**: Randomly assigns emotions to user and assistant roles from predefined emotion lists.
5. **Dialogue Generation**: Utilizes OpenAI's API to generate dialogues incorporating the above elements.
6. **Uniqueness Verification**: Ensures each generated dialogue is unique using hashing and semantic similarity checks.
7. **Data Balancing**: Balances the distribution of dialogues across various categories, services, and regions.
8. **Bias Mitigation**: Applies strategies to identify and reduce biases in generated dialogues.
9. **Saving Output**: Saves generated dialogues in batches to the specified output file, along with updating embeddings and hashes.

### Testing Dialogue Extraction

To test the extraction and processing of a single dialogue from the dataset, use the `--test_extraction` flag along with the `--extraction_index` to specify which dialogue to extract.

**Example: Test Extraction of Dialogue at Index 10**

```bash
python dialogue_generation.py --test_extraction --extraction_index 10
```

**What It Does:**

- Extracts the dialogue at the specified index from the MultiWOZ v2.2 dataset.
- Processes and logs the base conversation extracted from the dialogue.
- Useful for verifying that the extraction and processing functions are working correctly.

## Output Files

Upon execution, the script generates and manages the following output files:

1. **Generated Dialogues**

   - **File**: `generated_dialogues.json` (default) or as specified by `--output_file`.
   - **Description**: Contains the list of generated dialogues with detailed information, including:
     - `services`: List of services involved in the dialogue.
     - `dialogue_id`: Unique identifier for each generated dialogue.
     - `turns`: List of dialogue turns, each containing:
       - `turn_number`: Sequential number of the turn.
       - `utterance`: User's message.
       - `intent`: Intent classification of the user's message.
       - `assistant_response`: Assistant's reply.
       - `emotion`: Emotion assigned to the speaker.
     - `num_lines`: Number of turns in the dialogue.
     - `user_emotions`: Emotions assigned to the user.
     - `assistant_emotions`: Emotions assigned to the assistant.
     - `scenario_category`: Category of the scenario.
     - `generated_scenario`: Detailed scenario description.
     - `time_slot`: Time slot during which the scenario occurs.
     - `regions`: Geographical regions associated with the dialogue.
     - `resolution_status`: Outcome of the dialogue (Resolved, Failed, Escalated).

2. **Dialogue Hashes**

   - **File**: `dialogue_hashes.json` (default) or as specified by `--hash_file`.
   - **Description**: Stores SHA-256 hashes of generated dialogues to ensure uniqueness and prevent duplication.

3. **Dialogue Embeddings**

   - **File**: `dialogue_embeddings.npy` (default) or as specified by `--embedding_file`.
   - **Description**: Contains numerical embeddings of dialogues used for semantic similarity checks, ensuring generated dialogues are contextually unique.

4. **Log Files**

   - **File**: `dialogue_generation.log`
   - **Description**: Captures detailed logs of the script's execution, including information, warnings, errors, and activities related to data balancing, bias mitigation, and moderation.

## Logging

The script employs a robust logging mechanism to facilitate monitoring and debugging:

- **Log File**: `dialogue_generation.log`
- **Log Levels**:
  - **INFO**: General information about the script's progress and actions.
  - **WARNING**: Non-critical issues that do not halt execution but may require attention.
  - **ERROR**: Critical issues that may prevent certain functionalities from executing correctly.

**Sample Log Entries:**

```
2024-04-27 12:00:00 [INFO] Loading spaCy model...
2024-04-27 12:00:05 [INFO] Loading dataset from local path: ./local_datasets/multi_woz_v22
2024-04-27 12:00:10 [INFO] Dataset loaded successfully.
2024-04-27 12:00:15 [WARNING] Duplicate dialogue detected for dialogue_id 'dialogue_123_generated_abc123'. Skipping.
2024-04-27 12:00:20 [ERROR] OpenAI API error during scenario generation for category 'refund_request' and service 'hotel': Rate limit exceeded.
2024-04-27 12:00:25 [INFO] Data balancing applied: Ensured uniform distribution across services and regions.
2024-04-27 12:00:30 [WARNING] Bias mitigation detected potential gender bias in generated dialogue_id 'dialogue_456_generated_def456'.
2024-04-27 12:00:35 [INFO] Omni moderation passed for dialogue_id 'dialogue_789_generated_ghi789'.
2024-04-27 12:00:40 [INFO] Text clustering analysis completed for batch 1.
```

**Accessing Logs:**

Logs are saved both to the `dialogue_generation.log` file and printed to the console in real-time, allowing for immediate monitoring during script execution.

## Project Structure

```
dialogue-generation/
├── dialogue_generation.py
├── requirements.txt
├── README.md
├── .env.local
├── models/
│   └── sentence_transformer/
├── local_datasets/
│   ├── multi_woz_v22/
│   └── FinePersonas-v0.1-clustering-100k/
├── generated_dialogues.json
├── dialogue_hashes.json
├── dialogue_embeddings.npy
├── dialogue_generation.log
├── distilabel/
│   └── ... (Text clustering and analysis scripts)
└── text-clustering/
    └── ... (Text clustering and analysis scripts)
```

- **dialogue_generation.py**: Main script for generating dialogues.
- **requirements.txt**: Lists all Python dependencies.
- **README.md**: Documentation for the project.
- **.env.local**: Contains environment variables (e.g., OpenAI API key).
- **models/**: Directory containing the SentenceTransformer model.
- **local_datasets/**: Directory containing required datasets.
  - **multi_woz_v22/**: MultiWOZ v2.2 dataset files.
  - **FinePersonas-v0.1-clustering-100k/**: FinePersonas dataset files.
- **generated_dialogues.json**: Output file with generated dialogues.
- **dialogue_hashes.json**: File storing hashes of generated dialogues.
- **dialogue_embeddings.npy**: NumPy file with dialogue embeddings.
- **dialogue_generation.log**: Log file capturing script execution details.
- **distilabel/**: Repository for text clustering and analysis using Distilabel techniques.
- **text-clustering/**: Repository for additional text clustering and analysis functionalities.

## Troubleshooting

- **Missing Environment Variables**

  - **Issue**: `OPENAI_API_KEY environment variable not set.`
  - **Solution**: Ensure that the `.env.local` file exists in the root directory and contains the correct OpenAI API key.

- **Missing spaCy Model**

  - **Issue**: `spaCy model not found.`
  - **Solution**: The script attempts to download the `en_core_web_sm` model automatically. If it fails, manually install it using:

    ```bash
    python -m spacy download en_core_web_sm
    ```

- **Dataset Loading Errors**

  - **Issue**: `Failed to load dataset from ./local_datasets/multi_woz_v22`
  - **Solution**: Verify that the MultiWOZ v2.2 dataset is correctly downloaded and placed in the specified directory. Ensure proper permissions and dataset integrity.

- **OpenAI API Errors**

  - **Issue**: `OpenAI API error during scenario generation...`
  - **Solution**: Check your OpenAI API key and ensure it has sufficient quota. Review rate limits and consider implementing exponential backoff or retry mechanisms.

- **Model Loading Failures**

  - **Issue**: `Failed to load SentenceTransformer model`
  - **Solution**: Ensure that the SentenceTransformer model is correctly downloaded and placed in `./models/sentence_transformer`. Verify compatibility and model integrity.

- **Duplicate Dialogues Skipped**

  - **Issue**: Generated dialogues are being skipped due to duplication.
  - **Solution**: Adjust the `--similarity_threshold` parameter to a lower value if necessary. Ensure that the initial datasets provide sufficient diversity.

- **Insufficient Embeddings**

  - **Issue**: No embeddings loaded, leading to all dialogues being considered unique.
  - **Solution**: Ensure that `dialogue_embeddings.npy` is correctly generated and updated during previous runs. If starting fresh, this is expected behavior.

- **Error During Dialogue Processing**

  - **Issue**: `Error processing generated dialogue for dialogue_id 'original_id': ...`
  - **Solution**: Review the log file for specific error messages. Ensure that the OpenAI API responses are correctly formatted and that regex patterns match the expected dialogue structure.

- **Out of Bounds Extraction Index**

  - **Issue**: `Index 100 is out of bounds for the dataset.`
  - **Solution**: Verify the size of the dataset and provide a valid `--extraction_index` within the range.

- **Data Balancer Not Functioning Correctly**

  - **Issue**: Dialogues are not evenly distributed across categories or services.
  - **Solution**: Ensure that the data balancer is correctly configured and that the dataset contains sufficient samples for each category and service. Review log entries related to data balancing for additional insights.

- **Bias Mitigation Alerts**

  - **Issue**: Warnings or errors related to detected biases in generated dialogues.
  - **Solution**: Review the bias mitigation logs to understand the nature of detected biases. Adjust generation parameters or implement additional bias mitigation strategies as necessary.

- **Omni Moderation Failures**

  - **Issue**: Dialogues failing moderation checks.
  - **Solution**: Review the moderation logs to identify and rectify content issues. Ensure that moderation rules are correctly defined and that the moderation module is functioning as intended.

## Contributing

Contributions are welcome! If you encounter issues or have suggestions for improvements, please open an issue or submit a pull request.

1. **Fork the Repository**

   ```bash
   git clone https://github.com/yourusername/dialogue-generation.git
   cd dialogue-generation
   git checkout -b feature/YourFeatureName
   ```

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your message here"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open a Pull Request**

   Navigate to the repository on GitHub and open a pull request for your feature branch.

**Guidelines:**

- Follow consistent coding standards and include meaningful commit messages.
- Ensure that new features are well-documented.
- Write tests for new functionalities where applicable.
- Update the README.md if necessary to reflect changes.
- Respect the project's code of conduct and contribution guidelines.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Disclaimer: Ensure compliance with OpenAI's usage policies and data privacy regulations when using this script. Always handle API keys securely and avoid exposing sensitive information.*