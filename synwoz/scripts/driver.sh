#!/bin/bash

# Bash Driver Script for Dialogue Generation
# This script prompts the user for the total number of dialogues to generate (e.g., 5000).
# It then executes the Python dialogue_generation.py script in batches (e.g., 1000 at a time),
# randomly selecting hyperparameters from the specified ranges for each batch.

# Configuration
PYTHON_SCRIPT="parallel_gen.py"  # Replace with the correct path if different
BATCH_SIZE=200
# Hyperparameter ranges
temperature_options=(0.6 0.7 0.8)
top_p_options=( 0.95 1.0)
frequency_penalty_options=(0.4 0.5 0.6 0.7)
presence_penalty_options=(0.4 0.5 0.6 0.7)

# Function to select a random element from an array
get_random_element() {
    local array=("$@")
    local index=$(( RANDOM % ${#array[@]} ))
    echo "${array[$index]}"
}

# Prompt user for total number of generations
read -p "Enter the total number of dialogues to generate (e.g., 5000): " TOTAL_GENERATIONS

# Calculate the number of full batches and the remaining dialogues
FULL_BATCHES=$(( TOTAL_GENERATIONS / BATCH_SIZE ))
REMAINDER=$(( TOTAL_GENERATIONS % BATCH_SIZE ))

echo "Total Dialogues to Generate: $TOTAL_GENERATIONS"
echo "Batch Size: $BATCH_SIZE"
echo "Full Batches: $FULL_BATCHES"
echo "Remaining Dialogues in Last Batch: $REMAINDER"
echo "---------------------------------------------"

# Function to execute a single batch
execute_batch() {
    local batch_num=$1
    local batch_size=$2

    # Randomly select hyperparameters
    temperature=$(get_random_element "${temperature_options[@]}")
    top_p=$(get_random_element "${top_p_options[@]}")
    frequency_penalty=$(get_random_element "${frequency_penalty_options[@]}")
    presence_penalty=$(get_random_element "${presence_penalty_options[@]}")

    echo "Executing Batch $batch_num:"
    echo "  Dialogues: $batch_size"
    echo "  Temperature: $temperature"
    echo "  Top-p: $top_p"
    echo "  Frequency Penalty: $frequency_penalty"
    echo "  Presence Penalty: $presence_penalty"

    # Execute the Python script with the selected hyperparameters
    python3 "$PYTHON_SCRIPT" \
        --total_generations "$batch_size" \
        --temperature "$temperature" \
        --top_p "$top_p" \
        --frequency_penalty "$frequency_penalty" \
        --presence_penalty "$presence_penalty"

    if [ $? -ne 0 ]; then
        echo "Error: Batch $batch_num failed to execute."
        exit 1
    fi

    echo "Batch $batch_num completed successfully."
    echo "---------------------------------------------"
}

# Execute full batches
for (( i=1; i<=FULL_BATCHES; i++ ))
do
    execute_batch "$i" "$BATCH_SIZE"
done

# Execute the remaining dialogues if any
if [ "$REMAINDER" -gt 0 ]; then
    batch_num=$(( FULL_BATCHES + 1 ))
    execute_batch "$batch_num" "$REMAINDER"
fi

echo "All batches completed successfully. Total dialogues generated: $TOTAL_GENERATIONS."