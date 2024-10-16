from datasets import load_from_disk
import random
import ast
from collections import defaultdict
from prettytable import PrettyTable

def safe_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

def get_summary_labels(label):
    labels = safe_eval(label)
    if isinstance(labels, list):
        return tuple(sorted(labels))
    elif isinstance(labels, str):
        return (label,)
    return tuple()

def generate_diverse_personas(n, dataset_path):
    # Load the local dataset
    dataset = load_from_disk(dataset_path)
    
    # Get the first split (usually 'train')
    first_split = list(dataset.keys())[0]
    data = dataset[first_split]

    # Create dictionaries to store personas by cluster and summary labels
    cluster_dict = defaultdict(list)
    summary_dict = defaultdict(list)

    # Populate the dictionaries
    for i, (cluster, summary) in enumerate(zip(data['cluster_label'], data['summary_label'])):
        cluster_dict[cluster].append(i)
        summary_labels = get_summary_labels(summary)
        summary_dict[summary_labels].append(i)

    selected_personas = []
    used_clusters = set()
    used_summaries = set()
    iteration_count = 0
    reset_threshold = random.randint(10, 70)

    while len(selected_personas) < n:
        # Reset after reaching the random threshold
        if iteration_count >= reset_threshold:
            used_clusters.clear()
            used_summaries.clear()
            iteration_count = 0
            reset_threshold = random.randint(10, 70)
            print(f"Reset occurred. New threshold: {reset_threshold}")  # For debugging

        # Randomly choose between cluster-based and summary-based selection
        if random.choice([True, False]):
            # Cluster-based selection
            available_clusters = set(cluster_dict.keys()) - used_clusters
            if not available_clusters:
                available_clusters = set(cluster_dict.keys())
            
            cluster = random.choice(list(available_clusters))
            index = random.choice(cluster_dict[cluster])
            used_clusters.add(cluster)
        else:
            # Summary-based selection
            available_summaries = set(summary_dict.keys()) - used_summaries
            if not available_summaries:
                available_summaries = set(summary_dict.keys())
            
            summary = random.choice(list(available_summaries))
            index = random.choice(summary_dict[summary])
            used_summaries.add(summary)

        selected_personas.append(index)
        iteration_count += 1

    # Create a pretty table
    table = PrettyTable()
    table.field_names = ["Index", "Persona", "Cluster Label", "Summary Label"]
    
    for index in selected_personas:
        table.add_row([
            index,
            data[index]['persona'],
            data[index]['cluster_label'],
            data[index]['summary_label']
        ])

    return table

# Usage
def main():
    dataset_path = './local_datasets/FinePersonas-v0.1-clustering-100k'
    n = int(input("Enter the number of personas to generate: "))
    result_table = generate_diverse_personas(n, dataset_path)
    print(result_table)

if __name__ == "__main__":
    main()