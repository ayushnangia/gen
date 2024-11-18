import json
import random

# Read the original JSON file
with open('multiwoz_v22_data.json', 'r') as file:
    data = json.load(file)

# Select 20 random entries
random_entries = random.sample(data, 20)

# Write the random entries to a new JSON file
with open('random_20_entries.json', 'w') as file:
    json.dump(random_entries, file, indent=4)

print("20 random entries have been saved to 'random_20_entries.json'")