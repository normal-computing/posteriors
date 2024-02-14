import json
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("pg19", split="test", streaming=True)

# Create an iterator and a subset list
iterator = iter(dataset)
subset = []

# Collect 50 entries from the dataset
for i in range(50):
    subset.append(next(iterator))

# Save the subset to a JSON file
with open("./experiments/data/pg19-small.json", "w") as f:
    json.dump(subset, f)
