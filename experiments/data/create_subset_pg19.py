import json
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("pg19", split="test", streaming=True)

num_subset = 5

# Create an iterator and a subset list
iterator = iter(dataset)
subset = []


# Collect num_subset entries from the dataset
for i in range(num_subset):
    subset.append(next(iterator))


# Save the subset to a JSON file
with open("./experiments/data/pg19-even-smaller.json", "w") as f:
    json.dump(subset, f)
