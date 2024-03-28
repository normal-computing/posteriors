import json
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("pg19", split="train", streaming=True)

num_subset = 20
earliest_year = 1900
min_text_length = 1e5
max_text_length = 1e6

# Create an iterator and a subset list
iterator = iter(dataset)
subset = []

# Collect num_subset entries from the dataset
num_tried = 0
while len(subset) < num_subset:
    datum = next(iterator)
    if (
        len(datum["text"]) > min_text_length
        and len(datum["text"]) < max_text_length
        and datum["publication_date"] > earliest_year
    ):
        subset.append(datum)
    num_tried += 1

# Save the subset to a JSON file
with open(f"./examples/continual_lora/data/pg19-{num_subset}.json", "w") as f:
    json.dump(subset, f)
