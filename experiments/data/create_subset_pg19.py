from datasets import load_dataset
import pickle
import pandas as pd
from transformers import AutoTokenizer


dataset = load_dataset("pg19", split="test", streaming=True)
iterator = iter(dataset)
subset = []
for i in range(50):
    subset.append(next(iterator))

# with open("./experiments/data/pg19-small.pkl", "wb") as f:
#     pickle.dump(subset, f)


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token


d = pd.DataFrame(subset)


def split_doc(lst):
    start_index = int(len(lst) * 0.85)
    return lst[:start_index], lst[start_index:]


d["train_text"] = d["text"].apply(lambda x: split_doc(x)[0])
d["test_text"] = d["text"].apply(lambda x: split_doc(x)[1])

d["train_tokens"] = d["train_text"].apply(
    lambda x: tokenizer(
        x,
        padding="max_length",
        max_length=4096,
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)
)

d["test_tokens"] = d["test_text"].apply(
    lambda x: tokenizer(
        x,
        padding="max_length",
        max_length=4096,
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)
)

num_samples_per_task = 2
num_tasks = 3

train_datasets = [
    d.drop(columns=["test_tokens", "test_text"])
    .rename(columns={"train_tokens": "input_ids"})
    .iloc[(i * num_samples_per_task) : ((i + 1) * num_samples_per_task)]
    for i in range(num_tasks)
]
test_datasets = [
    d.drop(columns=["train_tokens", "train_text"])
    .rename(columns={"test_tokens": "input_ids"})
    .iloc[: ((i + 1) * num_samples_per_task)]
    for i in range(num_tasks)
]

with open("./experiments/data/pg19-train.pkl", "wb") as f:
    pickle.dump(train_datasets, f)

with open("./experiments/data/pg19-test.pkl", "wb") as f:
    pickle.dump(test_datasets, f)
