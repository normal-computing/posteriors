import torch
import pickle
import os
from transformers import AutoTokenizer

from experiments.utils import load_config
from experiments.laplace_lora import BayesTransformerModule

config = load_config("experiments/utils/configs/multiple_lora.yaml")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


with open(config.dataset_path, "rb") as f:
    dataset = pickle.load(f)


tokenizer = AutoTokenizer.from_pretrained(
    config.model_config.pretrained_model_name_or_path
)

tokenizer.pad_token = tokenizer.eos_token


with open(config.dataset_path, "rb") as f:
    dataset = pickle.load(f)


def split_doc(lst):
    start_index = int(len(lst) * 0.85)
    return lst[:start_index], lst[start_index:]


for sample in dataset:
    sample["input_ids"] = tokenizer(
        sample["text"],
        padding="max_length",
        max_length=config.max_length,
        truncation=True,
    )["input_ids"]


samples_per_task = [
    dataset[(i * config.num_samples_per_task) : ((i + 1) * config.num_samples_per_task)]
    for i in range(config.num_tasks)
]

split_samples = [
    [split_doc(sample["input_ids"]) for sample in samples]
    for samples in samples_per_task
]


train_datasets = [[sample[0] for sample in samples] for samples in split_samples]
test_datasets = [[sample[1] for sample in samples] for samples in split_samples]

train_dataloaders = [
    torch.utils.data.DataLoader(
        train,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    for train in train_datasets
]

batch = next(iter(train_dataloaders[0]))

model = BayesTransformerModule(config.model_config)
