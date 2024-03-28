from typing import Tuple
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset


class DictDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def load_pg19_dataloaders(
    config,
    tokenizer,
    batch_size,
) -> Tuple[DataLoader, DataLoader]:
    """
    Tokenizes before splitting into train and test sets
    Support for dropping last batch and shuffling (both prior to splitting)"""
    dataset = load_dataset("json", data_files=config.dataset_path)["train"]
    dataset = dataset.select(range(config.num_tasks))

    def tokenize_and_stride(example):
        example_text = example["text"]
        tokenized_text = tokenizer(
            example_text,
            truncation=True,
            max_length=config.stride_length,
            stride=config.stride_overlap,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized_text["input_ids"],
            "attention_mask": tokenized_text["attention_mask"],
        }

    tokenized_datasets = dataset.map(tokenize_and_stride, batched=False)
    tokenized_datasets = tokenized_datasets.remove_columns(
        [
            "text",
            "url",
            "short_book_title",
            "publication_date",
        ]
    )
    tokenized_datasets.set_format("torch")

    if config.drop_last:
        tokenized_datasets = tokenized_datasets.map(
            lambda x: {
                "input_ids": x["input_ids"][:-1],
                "attention_mask": x["attention_mask"][:-1],
            }
        )

    # tokenized_datasets = tokenized_datasets.filter(
    #     lambda x: len(x["input_ids"]) > 0 and len(x["attention_mask"]) > 0
    # )

    if config.shuffle:
        tokenized_datasets = tokenized_datasets.map(
            lambda x: {
                "input_ids": x["input_ids"][torch.randperm(x["input_ids"].size(0))],
                "attention_mask": x["attention_mask"][
                    torch.randperm(len(x["attention_mask"]))
                ],
            }
        )

    train_datasets = []
    test_datasets = []
    for ds in tokenized_datasets:
        train_len = int(len(ds["input_ids"]) * config.train_proportion)
        train_datasets.append({k: v[:train_len] for k, v in ds.items()})
        test_datasets.append({k: v[train_len:] for k, v in ds.items()})

    train_dataloaders = [
        DataLoader(
            DictDataset(train_ds),
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        for train_ds in train_datasets
    ]

    test_dataloaders = [
        DataLoader(
            DictDataset(test_ds),
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        for test_ds in test_datasets
    ]
    return train_dataloaders, test_dataloaders
