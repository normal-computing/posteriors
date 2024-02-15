from typing import Tuple
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class DictDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def load_pg19_dataloaders(
    config, tokenizer, batch_size
) -> Tuple[DataLoader, DataLoader]:
    dataset = load_dataset("json", data_files=config.dataset_path)["train"]

    def split_doc(examples):
        text_as_list = examples["text"].split()
        start_index = int(len(text_as_list) * 0.85)
        return {
            "train_text": " ".join(text_as_list[:start_index]),
            "test_text": " ".join(text_as_list[start_index:]),
        }

    def tokenize_and_stride(example_text):
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

    def tokenize_function(examples):
        train_input_ids_and_mask = tokenize_and_stride(examples["train_text"])
        test_input_ids_and_mask = tokenize_and_stride(examples["test_text"])
        return {
            "train_input_ids": train_input_ids_and_mask["input_ids"],
            "train_attention_mask": train_input_ids_and_mask["attention_mask"],
            "test_input_ids": test_input_ids_and_mask["input_ids"],
            "test_attention_mask": test_input_ids_and_mask["attention_mask"],
        }

    split_datasets = dataset.map(split_doc, batched=False)
    tokenized_datasets = split_datasets.map(tokenize_function, batched=False)
    tokenized_datasets = tokenized_datasets.remove_columns(
        [
            "text",
            "train_text",
            "test_text",
            "url",
            "short_book_title",
            "publication_date",
        ]
    )
    tokenized_datasets.set_format("torch")

    train_datasets = [
        tokenized_datasets.remove_columns(["test_input_ids", "test_attention_mask"])
        .rename_column("train_input_ids", "input_ids")
        .rename_column("train_attention_mask", "attention_mask")[i]
        for i in range(config.num_tasks)
    ]
    test_datasets = [
        tokenized_datasets.remove_columns(["train_input_ids", "train_attention_mask"])
        .rename_column("test_input_ids", "input_ids")
        .rename_column("test_attention_mask", "attention_mask")[i]
        for i in range(config.num_tasks)
    ]

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
