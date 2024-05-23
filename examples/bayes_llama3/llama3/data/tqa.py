import json
import os

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class TQADataLoader(pl.LightningDataModule):
    def __init__(self, data_config, num_workers=8, batch_size=10, shuffle=True):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.train_dataset = TQADataset(**data_config, split="train")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )


class TQADataset(Dataset):
    def __init__(
        self,
        folder,
        tokenizer_path,
        stride_length=300,
        stride_overlap=100,
        split="train",
    ):
        assert split in {"train", "val", "test"}

        self.stride_length = stride_length
        self.stride_overlap = stride_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.folder = os.path.join(folder, split)
        with open(os.path.join(self.folder, "tqa_v1_train.json")) as f:
            qa_doc = json.load(f)

        self.folder = os.path.join(folder, "val")
        with open(os.path.join(self.folder, "tqa_v1_val.json")) as f:
            qa_doc += json.load(f)

        self.folder = os.path.join(folder, "test")
        with open(os.path.join(self.folder, "tqa_v2_test.json")) as f:
            qa_doc += json.load(f)

        self.list_of_paragraphs = []
        for concept in qa_doc:
            for key in concept["topics"].keys():
                self.list_of_paragraphs.append(
                    concept["topics"][key]["content"]["text"]
                )

        self.tokenized_dataset = []
        for sample in self.list_of_paragraphs:
            sample = self.tokenize_and_stride(sample)
            batch_size = sample["input_ids"].size(0)
            for idx in range(batch_size):
                self.tokenized_dataset.append(
                    {
                        "input_ids": sample["input_ids"][idx],
                        "attention_mask": sample["attention_mask"][idx],
                    }
                )

    def tokenize_and_stride(self, sample):
        return self.tokenizer(
            sample,
            truncation=True,
            max_length=self.stride_length,
            stride=self.stride_overlap,
            padding="max_length",
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        return self.tokenized_dataset[idx]

    def __len__(self):
        return len(self.tokenized_dataset)
