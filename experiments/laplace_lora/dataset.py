"""HF Dataset"""
import torch
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
from transformers import AutoTokenizer
from datasets import load_dataset

from experiments.base import TorchDataset


class HuggingfaceDataset(TorchDataset):
    """
    HF Dataset
    """

    def __init__(self, config: FrozenConfigDict):
        super().__init__(config)
        self.name = config.name

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_pretrained_model_name_or_path"]
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.datasets = self.config
        self.dataloaders = self.config

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples[self.config.inputs_key],
            padding="max_length",
            max_length=self.config.max_length,
            truncation=self.config.truncation,
        )

    @property
    def datasets(self):
        return self._datasets

    @datasets.setter
    def datasets(self, config: ConfigDict):
        dataset = load_dataset(self.name)

        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns([config.inputs_key])
        tokenized_datasets.set_format("torch")

        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["test"]

        if self.config.small:
            train_dataset = train_dataset.shuffle(seed=42).select(range(100))
            eval_dataset = eval_dataset.shuffle(seed=42).select(range(100))

        self.trainset = train_dataset
        self.testset = eval_dataset
        self._datasets = self.trainset, self.testset

    @property
    def dataloaders(self):
        return self._dataloaders

    @dataloaders.setter
    def dataloaders(self, config: ConfigDict):
        self.train_dataloader = torch.utils.data.DataLoader(
            self.trainset, shuffle=True, batch_size=self.config.batch_size
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.config.batch_size
        )

        self._dataloaders = self.train_dataloader, self.test_dataloader
