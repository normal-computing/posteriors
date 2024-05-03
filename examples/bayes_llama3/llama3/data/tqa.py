import json
import os

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class TQADataLoader(pl.LightningDataModule):
    def __init__(self, folder, num_workers=8, batch_size=10, shuffle=True):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.train_dataset = TQADataset(folder, split="train")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )


class TQADataset(Dataset):
    def __init__(self, folder, split="train"):
        assert split in {"train", "val", "test"}

        self.folder = os.path.join(folder, split)
        with open(os.path.join(self.folder, f"tqa_v1_{split}.json")) as f:
            qa_doc = json.load(f)

        self.list_of_questions = []
        for concept in qa_doc:
            for key in concept["topics"].keys():
                self.list_of_questions.append(concept["topics"][key]["content"]["text"])

    def __getitem__(self, idx):
        return self.list_of_questions[idx]

    def __len__(self):
        return len(self.list_of_questions)
