from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch

import pickle
import os


DATA_SPLIT = 0.8
NUM_CLASSES = 151


class ClincOOSDataLoader(pl.LightningDataModule):
    def __init__(self, path, num_workers=8, batch_size=10, shuffle=True):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.train_set = ClincOOSDataset(path, "train")
        # self.validation_set = CollectiveDataset(datasets, "validation")
        # self.test_set = CollectiveDataset(datasets, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )


class ClincOOSDataset(Dataset):
    def __init__(self, path, split="train"):
        assert split in ["train", "validation"]

        with open(os.path.join(path, "oos_hidden_states.pkl"), "rb") as f:
            data = pickle.load(f)

        with open(os.path.join(path, "oos_test_intents.pkl"), "rb") as f:
            labels = pickle.load(f).tolist()
        with open(os.path.join(path, "oos_oos_intents.pkl"), "rb") as f:
            labels += pickle.load(f).tolist()

        assert len(labels) == len(data)

        data_len = len(data)
        split_size = int(data_len * 0.8)
        if split == "train":
            self.data = data[:split_size]
            self.labels = labels[:split_size]
        else:
            self.data = data[split_size:]
            self.labels = labels[split_size:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # labels = torch.tensor(self.labels[idx : idx + 1])
        # labels = F.one_hot(labels, NUM_CLASSES).float().squeeze(0)
        labels = torch.tensor(self.labels[idx : idx + 1])
        return self.data[idx][-1], labels


if __name__ == "__main__":
    ClincOOSDataset()
