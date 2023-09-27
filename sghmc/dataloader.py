from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch

import pickle


DATA_SPLIT = 0.8


class ClincOOSDataLoader(pl.LightningDataModule):
    def __init__(self, num_workers=8, batch_size=10, shuffle=True):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.train_set = ClincOOSDataset("train")
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
    def __init__(self, split="train"):
        assert split in ["train", "validation"]

        with open("oos_hidden_states.pkl", "rb") as f:
            data = pickle.load(f)

        with open("oos_test_intents.pkl", "rb") as f:
            labels = pickle.load(f)
        with open("oos_oos_intents.pkl", "rb") as f:
            labels += pickle.load(f)

        self.labels = [l.numpy() for l in labels]

        data_len = len(data)
        split_size = int(data_len * 0.8)
        if split == "train":
            self.data = data[:split_size]
        else:
            self.data = data[split_size:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.from_numpy(self.labels[idx])


if __name__ == "__main__":
    ClincOOSDataset()
