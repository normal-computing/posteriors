from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class TQADataLoader(pl.LightningDataModule):
    def __init__(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


class TQADataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self):
        pass

    def __len__(self):
        pass
