"""Base class for experiments"""

import os
import pickle
import pandas
import torch
from typing import Optional, Union
from abc import ABC, abstractmethod
from ml_collections.config_dict import ConfigDict, FrozenConfigDict


class Dataset(ABC):
    """
    Base class for all datasets
    """

    def __init__(self, config: FrozenConfigDict):
        self.config = ConfigDict(config)  # thaw config

        self.name = config.name
        self.bsz = config.batch_size

    @property
    @abstractmethod
    def datasets(
        self,
    ) -> tuple[Union[torch.utils.data.Dataset, pandas.DataFrame]]:
        """
        Store datasets, return train, test.
        Can be torch dataset, pandas dataframe, etc.
        """

    @datasets.setter
    @abstractmethod
    def datasets(
        self,
    ):
        """
        Set datasets, return train, test
        """


class TorchDataset(Dataset):
    """
    Base class for torch datasets
    """

    def __init__(self, config: FrozenConfigDict):
        super().__init__(config)
        self.cache_dir = config.cache_dir
        self.num_workers = config.num_workers

    @property
    @abstractmethod
    def dataloaders(
        self,
    ) -> tuple[torch.utils.data.DataLoader]:
        """
        Store torch dataloaders, return train, test
        """

    @dataloaders.setter
    @abstractmethod
    def dataloaders(
        self,
    ):
        """
        Set torch dataloaders, return train, test
        """


class Experiment(ABC):
    """
    Base class for experiments
    """

    def __init__(self, config: FrozenConfigDict):
        self.config = ConfigDict(config)  # thaw config
        self.experiment_log_dir = config.experiment_log_dir

    @property
    @abstractmethod
    def train_metrics(
        self,
    ) -> dict:
        """
        Define train metrics
        """

    @train_metrics.setter
    @abstractmethod
    def train_metrics(
        self,
    ):
        """
        Set train metrics
        """

    @property
    @abstractmethod
    def test_metrics(
        self,
    ) -> dict:
        """
        Define test metrics
        """

    @test_metrics.setter
    @abstractmethod
    def test_metrics(
        self,
    ):
        """
        Set test metrics
        """

    @abstractmethod
    def train(self, dataset: Dataset) -> dict:
        """
        Train model, return dictionary of train metrics.
        """

    @abstractmethod
    def test(self, dataset: Dataset) -> dict:
        """
        Test model, return dictionary of test metrics.
        """

    @abstractmethod
    def run_experiment(self, dataset: Dataset, resume: bool = None):
        """
        Run experiment pipeline
        """

    def save_results(
        self, results, metadata: str = None, checkpoint: Optional[int] = None
    ):  # To do: make more specific than pickle
        """
        Save results as pickle file
        """
        folder = "final" if not checkpoint else "checkpoints"
        experiment_dir = os.path.join(self.experiment_log_dir, folder)
        os.makedirs(experiment_dir, exist_ok=True)
        result_file = os.path.join(
            experiment_dir, f"results-{metadata}.pkl" if metadata else "results.pkl"
        )
        with open(result_file, "wb") as f:
            pickle.dump(results, f)

    def run(self, dataset: Dataset, resume: bool = None, **kwargs):
        """
        Run experiment and save results
        """
        results = self.run_experiment(dataset=dataset, resume=resume, **kwargs)
        self.save_results(results)
