import os
from typing import List
import torch
from torch import nn
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from ml_collections.config_dict import FrozenConfigDict


def parse_devices(devices):
    devices = devices.split(",")
    devices_list = []
    for device in devices:
        try:
            device = int(device)
        except ValueError:
            pass
        devices_list.append(device)
    return devices_list


def load_optimizer_param_to_model(model: nn.Module, groups: List[List[torch.Tensor]]):
    """Updates the model parameters in-place with the provided grouped parameters.

    Args:
        model: A torch.nn.Module object
        groups: A list of groups where each group is a list of parameters
    """

    optimizer_params = []
    for group in groups:
        for param in group:
            optimizer_params.append(torch.from_numpy(param))

    for model_param, optimizer_param in zip(list(model.parameters()), optimizer_params):
        model_param.data = optimizer_param


REQUIRED_PARAMS = ["dataset_config", "experiment_config"]


def load_config(file: str) -> FrozenConfigDict:
    """
    Load config file
    """
    config = OmegaConf.load(file)
    for param in REQUIRED_PARAMS:
        assert param in config, f"Missing key {param} in config"

    config = FrozenConfigDict(config)
    return config


@rank_zero_only
def save_config(conf: OmegaConf, fp: str):
    """
    Save config file, only once
    """
    OmegaConf.save(config=conf, f=fp)


@rank_zero_only
def create_log_dir(log_dir_name: str):
    """
    Create log directory, only once
    """
    if not os.path.exists(log_dir_name):
        os.mkdir(log_dir_name)


def setup_log_dir(
    log_dir_name: str,
    timestamp: str,
    resume: bool = False,
    experiment_name: str = None,
) -> str:
    """
    Setup log directory
    """
    if resume:
        return resume

    # Create parent log name
    if not os.path.exists(log_dir_name):
        os.mkdir(log_dir_name)

    # Create timestamp folder
    log_dir_name = os.path.join(log_dir_name, timestamp)

    # Add experiment name if specified
    if experiment_name is not None:
        log_dir_name += f"_{experiment_name}"

    create_log_dir(log_dir_name)

    # Create checkpoints folder
    create_log_dir(f"{log_dir_name}/checkpoints")

    return log_dir_name
