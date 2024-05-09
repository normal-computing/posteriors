import datetime
import os

import torch
import regex as re
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from ml_collections.config_dict import FrozenConfigDict


@rank_zero_only
def create_log_dir(log_dir_name: str):
    """
    Create log directory, only once
    """
    if not os.path.exists(log_dir_name):
        os.makedirs(log_dir_name)


def setup_log_dir(log_dir_name: str, experiment_name: str = None) -> str:
    """
    Setup log directory
    """

    # Create timestamp folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_dir_name = os.path.join(log_dir_name, timestamp)

    # Add experiment name if specified
    if experiment_name is not None:
        log_dir_name += f"_{experiment_name}"

    create_log_dir(log_dir_name)
    return log_dir_name


@rank_zero_only
def save_config(conf: OmegaConf, fp: str):
    """
    Save config file, only once
    """
    OmegaConf.save(config=conf, f=fp)


def load_config(file: str) -> FrozenConfigDict:
    """
    Load config file
    """
    return FrozenConfigDict(OmegaConf.load(file))


def parse_devices(devices):
    devices = devices.split(",")
    devices_list = []
    for device in devices:
        try:
            device = int(device)
        except:
            pass
        devices_list.append(device)
    return devices_list


def load_ensemble(filepaths):
    def load_from_checkpoint(filepath):
        parameters = torch.load(filepath)["state_dict"]["bayesian_layer"].params
        parameters = {
            re.sub(r"model\.layers\.\d+\.", "", k): v
            for k, v in parameters.items()
            if v.numel() > 0
        }
        return parameters

    return [load_from_checkpoint(filepath) for filepath in filepaths]
