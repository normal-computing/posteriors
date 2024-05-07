import torch
import regex as re
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only
import os
from ml_collections.config_dict import FrozenConfigDict
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

REQUIRED_PARAMS = ["experiment_config", "bayes_config"]


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
    config = OmegaConf.load(file)
    for param in REQUIRED_PARAMS:
        assert param in config, f"Missing key {param} in config"

    config = FrozenConfigDict(config)
    return config


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

    return [
        load_from_checkpoint(filepath) for filepath in filepaths
    ]
