import os
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from ml_collections.config_dict import FrozenConfigDict


REQUIRED_PARAMS = ["model_config", "experiment_name"]


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
