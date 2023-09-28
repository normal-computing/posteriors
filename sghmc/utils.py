from pytorch_lightning.utilities import rank_zero_only
from omegaconf import OmegaConf
import os


# REQUIRED_PARAMS = ["model", "data"]


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


# def load_config(file):
#     config = OmegaConf.load(file)
#     for param in REQUIRED_PARAMS:
#         assert param in config, f"Missing key {param} in config"
#     return config


# @rank_zero_only
# def save_config(conf, fp):
#     OmegaConf.save(config=conf, f=fp)


@rank_zero_only
def create_log_dir(log_dir_name):
    if not os.path.exists(log_dir_name):
        os.mkdir(log_dir_name)


def setup_log_dir(
    log_dir_name, timestamp, experiment_name=None, debug=False, resume=None
):
    if resume is not None and not debug:
        return resume

    # Create parent log name
    if not os.path.exists(log_dir_name):
        os.mkdir(log_dir_name)

    # Add to debug folder if specified
    if debug:
        log_dir_name = os.path.join(log_dir_name, "debug")
        create_log_dir(log_dir_name)

    # Create timestamp folder
    log_dir_name = os.path.join(log_dir_name, timestamp)

    # Add experiment name if specified
    if experiment_name is not None:
        log_dir_name += f"_{experiment_name}"

    create_log_dir(log_dir_name)

    # Create checkpoints folder
    create_log_dir(f"{log_dir_name}/checkpoints")

    return log_dir_name
