import torch
import regex as re
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

@rank_zero_only
def save_config(conf: OmegaConf, fp: str):
    """
    Save config file, only once
    """
    OmegaConf.save(config=conf, f=fp)


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
