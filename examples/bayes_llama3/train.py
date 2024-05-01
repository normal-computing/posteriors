import argparse
import os

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
import pytorch_lightning as pl
import numpy as np
import torch

from llama3.model import BayesLlama
from llama3.data.tqa import TQADataLoader
from llama3.utils.load_utils import parse_devices

# from llama3.utils.logger import Logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, required=True)
parser.add_argument("--resume", default=None, type=str)
parser.add_argument("--name", default=None, type=str)
parser.add_argument("--devices", default=parse_devices, type=str)
parser.add_argument("-d", "--debug", default=False, action="store_true")
args = parser.parse_args()

assert args.base is not None, "Configs not specified, specify at least resume or base"
config = OmegaConf.load(args.base)

torch.set_float32_matmul_precision("medium")
torch.manual_seed(int(config["seed"]))
np.random.seed(int(config["seed"]))

if __name__ == "__main__":
    device_type = "cpu" if callable(args.devices) else "gpu"
    trainer_kwargs = {
        "max_epochs": config["epochs"],
        "accelerator": device_type,
        "log_every_n_steps": config["metrics_log_frequency"],
        "num_sanity_val_steps": 0,
    }
    if not callable(args.devices):
        trainer_kwargs["devices"] = args.devices

    name = "lightning_logs" if args.name is None else args.name
    version = None

    if not args.debug:
        logger = TensorBoardLogger("", name=name, version=version)
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{logger.log_dir}/checkpoints",
            filename="{epoch}",
            save_weights_only=False,
            verbose=True,
            every_n_train_steps=config["save_frequency"],
        )
        trainer_kwargs["logger"] = logger
        trainer_kwargs["callbacks"] = [checkpoint_callback]

    trainer = pl.Trainer(**trainer_kwargs)
    dataloader = TQADataLoader(config["data_dir"])
    model = BayesLlama(
        len(dataloader.train_dataset),
        lr=config["learning_rate"],
    )

    try:
        trainer.fit(model, dataloader)
    except KeyboardInterrupt as e:
        if trainer.global_rank == 0:
            if not args.debug:
                epoch_num = trainer.current_epoch
                final_ckpt = os.path.join(
                    logger.log_dir, "checkpoints", f"embedding_{epoch_num}.ckpt"
                )
                trainer.save_checkpoint(final_ckpt)
