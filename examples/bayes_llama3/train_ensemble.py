import argparse
import pathlib
import sys
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


os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, required=True)
parser.add_argument("--resume", default=None, type=str)
parser.add_argument("--devices", default=parse_devices, type=str)
parser.add_argument("-d", "--debug", default=False, action="store_true")
args = parser.parse_args()

assert args.base is not None, "Configs not specified, specify at least resume or base"
config = OmegaConf.load(args.base)

torch.set_float32_matmul_precision("medium")

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

    version = pathlib.Path(args.base).stem

    for idx in range(config["num_ensembles"]):
        torch.manual_seed(int(config["seed"]) + idx)
        np.random.seed(int(config["seed"]) + idx)

        logger = TensorBoardLogger("", name=f"logs/{version}", version=idx)
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{logger.log_dir}/checkpoints",
            filename="{epoch}-{step}",
            save_weights_only=True,
            verbose=True,
            save_top_k=-1,
            every_n_train_steps=config["save_frequency"],
        )
        trainer_kwargs["logger"] = logger
        trainer_kwargs["callbacks"] = [checkpoint_callback]

        trainer = pl.Trainer(**trainer_kwargs)
        dataloader = TQADataLoader(
            config["data_loader"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
        model = BayesLlama(
            len(dataloader.train_dataset),
            pretrained_weights_folder=config["pretrained_weights_folder"],
            lr=config["learning_rate"],
            alpha=config["alpha"],
            beta=config["beta"],
            momenta=config["momenta"],
            set_temperature=config["set_temperature"],
            max_seq_len=config["data_loader"]["stride_length"],
        )

        try:
            trainer.fit(model, dataloader)
        except KeyboardInterrupt as e:
            if trainer.global_rank == 0:
                if not args.debug:
                    epoch_num = trainer.current_epoch
                    final_ckpt = os.path.join(
                        logger.log_dir, "checkpoints", f"{epoch_num}.ckpt"
                    )
                    trainer.save_checkpoint(final_ckpt)
                    sys.exit(0)
