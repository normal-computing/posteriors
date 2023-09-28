import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from sghmc.utils import parse_devices, setup_log_dir
from sghmc.dataloader import ClincOOSDataLoader
from sghmc.modules.transformer import MiniTransformer
from sghmc.model import SGHMCModel

import argparse
import datetime
import os


parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str)
parser.add_argument("--resume", default=None, type=str)
parser.add_argument("--devices", default=parse_devices, type=str)
parser.add_argument("-d", "--debug", default=False, action="store_true")
args = parser.parse_args()

timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
log_dir_name = setup_log_dir(
    "logs",
    timestamp,
    debug=args.debug,
    # resume=args.resume,
)

lr = 1e-1

if __name__ == "__main__":
    trainer_kwargs = {
        "max_epochs": 100,
        "accelerator": "cpu" if callable(args.devices) else "gpu",
        "strategy": "auto",
        "num_sanity_val_steps": 0,
        "num_processes": 1,
    }

    if not callable(args.devices):
        trainer_kwargs["devices"] = args.devices

    # logger = CSVLogger(name="csvlogger", save_dir=log_dir_name, version="")
    callbacks = [
        TQDMProgressBar(refresh_rate=1),
        ModelCheckpoint(
            dirpath=f"{log_dir_name}/checkpoints/trainstep_checkpoints",
            filename="{epoch:06}-{step:09}",
            every_n_train_steps=10,
            save_last=True,
            verbose=True,
            save_weights_only=True,
        ),
        ModelCheckpoint(
            dirpath=f"{log_dir_name}/checkpoints",
            filename="{epoch:06}",
            verbose=True,
            save_last=True,
            save_on_train_epoch_end=True,
            save_weights_only=False,
        ),
    ]
    trainer_kwargs["callbacks"] = callbacks

    dataset = ClincOOSDataLoader("data", batch_size=1000, shuffle=True, num_workers=8)
    model = SGHMCModel(MiniTransformer(), lr=lr)
    trainer = pl.Trainer(**trainer_kwargs)

    try:
        # resume_ckpt = None
        # if args.resume is not None:
        #     resume_ckpt = os.path.join(args.resume, "checkpoints", "last.ckpt")
        trainer.fit(model, dataset, ckpt_path=None)
    finally:
        if trainer.global_rank == 0:
            final_ckpt = os.path.join(log_dir_name, "checkpoints", "last.ckpt")
            trainer.save_checkpoint(final_ckpt)
