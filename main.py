from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from sghmc.utils import parse_devices
from sghmc.dataloader import ClincOOSDataLoader
from sghmc.modules.transformer import MiniTransformer
from sghmc.model import SGHMCModel

import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--name", default=None, type=str)
parser.add_argument("--resume", default=None, type=str)
parser.add_argument("--devices", default=parse_devices, type=str)
parser.add_argument("-d", "--debug", default=False, action="store_true")
parser.add_argument("-lr", "--learning_rate", default=1e-1, type=float)
parser.add_argument("--alpha", default=1e-7, type=float)
parser.add_argument("--beta", default=0.0, type=float)
parser.add_argument("--log_frequency", default=10, type=int)
parser.add_argument("--epochs", default=100, type=int)
args = parser.parse_args()

if __name__ == "__main__":
    device_type = "cpu" if callable(args.devices) else "gpu"
    strategy = "ddp_find_unused_parameters_false" if device_type == "gpu" else "auto"
    trainer_kwargs = {
        "max_epochs": args.epochs,
        "accelerator": device_type,
        "strategy": strategy,
        "log_every_n_steps": args.log_frequency,
        "num_sanity_val_steps": 0,
    }

    if not callable(args.devices):
        trainer_kwargs["devices"] = args.devices

    name = args.name
    if args.resume:
        name = args.resume.rstrip("/").split("/")[-1]
    logger = TensorBoardLogger("", version=name)
    dataset = ClincOOSDataLoader("data", batch_size=1000, shuffle=True, num_workers=8)
    model = SGHMCModel(
        MiniTransformer(), lr=args.learning_rate, alpha=args.alpha, beta=args.beta
    )

    trainer_kwargs["logger"] = logger
    trainer = pl.Trainer(**trainer_kwargs)

    try:
        resume_ckpt = None
        if args.resume is not None:
            resume_ckpt = os.path.join(args.resume, "checkpoints", "last.ckpt")
        trainer.fit(model, dataset, ckpt_path=resume_ckpt)
    finally:
        if trainer.global_rank == 0:
            final_ckpt = os.path.join(logger.log_dir, "checkpoints", "last.ckpt")
            trainer.save_checkpoint(final_ckpt)
