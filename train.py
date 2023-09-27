from sghmc.dataloader import ClincOOSDataLoader
from sghmc.model import SGHMCModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


log_dir_name = "logs/train-1"


logger = CSVLogger(name="csvlogger", save_dir=log_dir_name, version="")
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


trainer_kwargs = {
    "max_epochs": 100,
    "gradient_clip_val": 0.5,
    "accumulate_grad_batches": 2,
    "accelerator": "gpu",
    "devices": [0, 1],
    "strategy": "ddp_find_unused_parameters_false",
    "num_sanity_val_steps": 0,
}


data = ClincOOSDataLoader()
model = SGHMCModel()

trainer = pl.Trainer(**trainer_kwargs)
trainer.fit(model, data)
