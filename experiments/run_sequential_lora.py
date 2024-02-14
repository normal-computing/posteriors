import argparse
import os
import glob
import datetime
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer
from ml_collections.config_dict import ConfigDict
import pickle
from torch.utils.data import Dataset, DataLoader


from experiments.utils import parse_devices, load_config, save_config, setup_log_dir
from experiments.laplace_lora import BayesTransformerModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--name", default=None, type=str)
parser.add_argument("--resume", default=None, type=str)
parser.add_argument("--base", default=None, type=str)
parser.add_argument("--devices", default=parse_devices, type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--log_frequency", default=10, type=int)
parser.add_argument("--seed", default=42, type=int)

args = parser.parse_args()

if __name__ == "__main__":
    device_type = "cpu" if callable(args.devices) else "gpu"
    if args.resume is None:
        assert (
            args.base is not None
        ), "Configs not specified, specify at least resume or base"
        config = load_config(args.base)
    else:
        assert os.path.exists(
            args.resume
        ), "Provided path to resume training does not exist"
        config_paths = glob.glob(os.path.join(args.resume, "*.yaml"))
        assert len(config_paths) == 1, "Too many possible configs to resume from"
        config = load_config(config_paths[0])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    experiment_name = config.get("experiment_name", None)

    experiment_log_dir = setup_log_dir(
        config.get("logs_dir", "logs"),
        timestamp,
        resume=args.resume,
        experiment_name=experiment_name,
    )

    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(args.seed)

    trainer_kwargs = {
        "max_epochs": args.epochs,
        "accelerator": device_type,
        "log_every_n_steps": args.log_frequency,
    }
    model = BayesTransformerModule(config.model_config)

    config = ConfigDict(config)  # thaw
    logger = WandbLogger(
        log_model="all",
        project=config.get("experiment_name", ""),
        save_dir=config.get("logs_dir", "logs"),
    )
    config["wandb_name"] = logger.experiment.name
    config["wandb_id"] = logger.experiment.id

    config["epochs"] = args.epochs
    config["log_frequency"] = args.log_frequency
    config["seed"] = args.seed

    if args.resume is None:
        save_config(
            config.to_dict(), f"{experiment_log_dir}/{os.path.basename(args.base)}"
        )

    trainer = Trainer(**trainer_kwargs, logger=logger)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_config.pretrained_model_name_or_path
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    # train[test]_dataloaders has size (num_tasks, num_samples_per_task)
    # This will be updated in the future
    num_samples_per_task = 2
    num_tasks = 3
    with open("./experiments/data/pg19-train.pkl", "rb") as f:
        train_datasets = pickle.load(f)
    with open("./experiments/data/pg19-test.pkl", "rb") as f:
        test_datasets = pickle.load(f)

    # Annoying ... will try to get around
    class DataFrameDataset(Dataset):
        def __init__(self, dataframe, transform=None):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            sample = {"input_ids": row["input_ids"]}
            if self.transform:
                sample = self.transform(sample)
            return sample

    train_dataloaders = [
        DataLoader(
            DataFrameDataset(train, transform=None),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        for train in train_datasets
    ]
    test_dataloaders = [
        DataLoader(
            DataFrameDataset(test, transform=None),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        for test in test_datasets
    ]

    # Models                         Eval
    # Model1 = Tune on A(Model 0)    (On hold out from A)
    # Model2 = Tune on B(Model 1)    (On hold out from A, on hold out from B)
    # ...                            ....

    model_tuned = model
    for step in range(num_tasks):
        print(f"Training on task {step}")
        try:
            resume_ckpt = None
            if args.resume is not None:
                resume_ckpt = os.path.join(
                    args.resume, "checkpoints", str(step), "last.ckpt"
                )
            trainer.fit(
                model_tuned,
                train_dataloaders[step],
                val_dataloaders=test_dataloaders[step],
                ckpt_path=resume_ckpt,
            )
        finally:
            if trainer.global_rank == 0:
                final_ckpt = os.path.join(
                    experiment_log_dir, "checkpoints", str(step), "last.ckpt"
                )
                # trainer.save_checkpoint(final_ckpt)
        trainer.fit_loop.epoch_loop.val_loop._results.clear()
