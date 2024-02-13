import argparse
import os
import glob
import datetime
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from datasets import load_dataset
from transformers import AutoTokenizer
from ml_collections import ConfigDict

from experiments.utils import parse_devices, load_config, save_config, setup_log_dir
from experiments.laplace_lora import TransformerModule

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

    model = TransformerModule(config.model_config)

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
    model = TransformerModule(config.model_config)

    dataset = load_dataset(config.dataset_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            max_length=config.max_length,
            truncation=True,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns([config.inputs_key])
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]
    if config.small:
        train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
        eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    try:
        resume_ckpt = None
        if args.resume is not None:
            resume_ckpt = os.path.join(args.resume, "checkpoints", "last.ckpt")
        trainer.fit(model, train_dataloader, ckpt_path=resume_ckpt)
    finally:
        if trainer.global_rank == 0:
            final_ckpt = os.path.join(experiment_log_dir, "checkpoints", "last.ckpt")
            trainer.save_checkpoint(final_ckpt)
