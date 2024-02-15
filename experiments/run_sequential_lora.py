import argparse
import os
import glob
import datetime
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer
from ml_collections.config_dict import ConfigDict
from datasets import load_dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader

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

    dataset = load_dataset("json", data_files="./experiments/data/pg19-small.json")

    def split_doc(examples):
        text_as_list = examples["text"].split()
        start_index = int(len(text_as_list) * 0.85)
        return {
            "train_text": " ".join(text_as_list[:start_index]),
            "test_text": " ".join(text_as_list[start_index:]),
        }

    def tokenize_function(examples):
        return {
            "train_tokens": tokenizer(
                examples["train_text"],
                padding="max_length",
                max_length=4096,
                truncation=True,
                return_tensors="pt",
            )["input_ids"],
            "test_tokens": tokenizer(
                examples["test_text"],
                padding="max_length",
                max_length=4096,
                truncation=True,
                return_tensors="pt",
            )["input_ids"],
        }

    split_datasets = dataset.map(split_doc, batched=False)
    tokenized_datasets = split_datasets.map(tokenize_function, batched=False)

    tokenized_datasets = tokenized_datasets.remove_columns(["train_text", "test_text"])
    tokenized_datasets.set_format("torch")

    num_samples_per_task = config.num_samples_per_task
    num_tasks = config.num_tasks

    train_datasets = [
        tokenized_datasets["train"]
        .select(range(i * num_samples_per_task, (i + 1) * num_samples_per_task))
        .remove_columns(["test_text", "train_text", "test_tokens"])
        .rename_column("train_tokens", "input_ids")
        for i in range(num_tasks)
    ]
    test_datasets = [
        tokenized_datasets["train"]
        .select(range((i + 1) * num_samples_per_task))
        .remove_columns(["test_text", "train_text", "train_tokens"])
        .rename_column("test_tokens", "input_ids")
        for i in range(num_tasks)
    ]

    # This is a terrible fcn from chat gpt to correct the collate_fn .... will fix tomorrow
    def custom_collate_fn(batch):
        # Extracting input_ids and preparing for tensor conversion
        input_ids = []
        other_data = {
            k: [] for k in batch[0].keys() if k != "input_ids"
        }  # Preparing for other data

        for item in batch:
            # Assuming input_ids is a dictionary and you want the 'input_ids' key from it
            if "input_ids" in item and "input_ids" in item["input_ids"]:
                input_ids.append(item["input_ids"]["input_ids"])
            for k in other_data.keys():
                if k in item:
                    other_data[k].append(item[k])

        if input_ids:
            batched_input_ids = torch.stack([torch.tensor(i) for i in input_ids])
        else:
            batched_input_ids = (
                torch.Tensor()
            )  # Handle the case where input_ids might be missing

        for k, v in other_data.items():
            other_data[k] = default_collate(v)

        if input_ids:
            other_data["input_ids"] = batched_input_ids
        return other_data

    train_dataloaders = [
        DataLoader(
            train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn,
        )
        for train in train_datasets
    ]
    test_dataloaders = [
        DataLoader(
            test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn,
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
