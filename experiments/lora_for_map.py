import os
import argparse
import torch
import datetime
from transformers import AutoTokenizer
from lightning.pytorch import Trainer
from experiments.utils import load_config, parse_devices, save_config, setup_log_dir
from experiments.data.load_pg19 import load_pg19_dataloaders
from experiments.laplace_lora.lora_transformer_phoebe import BayesTransformerModule
from ml_collections.config_dict import ConfigDict
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--name", default=None, type=str)
parser.add_argument("--base", default=None, type=str)
parser.add_argument("--devices", default=parse_devices, type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--log_frequency", default=10, type=int)
parser.add_argument("--seed", default=42, type=int)

args = parser.parse_args()

device_type = "cpu" if callable(args.devices) else "gpu"

trainer_kwargs = {
    "max_epochs": args.epochs,
    "accelerator": device_type,
    "log_every_n_steps": args.log_frequency,
}
config = load_config(args.base)

timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
experiment_name = config.get("experiment_name", None)

experiment_log_dir = setup_log_dir(
    config.get("logs_dir", "logs"),
    timestamp,
    experiment_name=experiment_name,
)

torch.set_float32_matmul_precision("medium")
torch.manual_seed(args.seed)


tokenizer = AutoTokenizer.from_pretrained(
    config.model_config.pretrained_model_name_or_path
)

tokenizer.pad_token = tokenizer.eos_token

config = ConfigDict(config)  # thaw
config["model_config"]["experiment_log_dir"] = experiment_log_dir + "/eval_metrics.txt"

model = BayesTransformerModule(config.model_config)

wandb.init(
    project=config["experiment_name"],
    config=config.to_dict(),
    dir=config.logs_dir,
    notes=config["notes"] if "notes" in config else None,
    tags=config["tags"] if "tags" in config else None,
)
config.wandb_id = wandb.run.id
config.wandb_name = wandb.run.name
save_config(config.to_dict(), experiment_log_dir + "/config.yaml")
with open(experiment_log_dir + "/eval_metrics.txt", "w") as f:
    f.write(
        "epoch,step,task,val_task,metric_name,metric_value\n"
    )  # Header for CSV format

# To log model gradients and parameters
wandb.watch(model)

train_dataloaders, test_dataloaders = load_pg19_dataloaders(
    config, tokenizer, batch_size=config.batch_size
)

for book_ind in range(config.num_tasks):
    trainer = Trainer(**trainer_kwargs)
    print(f"Training on book {book_ind+1} of {config.num_tasks}")
    model.task_no = book_ind
    trainer.fit(
        model,
        train_dataloaders=train_dataloaders[book_ind],
        val_dataloaders=[
            test_dataloaders[i] for i in range(book_ind + 1)
        ],  # test on all tasks seen so far
    )
