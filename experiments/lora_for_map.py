import os
import argparse
from transformers import AutoTokenizer
from lightning.pytorch import Trainer
from experiments.utils import load_config, parse_devices
from experiments.data.load_pg19 import load_pg19_dataloaders
from experiments.laplace_lora import BayesTransformerModule

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


tokenizer = AutoTokenizer.from_pretrained(
    config.model_config.pretrained_model_name_or_path
)

tokenizer.pad_token = tokenizer.eos_token

train_dataloaders, test_dataloaders = load_pg19_dataloaders(config, tokenizer)


model = BayesTransformerModule(config.model_config)

batch = next(iter(train_dataloaders[0]))
batch = {k: v.to(model.device) for k, v in batch.items()}
output = model.model(labels=batch["input_ids"], **batch)


# Logs to tensorboard by default
trainer = Trainer(**trainer_kwargs)

for book_ind in range(config.num_tasks):
    print(f"Training on book {book_ind} of {config.num_tasks}")
    train_dataloader = train_dataloaders[book_ind]
    test_dataloader = test_dataloaders[book_ind]
    trainer.fit(
        model,
        train_dataloaders[book_ind],
        val_dataloaders=test_dataloaders[book_ind],
    )
