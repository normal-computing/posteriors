import os
import argparse
from transformers import AutoTokenizer
from lightning.pytorch import Trainer
import torch
import optree

import uqlib

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


####
config = load_config("experiments/utils/configs/lora_sam.yaml")
trainer_kwargs = {
    "max_epochs": 1,
    "accelerator": "gpu",
    "log_every_n_steps": 1,
}
#####


tokenizer = AutoTokenizer.from_pretrained(
    config.model_config.pretrained_model_name_or_path
)

tokenizer.pad_token = tokenizer.eos_token

train_dataloaders, test_dataloaders = load_pg19_dataloaders(
    config, tokenizer, batch_size=config.train_batch_size
)

laplace_train_dataloaders, _ = load_pg19_dataloaders(
    config, tokenizer, batch_size=config.laplace_batch_size
)

model = BayesTransformerModule(config.model_config)

model.prior_mean = optree.tree_map(torch.zeros_like, model.sub_params)
model.prior_sd = optree.tree_map(
    lambda x: torch.ones_like(x) * config.model_config.first_prior_sd, model.sub_params
)

# Logs to tensorboard by default
trainer = Trainer(**trainer_kwargs)

for book_ind in range(config.num_tasks):
    print(f"Training on book {book_ind} of {config.num_tasks}")

    # Train for MAP
    model.num_data = len(train_dataloaders[book_ind].dataset)

    trainer.fit(
        model,
        train_dataloaders[book_ind],
        val_dataloaders=[test_dataloaders[i] for i in range(book_ind + 1)],
    )
    trainer.fit_loop.epoch_loop._results.clear()

    if config.lambda_param > 0.0:
        # Get Laplace precision diag
        laplace_transform = uqlib.laplace.diag_fisher.build(
            model.sub_param_to_log_posterior
        )
        laplace_state = laplace_transform.init(model.sub_params)
        for batch in laplace_train_dataloaders[book_ind]:
            batch = optree.tree_map(lambda x: x.to(model.device), batch)
            laplace_state = laplace_transform.update(
                laplace_state,
                batch,
            )

        # Update sequential prior
        model.prior_mean = optree.tree_map(
            lambda mu, q, sig, f: (sig**-2 * mu + config.lambda_param * f * q)
            / (sig**-2 + config.lambda_param * f),
            model.prior_mean,
            laplace_state.mean,
            model.prior_sd,
            laplace_state.prec_diag,
        )
        model.prior_sd = optree.tree_map(
            lambda sig, f: 1 / torch.sqrt(sig**-2 + f * config.lambda_param),
            model.prior_sd,
            laplace_state.prec_diag,
        )
