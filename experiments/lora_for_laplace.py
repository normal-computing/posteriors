import os
import argparse
import datetime
from transformers import AutoTokenizer
from lightning.pytorch import Trainer
import torch
import optree
import tqdm
from ml_collections.config_dict import ConfigDict

import uqlib

from experiments.utils import load_config, parse_devices, setup_log_dir, save_config
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
parser.add_argument("--sanity_checks", default=0, type=int)

args = parser.parse_args()

device_type = "cpu" if callable(args.devices) else "gpu"


trainer_kwargs = {
    "max_epochs": args.epochs,
    "accelerator": device_type,
    "log_every_n_steps": args.log_frequency,
    "num_sanity_val_steps": args.sanity_checks,
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
save_config(config.to_dict(), experiment_log_dir + "/config.yaml")
with open(experiment_log_dir + "/eval_metrics.txt", "w") as f:
    f.write(
        "epoch,step,task,val_task,metric_name,metric_value\n"
    )  # Header for CSV format

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

for book_ind in range(config.num_tasks):
    print(f"Training on book {book_ind + 1} of {config.num_tasks}")

    # Logs to tensorboard by default
    trainer = Trainer(**trainer_kwargs)

    model.task_no = book_ind

    model.num_data = len(train_dataloaders[book_ind].dataset)

    model.set_sub_params()
    model.zero_grad()

    # Train for MAP
    trainer.fit(
        model,
        train_dataloaders[book_ind],
        val_dataloaders=[test_dataloaders[i] for i in range(book_ind + 1)],
    )

    if config.lambda_param > 0.0 and book_ind < config.num_tasks - 1:
        print(f"Fitting Laplace on book {book_ind + 1} of {config.num_tasks}")

        # Get Laplace precision diag
        laplace_transform = uqlib.laplace.diag_fisher.build(
            model.sub_param_to_log_posterior
        )
        model.to(
            "cuda" if device_type == "gpu" else "cpu"
        )  # not using lightning for this part so need to move to device ourselves
        model.configure_optimizers()  # moves sub_params, prior_mean, prior_sd to device
        laplace_state = laplace_transform.init(model.sub_params)
        for batch in tqdm.tqdm(laplace_train_dataloaders[book_ind]):
            batch = optree.tree_map(lambda x: x.to(model.device), batch)
            laplace_state = laplace_transform.update(
                laplace_state, batch, inplace=False
            )

        def detach(ten):
            if isinstance(ten, torch.Tensor):
                return ten.detach()

        laplace_state = optree.tree_map(detach, laplace_state)

        # Update sequential prior
        if config.average_priors:
            rescale_param = model.num_data * config.lambda_param
            model.prior_mean = optree.tree_map(
                lambda mu, q, sig, f: (sig**-2 * mu + rescale_param * f * q)
                / (sig**-2 + rescale_param * f),
                model.prior_mean,
                laplace_state.mean,
                model.prior_sd,
                laplace_state.prec_diag,
            )
            model.prior_sd = optree.tree_map(
                lambda sig, f: 1 / torch.sqrt(sig**-2 + f * rescale_param),
                model.prior_sd,
                laplace_state.prec_diag,
            )

        else:
            model.prior_mean = optree.tree_map(lambda x: x.clone(), laplace_state.mean)
            model.prior_sd = optree.tree_map(
                lambda f: 1 / torch.sqrt(f * config.lambda_param),
                laplace_state.prec_diag,
            )
