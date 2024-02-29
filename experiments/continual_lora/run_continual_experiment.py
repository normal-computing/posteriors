import os
import argparse
import datetime
from transformers import AutoTokenizer
import torch
from torch.distributions import Categorical
import optree
import tqdm
from ml_collections.config_dict import ConfigDict
from functools import partial

import uqlib

from experiments.continual_lora import utils
from experiments.continual_lora.load_pg19 import load_pg19_dataloaders
from experiments.continual_lora.load_model import load_lora

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load user arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base", default=None, type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--seed", default=42, type=int)

args = parser.parse_args()

# Torch global settings
torch.set_float32_matmul_precision("medium")
torch.manual_seed(args.seed)

# Get config with info for dataset, model and simulation params
config = utils.load_config(args.base)


# Set up logging
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
experiment_name = config.get("experiment_name", None)
experiment_log_dir = utils.setup_log_dir(
    config.get("logs_dir", "logs"),
    timestamp,
    experiment_name=experiment_name,
)
config = ConfigDict(config)  # thaw
config["train_log_dir"] = experiment_log_dir + "/train_metrics.txt"
config["eval_log_dir"] = experiment_log_dir + "/eval_metrics.txt"
utils.save_config(config.to_dict(), experiment_log_dir + "/config.yaml")
with open(experiment_log_dir + "/train_metrics.txt", "w") as f:
    f.write("epoch,task,metric_name,metric_value\n")  # Train header
with open(experiment_log_dir + "/eval_metrics.txt", "w") as f:
    f.write("epoch,task,val_task,metric_name,metric_value\n")  # Eval header


def log_metrics(dir, current_epoch, episode_ind, val_ind, metrics):
    with open(dir, "a") as f:
        for metric_name, metric_value in metrics.items():
            f.write(
                f"{current_epoch},{episode_ind},{val_ind},{metric_name},{metric_value}\n"
            )


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    config.model_config.pretrained_model_name_or_path
)
tokenizer.pad_token = tokenizer.eos_token

# Load dataloaders
train_dataloaders, test_dataloaders = load_pg19_dataloaders(
    config, tokenizer, batch_size=config.train_batch_size
)

# Laplace requires smaller batch size
laplace_train_dataloaders, _ = load_pg19_dataloaders(
    config, tokenizer, batch_size=config.laplace_batch_size
)

# Load LoRA language model
model = load_lora(config.model_config)
model_func = uqlib.model_to_function(model)
model.print_trainable_parameters()


# Convert logits to log likelihood
# scale_factor is required to rescale stochastic log likelihood so that it is
# an unbiased estimate of the full log likelihood
def logits_to_log_lik(logits, labels, keep_mask):
    # pred_seq_len = seq_length - ignore_first
    logits_start = config.model_config.ignore_first
    logits = logits[
        :, (logits_start - 1) : -1, :
    ].contiguous()  # (batch_size, pred_seq_len, vocab_size)
    labels = labels[:, logits_start:].contiguous()  # (batch_size, pred_seq_len)
    keep_mask = keep_mask[:, logits_start:].contiguous()  # (batch_size, pred_seq_len)
    log_lik_all = Categorical(logits=logits, validate_args=False).log_prob(labels)
    log_lik_all *= keep_mask
    # sum over sequence, average over batch to give unbiased estimate of single sequence log likelihood
    # (within sequence log likelihoods are not independent)
    log_lik = log_lik_all.sum(1).mean()
    return log_lik  # rescale to full log likelihood


# Full parameter set log_likelihood
def param_to_log_likelihood(params, batch):
    output = model_func(params, labels=batch["input_ids"], **batch)
    log_lik = logits_to_log_lik(
        output.logits, batch["input_ids"], batch["attention_mask"]
    )
    return log_lik, output


# Extract only the LoRA parameters that require gradients
sub_params, sub_param_to_log_likelihood = uqlib.extract_requires_grad_and_func(
    dict(model.named_parameters()), param_to_log_likelihood
)


# Generic (unnormalised) Normal log prior
def normal_log_prior(params, prior_mean, prior_sd) -> float:
    per_group_vals = optree.tree_map(
        lambda p, m, sd: (-0.5 * ((p - m) / sd) ** 2).sum(),
        params,
        prior_mean,
        prior_sd,
    )
    return optree.tree_reduce(torch.add, per_group_vals)


# Combine likelihod and prior (on LoRA params only)
# Gives unbiased estimate of log posterior / num_sequences
# Where num_sequences is the number of sequences in the training data
def sub_param_to_log_posterior(p, batch, prior_mean, prior_sd, num_sequences):
    log_lik, output = sub_param_to_log_likelihood(p, batch)
    log_prior = normal_log_prior(p, prior_mean, prior_sd)
    log_post = log_lik + log_prior / num_sequences
    return log_post, output


# Move model to device
model.to(args.device)

# Set inital prior
current_prior_mean = optree.tree_map(torch.zeros_like, sub_params)
current_prior_sd = optree.tree_map(
    lambda p: torch.ones_like(p) * config.model_config.first_prior_sd,
    sub_params,
)

# Run experiment!
for episode_ind in range(config.num_tasks):
    print(f"Training on episode {episode_ind + 1} of {config.num_tasks}")

    # Set up log_posterior
    train_sub_param_to_log_posterior = partial(
        sub_param_to_log_posterior,
        prior_mean=current_prior_mean,
        prior_sd=current_prior_sd,
        num_sequences=len(train_dataloaders[episode_ind].dataset.data["input_ids"]),
    )

    # Set up optimizer
    optimizer = torch.optim.AdamW(sub_params.values(), lr=config.lr, maximize=True)

    # Train for MAP
    train_dl = train_dataloaders[episode_ind]
    test_dls = [test_dataloaders[i] for i in range(episode_ind + 1)]
    optimizer.zero_grad()
    best_loss = torch.inf
    for epoch in tqdm.tqdm(range(args.epochs)):
        for i, batch in enumerate(train_dl):
            batch = optree.tree_map(lambda x: x.to(args.device), batch)
            log_post, output = train_sub_param_to_log_posterior(sub_params, batch)
            log_post.backward()
            log_metrics(
                config.train_log_dir,
                epoch,
                episode_ind,
                episode_ind,
                {"train_log_post": log_post.item()},
            )
            log_metrics(
                config.train_log_dir,
                epoch,
                episode_ind,
                episode_ind,
                {"train_loss": output.loss.item()},
            )

            if (
                (i + 1) % config.accumulate_gradients_every == 0
            ):  # Wait for several backward steps
                optimizer.step()
                optimizer.zero_grad()

        # Validation across all tasks seen so far
        with torch.inference_mode():
            for val_ind, test_dl in enumerate(test_dls):
                losses = []
                for batch in test_dl:
                    batch = optree.tree_map(lambda x: x.to(args.device), batch)
                    log_lik, output = sub_param_to_log_likelihood(sub_params, batch)
                    losses.append(-log_lik.item())

                # Print validation
                print(
                    f"\nEpisode {episode_ind}, \t Epoch {epoch}, \t Val ind {val_ind}, \t Val loss: {torch.tensor(losses).mean()}"
                )

                # Log validation
                log_metrics(
                    config.eval_log_dir,
                    epoch,
                    episode_ind,
                    val_ind,
                    {f"val_loss_task_{val_ind}": torch.tensor(losses).mean()},
                )

        if config.early_stopping:
            if torch.tensor(losses).mean() < best_loss:
                best_loss = torch.tensor(losses).mean()
            else:
                print(f"Early stopping on epoch {epoch}.")
                break

    # Laplace approximation
    if config.lambda_param > 0.0 and episode_ind < config.num_tasks - 1:
        print(f"Fitting Laplace on episode {episode_ind + 1} of {config.num_tasks}")

        # Get Laplace precision diag
        laplace_transform = uqlib.laplace.diag_fisher.build(sub_param_to_log_likelihood)

        # Update Laplace state through data
        laplace_state = laplace_transform.init(sub_params)
        for batch in tqdm.tqdm(laplace_train_dataloaders[episode_ind]):
            batch = optree.tree_map(lambda x: x.to(args.device), batch)
            laplace_state = laplace_transform.update(
                laplace_state, batch, inplace=False
            )

        # Detach Laplace state to stop gradients
        def detach(ten):
            if isinstance(ten, torch.Tensor):
                return ten.detach()

        laplace_state = optree.tree_map(detach, laplace_state)

        # Update sequential prior
        current_prior_mean = optree.tree_map(lambda x: x.clone(), laplace_state.mean)
        current_prior_sd = optree.tree_map(
            lambda p, f: (p**-2 + f * config.lambda_param) ** -0.5,
            current_prior_sd,
            laplace_state.prec_diag,
        )
