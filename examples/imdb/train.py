import os
import argparse
import pickle
import importlib
from tqdm import tqdm
from optree import tree_map
import torch
import posteriors

from examples.imdb.model import CNNLSTM
from examples.imdb.data import load_imdb_dataset
from examples.imdb import utils


# Get args from user
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--temperature", default=1.0, type=float)
args = parser.parse_args()


# Import configuration
config = importlib.import_module(args.config.replace("/", ".").replace(".py", ""))


# Set seed
if args.seed != 42:
    config.save_dir += f"_seed{args.seed}"
else:
    args.seed = 42
torch.manual_seed(args.seed)

# Load model
model = CNNLSTM(num_classes=2)
model.to(args.device)

if config.params_dir is not None:
    with open(config.params_dir, "rb") as f:
        state = pickle.load(f)
        model.load_state_dict(state.params)

# Load data
train_dataloader, test_dataloader = load_imdb_dataset(batch_size=config.batch_size)
num_data = len(train_dataloader.dataset)


# Set temperature
if "temperature" in config.config_args and config.config_args["temperature"] is None:
    config.config_args["temperature"] = args.temperature / num_data
    temp_str = str(args.temperature).replace(".", "-")
    config.save_dir += f"_temp{temp_str}"


# Create save directory if it does not exist
if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)

# Save config
utils.save_config(args, config.save_dir)
print(f"Config saved to {config.save_dir}")

# Extract model parameters
params = dict(model.named_parameters())
num_params = posteriors.tree_size(params).item()
print(f"Number of parameters: {num_params/1e6:.3f}M")


# Define log posterior
def forward(p, batch):
    x, y = batch
    logits = torch.func.functional_call(model, p, x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    return logits, (loss, logits)


def outer_log_lik(logits, batch):
    _, y = batch
    return -torch.nn.functional.cross_entropy(logits, y, reduction="sum")


def log_posterior(p, batch):
    x, y = batch
    logits = torch.func.functional_call(model, p, x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    log_post = (
        -loss
        + posteriors.diag_normal_log_prob(p, sd_diag=config.prior_sd, normalize=False)
        / num_data
    )
    return log_post, (loss, logits)


# Build transform
if config.method == posteriors.laplace.diag_ggn:
    transform = config.method.build(forward, outer_log_lik, **config.config_args)
else:
    transform = config.method.build(log_posterior, **config.config_args)

# Initialize state
state = transform.init(params)

# Train
i = j = 0
num_batches = len(train_dataloader)
log_dict = {k: [] for k in config.log_metrics.keys()} | {"loss": []}
log_bar = tqdm(total=0, position=1, bar_format="{desc}")
for epoch in range(args.epochs):
    for batch in tqdm(
        train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", position=0
    ):
        batch = tree_map(lambda x: x.to(args.device), batch)
        state = transform.update(state, batch)

        # Update metrics
        log_dict = utils.append_metrics(log_dict, state, config.log_metrics)
        log_bar.set_description_str(
            f"{config.display_metric}: {log_dict[config.display_metric][-1]:.2f}"
        )

        # Log
        i += 1
        if i % config.log_frequency == 0 or i % num_batches == 0:
            utils.log_metrics(
                log_dict,
                config.save_dir,
                window=config.log_window,
                file_name="training",
            )

        # Save sequential state if desired
        if (
            config.save_frequency is not None
            and (i - config.burnin) >= 0
            and (i - config.burnin) % config.save_frequency == 0
        ):
            with open(f"{config.save_dir}/state_{j}.pkl", "wb") as f:
                pickle.dump(state, f)
            j += 1

# Save final state
with open(f"{config.save_dir}/state.pkl", "wb") as f:
    pickle.dump(state, f)
