import os
import argparse
import importlib
import pickle
import torch
from tqdm import tqdm
from optree import tree_map
import posteriors

from llama3.model import load_model, load_dataloaders
from examples.yelp import utils

# Get config path and device from user
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--seed", default=None, type=int)
args = parser.parse_args()

# Import configuration
config = importlib.import_module(args.config.replace("/", ".").replace(".py", ""))

# Set seed
if args.seed is not None:
    config.save_dir += f"_seed{args.seed}"
else:
    args.seed = 42
torch.manual_seed(args.seed)

# Create save directory if it does not exist
if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)

# Save config
utils.save_config(args.config, config.save_dir)

# Load dataloaders
train_dataloader, test_dataloader = load_dataloaders(
    config, batch_size=config.train_batch_size
)

model, sub_param_to_log_posterior, sub_params = load_model(
    config["bayes_config"], device=args.device
)

num_data = len(train_dataloader.dataset)
# Update temperature if method requires it
if "temperature" in config.config_args and config.config_args["temperature"] is None:
    config.config_args["temperature"] = 1 / num_data

# Extract model parameters
num_params = posteriors.tree_size(sub_params).item()
print(f"Number of parameters: {num_params/1e6:.3f}M")

# Build transform
transform = config.method.build(sub_param_to_log_posterior, **config.config_args)

# Initialize state
state = transform.init(sub_params)

# Train
i = j = 0
num_batches = len(train_dataloader)
log_dict = {k: [] for k in config.log_metrics.keys()}
log_bar = tqdm(total=0, position=1, bar_format="{desc}")
for epoch in range(config.n_epochs):
    for batch in tqdm(
        train_dataloader, desc=f"Epoch {epoch+1}/{config.n_epochs}", position=0
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
