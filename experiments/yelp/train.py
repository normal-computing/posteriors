import os
import argparse
import pickle
import importlib
from tqdm import tqdm
from optree import tree_map
import uqlib

from experiments.yelp.load import load_dataloaders, load_model
from experiments.yelp import utils

# Get config path and device from user
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--device", default="cpu", type=str)
args = parser.parse_args()

# Import configuration
config = importlib.import_module(args.config.replace("/", ".").replace(".py", ""))

# Create save directory if it does not exist
if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)

# Save config
utils.save_config(args.config, config.save_dir)

# Load data and model
train_dataloader, test_dataloader = load_dataloaders(
    small=config.small_dataset, batch_size=config.batch_size
)
num_data = len(train_dataloader.dataset)
model, log_posterior = load_model(
    prior_sd=config.prior_sd, num_data=num_data, params_dir=config.params_dir
)
model.to(args.device)
print("Device: ", model.device)

if "temperature" in config.config_args and config.config_args["temperature"] is None:
    config.config_args["temperature"] = 1 / num_data

# Extract model parameters
params = dict(model.named_parameters())
num_params = uqlib.tree_size(params).item()
print(f"Number of parameters: {int(num_params/1e6)}M")

# Build transform
transform = config.method.build(log_posterior, **config.config_args)

# Initialize state
state = transform.init(params)

# Train
i = 0
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
            utils.log_metrics(log_dict, config.save_dir, window=config.log_window)

    # Save state
    with open(f"{config.save_dir}/state2.pkl", "wb") as f:
        pickle.dump(state, f)
