import json
import os
import pickle
import importlib
import torch
from tqdm import tqdm
from optree import tree_map
import uqlib

from experiments.yelp.load import load_dataloaders, load_model


# Parser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_dir = "experiments/yelp/configs/map.py"

# Import configuration
config = importlib.import_module(config_dir.replace("/", ".").replace(".py", ""))

if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)


def log_training_metrics(losses, log_posts):
    save_file = f"{config.save_dir}/training.json"

    log_dict = {"losses": losses, "log_posts": log_posts}

    with open(save_file, "w") as f:
        json.dump(log_dict, f)


# Load data and model
train_dataloader, test_dataloader = load_dataloaders(
    small=config.small_dataset, batch_size=8
)
num_data = len(train_dataloader.dataset)
model, log_posterior = load_model(
    prior_sd=config.prior_sd, num_data=num_data, params_dir=config.params_dir
)
model.to(device)


# Extract parameters
params = dict(model.named_parameters())
num_params = uqlib.tree_size(params).item()
print(f"Number of parameters: {int(num_params/1e6)}M")


# Build transform
transform = config.method.build(log_posterior, **config.config_args)

# Initialize state
state = transform.init(params)

# Train
losses = []
log_posts = []
log_post_bar = tqdm(total=0, position=1, bar_format="{desc}")
for epoch in range(config.n_epochs):
    for batch in tqdm(
        train_dataloader, desc=f"Epoch {epoch+1}/{config.n_epochs}", position=0
    ):
        batch = tree_map(lambda x: x.to(device), batch)
        state = transform.update(state, batch)

        losses.append(state.aux.loss.item())
        log_posts.append(state.loss.item())
        log_post_bar.set_description_str(f"Log posterior: {log_posts[-1]:.2f}")

    log_training_metrics(losses, log_posts)
    with open(f"{config.save_dir}/state.pkl", "wb") as f:
        pickle.dump(state, f)
