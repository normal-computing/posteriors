import pickle
import torch
import os
from optree import tree_map

sghmc_parallel = True

base = "experiments/yelp/results/"

if sghmc_parallel:
    # Get paths with sghmc_parallel_seed in them
    load_paths = [
        base + file + "/state.pkl"
        for file in os.listdir(base)
        if "sghmc_parallel_seed" in file
    ]

    save_dir = "experiments/yelp/results/sghmc_parallel_combined"
else:
    # Get paths with sghmc/stat_ in them
    load_paths = [
        base + file for file in os.listdir(base + "sghmc") if "state_" in file
    ]

    save_dir = "experiments/yelp/results/sghmc"

load_paths.sort()

# Create save directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# Save txt file with paths
with open(f"{save_dir}/paths.txt", "w") as f:
    f.write("\n".join(load_paths))


# Load states
states = [pickle.load(open(d, "rb")) for d in load_paths]

# Delete auxiliary info
for s in states:
    del s.aux

# Move states to cpu
states = tree_map(lambda x: x.detach().to("cpu"), states)

# Combine states
combined_state = tree_map(lambda *x: torch.stack(x), *states)

# Save state
with open(f"{save_dir}/state.pkl", "wb") as f:
    pickle.dump(combined_state, f)
