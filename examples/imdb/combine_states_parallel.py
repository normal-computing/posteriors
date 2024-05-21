import pickle
import torch
import os
from optree import tree_map
import shutil


temperatures = [0.03, 0.1, 0.3, 1.0, 3.0]


# SGHMC Parallel
base = "examples/imdb/results/sghmc_parallel/sghmc_parallel"
save_dir_base = "examples/imdb/results/sghmc_parallel"


bootstrap_seeds = [
    (torch.multinomial(torch.ones(35), 15, replacement=False) + 1).tolist()
    for _ in range(5)
]


for temp in temperatures:
    temp_str = str(temp).replace(".", "-")

    load_paths = []

    for k, seeds in enumerate(bootstrap_seeds):
        for seed in seeds:
            spec_base = base
            spec_base += f"_seed{seed}"
            spec_base += f"_temp{temp_str}/"

            match_str = (
                "state_" if seed is None else "state"
            )  # don't need last save for serial

            load_paths += [
                spec_base + file for file in os.listdir(spec_base) if match_str in file
            ]

        save_dir = save_dir_base + f"_seed{k+1}" + f"_temp{temp_str}/"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            shutil.copy(spec_base + "config.py", save_dir + "/config.py")

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
        with open(f"{save_dir}state.pkl", "wb") as f:
            pickle.dump(combined_state, f)
