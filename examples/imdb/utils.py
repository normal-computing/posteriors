import json
import matplotlib.pyplot as plt
import numpy as np
import shutil


# Save config
def save_config(args, save_dir):
    shutil.copy(args.config, save_dir + "/config.py")
    json.dump(vars(args), open(f"{save_dir}/args.json", "w"))


# Function to calculate moving average and accompanying x-axis values
def moving_average(x, w):
    w_check = 1 if w >= len(x) else w
    y = np.convolve(x, np.ones(w_check), "valid") / w_check
    return range(w_check - 1, len(x)), y


# Function to log and plot metrics
def log_metrics(log_dict, save_dir, window=1, file_name="metrics", plot=True):
    save_file = f"{save_dir}/{file_name}.json"

    with open(save_file, "w") as f:
        json.dump(log_dict, f)

    if plot:
        for k, v in log_dict.items():
            fig, ax = plt.subplots()
            x, y = moving_average(v, window)
            ax.plot(x, y, label=k, alpha=0.7)
            ax.legend()
            ax.set_xlabel("Iteration")
            fig.tight_layout()
            fig.savefig(f"{save_dir}/{file_name}_{k}.png", dpi=200)
            plt.close(fig)


# Function to append metrics to log_dict
def append_metrics(log_dict, state, loss, config_dict):
    for k, v in config_dict.items():
        log_dict[k].append(getattr(state, v).mean().item())

    log_dict["loss"].append(loss.mean().item())
    return log_dict
