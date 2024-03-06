import torch
import uqlib

name = "laplace"
save_dir = "experiments/yelp/results/" + name
params_dir = "experiments/yelp/results/map/state.pkl"  # directory to load state containing initialisation params

prior_sd = torch.inf
small_dataset = False
batch_size = 8

n_epochs = 1

method = uqlib.laplace.diag_fisher
config_args = {}  # arguments for method.build (aside from log_posterior)
log_metrics = {
    "loss": "aux.loss",
}  # dict containing names of metrics as keys and their paths in state as values
display_metric = "loss"  # metric to display in tqdm progress bar

log_frequency = 100  # frequency at which to log metrics
log_window = 40  # window size for moving average
