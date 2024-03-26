import torch
import posteriors
from optree import tree_map

name = "laplace_last_layer"
save_dir = "experiments/yelp/results/" + name
params_dir = "experiments/yelp/results/map/state.pkl"  # directory to load state containing initialisation params
last_layer_params_dir = "experiments/yelp/results/map_last_layer/state.pkl"  # directory to load state containing initialisation params for last layer

prior_sd = torch.inf
small_dataset = True
batch_size = 32
last_layer = True
burnin = None
save_frequency = None

n_epochs = 1

method = posteriors.laplace.diag_fisher
config_args = {}  # arguments for method.build (aside from log_posterior)
log_metrics = {
    "loss": "aux.loss",
}  # dict containing names of metrics as keys and their paths in state as values
display_metric = "loss"  # metric to display in tqdm progress bar

log_frequency = 100  # frequency at which to log metrics
log_window = 40  # window size for moving average

test_batch_size = batch_size
n_test_samples = 10
n_linearised_test_samples = 100000
test_save_dir = save_dir
epsilon = 1e-3  # small value to avoid division by zero in to_sd_diag


def to_sd_diag(state):
    return tree_map(lambda x: torch.sqrt(1 / (x + epsilon)), state.prec_diag)
