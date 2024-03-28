import posteriors
import torchopt
import torch

name = "map"
save_dir = "examples/yelp/results/" + name
params_dir = None  # directory to load state containing initialisation params
last_layer_params_dir = None

prior_sd = torch.inf
small_dataset = True
batch_size = 32
last_layer = False
burnin = None
save_frequency = None

n_epochs = 20

method = posteriors.torchopt
config_args = {
    "optimizer": torchopt.adamw(lr=1e-5, maximize=True)
}  # arguments for method.build (aside from log_posterior)
log_metrics = {
    "log_post": "loss",
    "loss": "aux.loss",
}  # dict containing names of metrics as keys and their paths in state as values
display_metric = "log_post"  # metric to display in tqdm progress bar

log_frequency = 100  # frequency at which to log metrics
log_window = 10  # window size for moving average

test_batch_size = batch_size
test_save_dir = save_dir
