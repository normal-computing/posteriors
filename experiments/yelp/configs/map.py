import uqlib
import torchopt

name = "map"
save_dir = "experiments/yelp/results/" + name
params_dir = None

prior_sd = 1e3
small_dataset = False
batch_size = 8

n_epochs = 1

method = uqlib.torchopt
config_args = {
    "optimizer": torchopt.adamw(lr=1e-5, maximize=True)
}  # arguments for method.build (aside from log_posterior)
log_metrics = {
    "log_post": "aux.loss",
    "loss": "loss",
}  # dict containing names of metrics as keys and their paths in state as values
display_metric = "loss"  # metric to display in tqdm progress bar

log_frequency = 100  # frequency at which to log metrics
