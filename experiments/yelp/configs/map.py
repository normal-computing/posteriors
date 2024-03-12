import uqlib
import torchopt

name = "map"
save_dir = "experiments/yelp/results/" + name
params_dir = None  # directory to load state containing initialisation params

prior_sd = 1e3
small_dataset = False
batch_size = 32

n_epochs = 1

method = uqlib.torchopt
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
