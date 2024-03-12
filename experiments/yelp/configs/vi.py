import uqlib
import torchopt
from optree import tree_map

name = "vi"
save_dir = "experiments/yelp/results/" + name
params_dir = None  # directory to load state containing initialisation params

prior_sd = 1e3
small_dataset = False
batch_size = 32

n_epochs = 1

method = uqlib.vi.diag
config_args = {
    "optimizer": torchopt.adamw(lr=1e-4),
    "temperature": None,
    "n_samples": 1,
    "stl": True,
    "init_log_sds": -5,
}  # arguments for method.build (aside from log_posterior)
log_metrics = {
    "nelbo": "nelbo",
    "loss": "aux.loss",
}  # dict containing names of metrics as keys and their paths in state as values
display_metric = "nelbo"  # metric to display in tqdm progress bar

log_frequency = 100  # frequency at which to log metrics
log_window = 10  # window size for moving average

test_batch_size = 2
n_test_samples = 10


def to_sd_diag(state):
    return tree_map(lambda x: x.exp(), state.log_sd_diag)
