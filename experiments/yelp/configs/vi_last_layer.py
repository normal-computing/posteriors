import uqlib
import torchopt
from optree import tree_map

name = "vi_last_layer"
save_dir = "experiments/yelp/results/" + name
params_dir = "experiments/yelp/results/map/state.pkl"  # directory to load state containing initialisation params
last_layer_params_dir = None


prior_sd = 1e3
small_dataset = True
batch_size = 32
last_layer = True
burnin = None
save_frequency = None

n_epochs = 20

method = uqlib.vi.diag
config_args = {
    "optimizer": torchopt.adamw(lr=1e-2),
    "temperature": None,  # None temperature gets set to 1/num_data
    "n_samples": 1,
    "stl": True,
    "init_log_sds": -2,
}  # arguments for method.build (aside from log_posterior)
log_metrics = {
    "nelbo": "nelbo",
    "loss": "aux.loss",
}  # dict containing names of metrics as keys and their paths in state as values
display_metric = "nelbo"  # metric to display in tqdm progress bar

log_frequency = 100  # frequency at which to log metrics
log_window = 10  # window size for moving average

test_batch_size = batch_size
test_save_dir = save_dir
n_test_samples = 10
n_linearised_test_samples = 1000


def to_sd_diag(state):
    return tree_map(lambda x: x.exp(), state.log_sd_diag)
