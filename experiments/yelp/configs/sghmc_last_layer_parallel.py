import uqlib

name = "sghmc_parallel"
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

method = uqlib.sgmcmc.sghmc
config_args = {
    "lr": 1e-1,
    "alpha": 1e-2,
    "beta": 0.0,
    "temperature": None,  # None temperature gets set to 1/num_data
    "momenta": 0.0,
}  # arguments for method.build (aside from log_posterior)
log_metrics = {
    "log_post": "log_posterior",
    "loss": "aux.loss",
}  # dict containing names of metrics as keys and their paths in state as values
display_metric = "log_post"  # metric to display in tqdm progress bar

log_frequency = 100  # frequency at which to log metrics
log_window = 10  # window size for moving average

combined_dir = "experiments/yelp/results/sghmc_parallel_combined"
test_save_dir = combined_dir
test_batch_size = batch_size
