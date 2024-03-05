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
