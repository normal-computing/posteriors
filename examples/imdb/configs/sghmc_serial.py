import posteriors
import torch

name = "sghmc_serial"
save_dir = "examples/imdb/results/" + name
params_dir = None  # directory to load state containing initialisation params


prior_sd = (1 / 40) ** 0.5
batch_size = 32
burnin = 20000
save_frequency = 1000

lr = 1e-1

method = posteriors.sgmcmc.sghmc
config_args = {
    "lr": lr,
    "alpha": 1.0,
    "beta": 0.0,
    "temperature": None,  # None temperature gets set by train.py
    "momenta": 0.0,
}  # arguments for method.build (aside from log_posterior)
log_metrics = {
    "log_post": "log_posterior",
}  # dict containing names of metrics as keys and their paths in state as values
display_metric = "log_post"  # metric to display in tqdm progress bar

log_frequency = 100  # frequency at which to log metrics
log_window = 30  # window size for moving average


def forward(model, state, batch):
    x, _ = batch

    def model_func(p, x):
        return torch.func.functional_call(model, p, x)

    logits = torch.vmap(model_func, in_dims=(0, None))(state.params, x).transpose(0, 1)
    return logits


forward_dict = {"Serial SGHMC": forward}
