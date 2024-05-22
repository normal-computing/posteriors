import torch
import posteriors
from optree import tree_map

name = "laplace_fisher"
save_dir = "examples/imdb/results/" + name
params_dir = "examples/imdb/results/map"  # directory to load state containing initialisation params

prior_sd = torch.inf
batch_size = 32
burnin = None
save_frequency = None

method = posteriors.laplace.diag_fisher
config_args = {}  # arguments for method.build (aside from log_posterior)
log_metrics = {}  # dict containing names of metrics as keys and their paths in state as values
display_metric = "loss"  # metric to display in tqdm progress bar

log_frequency = 100  # frequency at which to log metrics
log_window = 30  # window size for moving average

n_test_samples = 50
n_linearised_test_samples = 10000
epsilon = 1e-3  # small value to avoid division by zero in to_sd_diag


def to_sd_diag(state, temperature=1.0):
    return tree_map(lambda x: torch.sqrt(temperature / (x + epsilon)), state.prec_diag)


def forward(model, state, batch, temperature=1.0):
    x, _ = batch
    sd_diag = to_sd_diag(state, temperature)

    sampled_params = posteriors.diag_normal_sample(
        state.params, sd_diag, (n_test_samples,)
    )

    def model_func(p, x):
        return torch.func.functional_call(model, p, x)

    logits = torch.vmap(model_func, in_dims=(0, None))(sampled_params, x).transpose(
        0, 1
    )
    return logits


def forward_linearized(model, state, batch, temperature=1.0):
    x, _ = batch
    sd_diag = to_sd_diag(state, temperature)

    def model_func_with_aux(p, x):
        return torch.func.functional_call(model, p, x), torch.tensor([])

    lin_mean, lin_chol, _ = posteriors.linearized_forward_diag(
        model_func_with_aux,
        state.params,
        x,
        sd_diag,
    )

    samps = torch.randn(
        lin_mean.shape[0],
        n_linearised_test_samples,
        lin_mean.shape[1],
        device=lin_mean.device,
    )
    lin_logits = lin_mean.unsqueeze(1) + samps @ lin_chol.transpose(-1, -2)
    return lin_logits


forward_dict = {"Laplace EF": forward, "Laplace EF Linearized": forward_linearized}
