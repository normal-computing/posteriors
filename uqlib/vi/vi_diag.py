from typing import Callable, Tuple

import torch
from torch.distributions import Normal

from uqlib.utils import tree_map, model_to_function


def diagonal_normal_log_prob(x: dict, mean: dict, sd_diag: dict) -> float:
    output = 0
    for k in x.keys():
        output += Normal(mean[k], sd_diag[k]).log_prob(x[k]).sum()
    return output


def diagonal_nelbo(
    model: torch.nn.Module,
    log_prior: Callable,
    log_likelihood: Callable,
    batch: Tuple,
    mean: dict,
    log_var_diag: dict,
    n_samples: int = 1,
) -> float:
    """Returns the negative evidence lower bound (NELBO) for a diagonal Gaussian
    variational distribution over the parameters of a model.

    Averages NELBO over the batch. Monte Carlo estimate with n_samples from q.

    NELBO = -E_q[log p(y|x, θ) + log p(θ) - log q(θ)]

    Args:
        model: Model.
        log_prior: Function that takes a dictionary of parameters and
            returns the log prior (which can be unnormalised).
        log_likelihood: Function that takes a batch of output data
            and output data from the model and returns the log likelihood
            (which can be unnormalised).
            log_lik = log_likelihood(y, model(X))
        batch: Tuple of (x, y) data.
        mean: Mean of the variational distribution.
        log_var_diag: Log of the diagonal of the covariance matrix of the
            variational distribution.
        n_samples: Number of samples to use for Monte Carlo estimate.

    Returns:
        The sampled approximate NELBO averaged over the batch.
    """
    model_func = model_to_function(model)

    x, y = batch

    sd_diag = tree_map(lambda x: x.exp() ** 0.5, log_var_diag)

    def sample_params():
        return tree_map(lambda m, s: m + torch.randn_like(m) * s, mean, sd_diag)

    def single_nelbo(sampled_params):
        log_p = log_prior(sampled_params)
        log_q = diagonal_normal_log_prob(sampled_params, mean, sd_diag)

        log_p_y_given_xp = log_likelihood(y, model_func(sampled_params, x))

        return -(log_p + log_p_y_given_xp - log_q).mean()

    loss = 0
    for _ in range(n_samples):
        loss += single_nelbo(sample_params()) / n_samples

    return loss
