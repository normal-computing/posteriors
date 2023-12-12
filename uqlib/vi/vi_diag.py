from typing import Callable, Tuple, Any

import torch
from torch.distributions import Normal

from uqlib.utils import tree_map


def diagonal_normal_log_prob(x: dict, mean: dict, sd_diag: dict) -> float:
    output = 0
    for k in x.keys():
        output += Normal(mean[k], sd_diag[k]).log_prob(x[k]).sum()
    return output


def diagonal_nelbo(
    log_posterior: Callable[[Any, Any], float],
    batch: Tuple,
    mean: dict,
    log_sd_diag: dict,
    n_samples: int = 1,
) -> float:
    """Returns the negative evidence lower bound (NELBO) for a diagonal Gaussian
    variational distribution over the parameters of a model.

    Averages NELBO over the batch. Monte Carlo estimate with n_samples from q.

    NELBO = -E_q[log p(y|x, θ) + log p(θ) - log q(θ)]

    Args:
        model: Model.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised).
        batch: Tuple of (x, y) data.
        mean: Mean of the variational distribution.
        log_sd_diag: Log of the square-root diagonal of the covariance matrix of the
            variational distribution.
        n_samples: Number of samples to use for Monte Carlo estimate.

    Returns:
        The sampled approximate NELBO averaged over the batch.
    """
    sd_diag = tree_map(lambda x: x.exp(), log_sd_diag)

    def sample_params():
        return tree_map(lambda m, s: m + torch.randn_like(m) * s, mean, sd_diag)

    def single_nelbo(sampled_params):
        log_p = log_posterior(sampled_params, batch)
        log_q = diagonal_normal_log_prob(sampled_params, mean, sd_diag)
        return -(log_p - log_q).mean()

    nelbo = 0
    for _ in range(n_samples):
        nelbo += single_nelbo(sample_params()) / n_samples
    return nelbo
