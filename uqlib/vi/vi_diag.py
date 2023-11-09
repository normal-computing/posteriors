from typing import Callable, Tuple

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.distributions import Normal


def nelbo(
    model: torch.nn.Module,
    log_prior: Callable,
    log_likelihood: Callable,
    batch: Tuple,
    mu: torch.Tensor,
    log_sigma_diag: torch.Tensor,
) -> float:
    """Returns the negative evidence lower bound (NELBO) for a diagonal Gaussian
    variational distribution over the parameters of a model.

    Averages NELBO over the batch.

    NELBO = -E_q[log p(y|x, p) + log p(p) - log q(p)]

    Args:
        model: Model.
        log_prior: Function that takes a vector of parameters and returns the log prior
            (which can be unnormalised).
        log_likelihood: Function that takes a batch of output data
            and output data from the model and returns the log likelihood
            (which can be unnormalised).
            log_lik = log_likelihood(y, model(X))
        batch: Tuple of (x, y) data.
        mu: Mean of the variational distribution.
        log_sigma_diag: Log of the diagonal of the covariance matrix of the
            variational distribution.

    Returns:
        The NELBO averaged over the batch.
    """
    orig_p = parameters_to_vector(model.parameters())

    x, y = batch

    sigma_diag = log_sigma_diag.exp()

    # Sample p from using the reparameterization trick
    eps = torch.randn_like(log_sigma_diag)
    p = mu + eps * sigma_diag**0.5

    log_p = log_prior(p)

    vector_to_parameters(p, model.parameters())
    log_p_y_given_xp = log_likelihood(y, model(x))

    log_q = Normal(mu, sigma_diag).log_prob(p).sum(dim=1)

    loss = -(log_p + log_p_y_given_xp - log_q).mean()

    vector_to_parameters(orig_p, model.parameters())
    return loss
