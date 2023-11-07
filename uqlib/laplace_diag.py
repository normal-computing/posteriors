from typing import Callable

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from utils import diagonal_hessian


def get_covariance(
    model: torch.nn.Module,
    log_prior: Callable,
    log_likelihood: Callable,
    train_dataloader: Callable,
):
    """Fit diagonal Hessian to data.

    Args:
        model: Model with parameters trained to MAP estimator.
        log_prior: Function that takes a vector of parameters and returns the log prior
            (which can be unnormalised).
        log_likelihood: Function that takes a batch of output data
            and output data from the model and returns the log likelihood
            (which can be unnormalised).
            log_lik = log_likelihood(y, model(X))
        train_dataloader: The training data. Iterable that supplies batches in the form
            of (X, y) tuples.

    Returns:
        The fitted model.
    """
    orig_p = parameters_to_vector(model.parameters())

    def p_to_log_lik(p, X, y):
        vector_to_parameters(p, model.parameters())
        return torch.mean(log_likelihood(y, model(X)))

    diag_prior_hess = diagonal_hessian(log_prior)(orig_p)
    diag_lik_hess = torch.zeros_like(diag_prior_hess)

    n_data = 0

    for X, y in train_dataloader:
        n_data += X.shape[0]
        diag_lik_hess += diagonal_hessian(p_to_log_lik)(orig_p, X, y)

    vector_to_parameters(orig_p, model.parameters())
    return diag_lik_hess / n_data + diag_lik_hess
