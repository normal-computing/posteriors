from typing import Callable

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from uqlib.utils import diagonal_hessian


def fit_diagonal_hessian(
    model: torch.nn.Module,
    log_prior: Callable,
    log_likelihood: Callable,
    train_dataloader: DataLoader,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Fit diagonal Hessian to data. Rescales by dividing by the number of data points
    in train_dataloader.

    Args:
        model: Model with parameters trained to MAP estimator.
        log_prior: Function that takes a vector of parameters and returns the log prior
            (which can be unnormalised).
        log_likelihood: Function that takes a batch of output data
            and output data from the model and returns the log likelihood
            (which can be unnormalised).
            log_lik = log_likelihood(y, model(X))
        train_dataloader: The training data. Iterable that supplies batches in the form
            of (x, y) tuples.
        epsilon: Minimum value of the diagonal Hessian. Defaults to 0.

    Returns:
        The fitted diagonal Hessian divided by the number of data points
        in train_dataloader.
    """
    orig_p = parameters_to_vector(model.parameters())

    def p_to_log_lik(p, x, y):
        vector_to_parameters(p, model.parameters())
        return torch.mean(log_likelihood(y, model(x)))

    diag_prior_hess = diagonal_hessian(log_prior)(orig_p)
    diag_lik_hess = torch.zeros_like(diag_prior_hess)

    n_data = 0

    for x, y in train_dataloader:
        n_data += x.shape[0]
        diag_lik_hess += diagonal_hessian(p_to_log_lik)(orig_p, x, y)

    diag_hess = diag_prior_hess / n_data + diag_lik_hess
    vector_to_parameters(orig_p, model.parameters())
    return torch.where(diag_hess > epsilon, diag_hess, torch.zeros_like(diag_hess))
