from typing import Callable

import torch
from torch.utils.data import DataLoader

from uqlib.utils import diagonal_hessian, dict_map, model_to_function


def fit_diagonal_hessian(
    model: torch.nn.Module,
    log_prior: Callable,
    log_likelihood: Callable,
    train_dataloader: DataLoader,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Fit diagonal Hessian to data. Rescales by dividing by the number of data points
    in train_dataloader and also negates to give a valid (diagonal) precision matrix.

    Args:
        model: Model with parameters trained to MAP estimator.
        log_prior: Function that takes a dictionary of parameters and
            returns the log prior (which can be unnormalised).
        log_likelihood: Function that takes a batch of output data
            and outputs from the model and returns the log likelihood
            (which can be unnormalised).
            i.e. log_lik = log_likelihood(y, model(X))
        train_dataloader: The training data. Iterable that supplies batches in the form
            of (x, y) tuples.
        epsilon: Minimum value of the diagonal Hessian. Defaults to 0.

    Returns:
        The fitted diagonal Hessian divided by the number of data points
        in train_dataloader and also negated (to correspond to positive precision).
    """
    model_func = model_to_function(model)
    orig_p = dict(model.named_parameters())

    def p_to_log_lik(p, x, y):
        predictions = model_func(p, x)
        return torch.mean(log_likelihood(y, predictions))

    diag_prior_hess = diagonal_hessian(log_prior)(orig_p)
    diag_lik_hess = dict_map(lambda x: torch.zeros_like(x), diag_prior_hess)

    n_data = 0

    for x, y in train_dataloader:
        n_data += x.shape[0]
        diag_lik_hess = dict_map(
            lambda x, y: x + y,
            diag_lik_hess,
            diagonal_hessian(p_to_log_lik)(orig_p, x, y),
        )

    diag_hess = dict_map(lambda x, y: x / n_data + y, diag_prior_hess, diag_lik_hess)
    diag_hess = dict_map(
        lambda x: torch.where(-x > epsilon, -x, torch.zeros_like(x)), diag_hess
    )

    return diag_hess
