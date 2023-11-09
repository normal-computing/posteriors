from typing import Callable

import torch
from torch.utils.data import DataLoader
from torch.func import functional_call

from uqlib.utils import diagonal_hessian


def add_similar_dicts(d1, d2):
    return {k: d1[k] + d2[k] for k in d1.keys()}


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
        in train_dataloader and also negated (to correspond to positive precision).
    """

    orig_p = dict(model.named_parameters())

    def p_to_log_lik(p, x, y):
        predictions = functional_call(model, p, (x,))
        return torch.mean(log_likelihood(y, predictions))

    diag_prior_hess = diagonal_hessian(log_prior)(orig_p)
    diag_lik_hess = {k: torch.zeros_like(v) for k, v in diag_prior_hess.items()}

    n_data = 0

    for x, y in train_dataloader:
        n_data += x.shape[0]
        diag_lik_hess = add_similar_dicts(
            diag_lik_hess, diagonal_hessian(p_to_log_lik)(orig_p, x, y)
        )

    diag_prior_hess = {k: v / n_data for k, v in diag_prior_hess.items()}

    diag_hess = add_similar_dicts(diag_prior_hess, diag_lik_hess)
    diag_hess = {
        k: torch.where(-v > epsilon, -v, torch.zeros_like(v))
        for k, v in diag_hess.items()
    }
    return diag_hess
