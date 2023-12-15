from typing import Callable, Any
import torch

from uqlib.utils import hessian_diag, tree_map, diag_normal_sample
from uqlib.laplace.diag_fisher import DiagLaplaceState


def init(
    init_mean: Any,
    init_prec_diag: Any = None,
) -> DiagLaplaceState:
    """Initialise diagonal Normal distribution over parameters.

    Args:
        init_mean: Initial mean of the variational distribution.
        init_prec_diag: Initial diagonal of the precision matrix. Defaults to zero.

    Returns:
        Initial DiagVIState.
    """
    if init_prec_diag is None:
        init_prec_diag = tree_map(
            lambda x: torch.zeros_like(x, requires_grad=False), init_mean
        )
    return DiagLaplaceState(init_mean, init_prec_diag)


def update(
    state: DiagLaplaceState,
    log_posterior: Callable[[Any, Any], float],
    batch: Any,
) -> DiagLaplaceState:
    """Adds diagonal negative Hessian summed across given batch.

    At the MAP estimate, the inverse negative Hessian approximates the covariance, but
    it needs to be semipositive-definite which is not guaranteed.

    log_posterior expects to take parameters and input batch and return a tensor
    containing log posterior evaluations for each batch member:

    ```
    batch_vals = log_posterior(params, batch)
    ```

    where each element of batch_vals is an unbiased estimate of the log posterior.
    I.e. batch_vals.mean() is an unbiased estimate of the log posterior.

    Args:
        state: Current state.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised) for each batch member.
        batch: Input data to log_posterior.

    Returns:
        Updated DiagLaplaceState.
    """
    with torch.no_grad():
        batch_diag_hess = hessian_diag(lambda x: log_posterior(x, batch).sum())(
            state.mean
        )
    batch_prec_diag = tree_map(lambda x, y: x - y, state.prec_diag, batch_diag_hess)
    return DiagLaplaceState(state.mean, batch_prec_diag)


def sample(state: DiagLaplaceState):
    """Single sample from diagonal Normal distribution over parameters.

    Args:
        state: State encoding mean and diagonal precision.

    Returns:
        Sample from Normal distribution.
    """
    sd_diag = tree_map(lambda x: x.sqrt().reciprocal(), state.diag_prec)
    return diag_normal_sample(state.mean, sd_diag)
