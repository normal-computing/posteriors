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
    """Adds diagonal square-root of inverse negative Hessian for given batch.

    At the MAP estimate, the inverse negative Hessian approximates the covariance, but
    it needs to be semipositive-definite which is not guaranteed.

    log_posterior expects to take parameters and input batch and return the scalar log
    posterior:

    ```
    val = log_posterior(params, batch)
    ```

    Args:
        state: Current state.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised).
        batch: Input data to log_posterior.
        epsilon: Minimum value of the negative diagonal Hessian. Defaults to 0.

    Returns:
        Updated DiagLaplaceState.
    """
    with torch.no_grad():
        batch_diag_hess = hessian_diag(lambda x: log_posterior(x, batch).mean())(
            state.mean
        )
    batch_prec_diag = tree_map(lambda x, y: x + y, state.prec_diag, batch_diag_hess)
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
