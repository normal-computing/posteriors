from typing import Callable, Any, NamedTuple
import torch
from torch.func import jacrev

from uqlib.utils import tree_map, diag_normal_sample


class DiagLaplaceState(NamedTuple):
    """State encoding a diagonal Normal distribution over parameters.

    Args:
        mean: Mean of the variational distribution.
        prec_diag: Diagonal of the precision matrix of the Normal distribution.
    """

    mean: Any
    prec_diag: Any


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
    """Adds diagonal Fisher square-root of covariance for given batch.

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

    Returns:
        Updated DiagLaplaceState.
    """
    with torch.no_grad():
        batch_diag_score_sq = tree_map(
            lambda jac: jac.square().mean(0), jacrev(log_posterior)(state.mean, batch)
        )
    diag_prec = tree_map(lambda x, y: x + y, state.prec_diag, batch_diag_score_sq)
    return DiagLaplaceState(state.mean, diag_prec)


def sample(state: DiagLaplaceState):
    """Single sample from diagonal Normal distribution over parameters.

    Args:
        state: State encoding mean and diagonal precision.

    Returns:
        Sample from Normal distribution.
    """
    sd_diag = tree_map(lambda x: x.sqrt().reciprocal(), state.diag_prec)
    return diag_normal_sample(state.mean, sd_diag)
