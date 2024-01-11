from functools import partial
from typing import Callable, Any, NamedTuple
import torch
from torch.func import jacrev, vmap
from optree import tree_map, tree_map_

from uqlib.utils import diag_normal_sample


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
        init_prec_diag = tree_map(lambda x: torch.zeros_like(x), init_mean)
    return DiagLaplaceState(init_mean, init_prec_diag)


def update(
    state: DiagLaplaceState,
    log_posterior: Callable[[Any, Any], float],
    batch: Any,
    per_sample: bool = False,
    respect_requires_grad: bool = True,
) -> DiagLaplaceState:
    """Adds diagonal empirical Fisher information matrix of covariance summed over
    given batch.

    log_posterior expects to take parameters and input batch and return a scalar
    unbiased estimate of the full batch log posterior:

    ```
    log_posterior_eval = log_posterior(params, batch)
    ```

    Args:
        state: Current state.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised).
        batch: Input data to log_posterior.
        per_sample: If True, then log_posterior is assumed to return a vector of
            log posteriors for each sample in the batch. If False, then log_posterior
            is assumed to return a scalar log posterior for the whole batch, in this
            case torch.func.vmap will be called, this is typically slower than
            directly writing log_posterior to be per sample.
        respect_requires_grad: If True, then the diagonal of the Fisher information
            matrix will be set to zero for parameters that do not require gradients.

    Returns:
        Updated DiagLaplaceState.
    """

    if per_sample:
        log_posterior_per_sample = log_posterior
    else:
        # per-sample gradients following https://pytorch.org/tutorials/intermediate/per_sample_grads.html
        @partial(vmap, in_dims=(None, 0))
        def log_posterior_per_sample(params, batch):
            batch = tree_map(lambda x: x.unsqueeze(0), batch)
            return log_posterior(params, batch)

    with torch.no_grad():
        batch_diag_score_sq = tree_map(
            lambda jac: jac.square().sum(0),
            jacrev(log_posterior_per_sample)(state.mean, batch),
        )
    prec_diag = tree_map(lambda x, y: x + y, state.prec_diag, batch_diag_score_sq)

    if respect_requires_grad:

        def zero_if_not_requires_grad(x, p):
            if not p.requires_grad:
                x *= 0

        tree_map_(zero_if_not_requires_grad, prec_diag, state.mean)

    return DiagLaplaceState(state.mean, prec_diag)


def sample(state: DiagLaplaceState, sample_shape: torch.Size = torch.Size([])):
    """Single sample from diagonal Normal distribution over parameters.

    Args:
        state: State encoding mean and diagonal precision.

    Returns:
        Sample from Normal distribution.
    """
    sd_diag = tree_map(lambda x: x.sqrt().reciprocal(), state.prec_diag)
    return diag_normal_sample(state.mean, sd_diag, sample_shape=sample_shape)
