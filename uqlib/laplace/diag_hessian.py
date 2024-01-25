from functools import partial
from typing import Callable, Any
import torch
from optree import tree_map, tree_flatten

from uqlib.types import TensorTree, Transform
from uqlib.utils import hessian_diag, diag_normal_sample, flexi_tree_map
from uqlib.laplace.diag_fisher import DiagLaplaceState


def init(
    params: TensorTree,
    init_prec_diag: TensorTree | None = None,
) -> DiagLaplaceState:
    """Initialise diagonal Normal distribution over parameters.

    Args:
        params: Initial mean of the variational distribution.
        init_prec_diag: Initial diagonal of the precision matrix. Defaults to zero.

    Returns:
        Initial DiagVIState.
    """
    if init_prec_diag is None:
        init_prec_diag = tree_map(lambda x: torch.zeros_like(x), params)
    return DiagLaplaceState(params, init_prec_diag)


def update(
    state: DiagLaplaceState,
    batch: Any,
    log_posterior: Callable[[TensorTree, Any], float],
    inplace: bool = True,
) -> DiagLaplaceState:
    """Adds diagonal negative Hessian summed across given batch.

    At the MAP estimate, the negative Hessian approximates the precision matrix, but
    note it needs to be semi-positive-definite which is not guaranteed.

    Args:
        state: Current state.
        batch: Input data to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised) for each batch member.
        inplace: If True, then state is updated in place, otherwise a new state is
            returned.

    Returns:
        Updated DiagLaplaceState.
    """
    batch_size = len(tree_flatten(batch)[0][0])

    with torch.no_grad():
        batch_diag_hess = hessian_diag(lambda x: log_posterior(x, batch))(state.mean)

    def update_func(x, y):
        return x - y * batch_size

    prec_diag = flexi_tree_map(
        update_func, state.prec_diag, batch_diag_hess, inplace=inplace
    )

    return DiagLaplaceState(state.mean, prec_diag)


def build(
    log_posterior: Callable[[TensorTree, Any], float],
    init_prec_diag: TensorTree | None = None,
) -> Transform:
    """Builds a transform for diagonal Hessian Laplace approximation.

    Args:
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised).
        per_sample: If True, then log_posterior is assumed to return a vector of
            log posteriors for each sample in the batch. If False, then log_posterior
            is assumed to return a scalar log posterior for the whole batch, in this
            case torch.func.vmap will be called, this is typically slower than
            directly writing log_posterior to be per sample.
        init_prec_diag: Initial diagonal of the precision matrix. Defaults to zero.

    Returns:
        Diagonal Hessian Laplace approximation transform
        (uqlib.types.Transform instance).
    """
    init_fn = partial(init, init_prec_diag=init_prec_diag)
    update_fn = partial(update, log_posterior=log_posterior)
    return Transform(init_fn, update_fn)


def sample(
    state: DiagLaplaceState, sample_shape: torch.Size = torch.Size([])
) -> TensorTree:
    """Sample from diagonal Normal distribution over parameters.

    Args:
        state: State encoding mean and diagonal precision.

    Returns:
        Sample(s) from Normal distribution.
    """
    sd_diag = tree_map(lambda x: x.sqrt().reciprocal(), state.prec_diag)
    return diag_normal_sample(state.mean, sd_diag, sample_shape=sample_shape)
