from functools import partial
from typing import Callable, Any, NamedTuple
import torch
from torch.func import jacrev, vmap
from optree import tree_map

from uqlib.types import TensorTree, Transform
from uqlib.utils import diag_normal_sample, flexi_tree_map


class DiagLaplaceState(NamedTuple):
    """State encoding a diagonal Normal distribution over parameters.

    Args:
        mean: Mean of the variational distribution.
        prec_diag: Diagonal of the precision matrix of the Normal distribution.
    """

    mean: TensorTree
    prec_diag: TensorTree


def init(
    params: TensorTree,
    init_prec_diag: TensorTree | None = None,
) -> DiagLaplaceState:
    """Initialise diagonal Normal distribution over parameters.

    Args:
        params: Mean of the Normal distribution.
        init_prec_diag: Initial diagonal of the precision matrix. Defaults to zero.

    Returns:
        Initial DiagVIState.
    """
    if init_prec_diag is None:
        init_prec_diag = tree_map(
            lambda x: torch.zeros_like(x, requires_grad=x.requires_grad), params
        )

    return DiagLaplaceState(params, init_prec_diag)


def update(
    state: DiagLaplaceState,
    batch: Any,
    log_posterior: Callable[[TensorTree, Any], float],
    per_sample: bool = False,
    inplace: bool = True,
) -> DiagLaplaceState:
    """Adds diagonal empirical Fisher information matrix of covariance summed over
    given batch.

    Args:
        state: Current state.
        batch: Input data to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised).
        per_sample: If True, then log_posterior is assumed to return a vector of
            log posteriors for each sample in the batch. If False, then log_posterior
            is assumed to return a scalar log posterior for the whole batch, in this
            case torch.func.vmap will be called, this is typically slower than
            directly writing log_posterior to be per sample.
        inplace: If True, then the state is updated in place, otherwise a new state
            is returned.

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

    def update_func(x, y):
        return x + y

    prec_diag = flexi_tree_map(
        update_func, state.prec_diag, batch_diag_score_sq, inplace=inplace
    )

    return DiagLaplaceState(state.mean, prec_diag)


def build(
    log_posterior: Callable[[TensorTree, Any], float],
    per_sample: bool = False,
    init_prec_diag: TensorTree | None = None,
) -> Transform:
    """Builds a transform for diagonal empirical Fisher information
    Laplace approximation.

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
        Diagonal empirical Fisher information Laplace approximation transform
        (uqlib.types.Transform instance).
    """
    init_fn = partial(init, init_prec_diag=init_prec_diag)
    update_fn = partial(update, log_posterior=log_posterior, per_sample=per_sample)
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
