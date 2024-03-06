from functools import partial
from typing import Any
import torch
from torch.func import jacrev
from optree import tree_map
from dataclasses import dataclass

from uqlib.types import TensorTree, Transform, LogProbFn
from uqlib.utils import diag_normal_sample, flexi_tree_map, per_samplify, is_scalar


@dataclass
class DiagLaplaceState:
    """State encoding a diagonal Normal distribution over parameters.

    Args:
        params: Mean of the Normal distribution.
        prec_diag: Diagonal of the precision matrix of the Normal distribution.
        aux: Auxiliary information from the log_posterior call.
    """

    params: TensorTree
    prec_diag: TensorTree
    aux: Any = None


def init(
    params: TensorTree,
    init_prec_diag: TensorTree | float = 0.0,
) -> DiagLaplaceState:
    """Initialise diagonal Normal distribution over parameters.

    Args:
        params: Mean of the Normal distribution.
        init_prec_diag: Initial diagonal precision matrix.
            Can be tree like params or scalar. Defaults to zero.

    Returns:
        Initial DiagLaplaceState.
    """
    if is_scalar(init_prec_diag):
        init_prec_diag = tree_map(
            lambda x: torch.full_like(x, init_prec_diag, requires_grad=x.requires_grad),
            params,
        )

    return DiagLaplaceState(params, init_prec_diag)


def update(
    state: DiagLaplaceState,
    batch: Any,
    log_posterior: LogProbFn,
    per_sample: bool = False,
    inplace: bool = False,
) -> DiagLaplaceState:
    """Adds diagonal empirical Fisher information matrix of covariance summed over
    given batch.

    Args:
        state: Current state.
        batch: Input data to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
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
    if not per_sample:
        log_posterior = per_samplify(log_posterior)

    with torch.no_grad():
        jac, aux = jacrev(log_posterior, has_aux=True)(state.params, batch)
        batch_diag_score_sq = tree_map(lambda j: j.square().sum(0), jac)

    def update_func(x, y):
        return x + y

    prec_diag = flexi_tree_map(
        update_func, state.prec_diag, batch_diag_score_sq, inplace=inplace
    )

    if inplace:
        state.aux = aux
        return state
    return DiagLaplaceState(state.params, prec_diag, aux)


def build(
    log_posterior: LogProbFn,
    per_sample: bool = False,
    init_prec_diag: TensorTree | float = 0.0,
) -> Transform:
    """Builds a transform for diagonal empirical Fisher information
    Laplace approximation.

    Args:
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        per_sample: If True, then log_posterior is assumed to return a vector of
            log posteriors for each sample in the batch. If False, then log_posterior
            is assumed to return a scalar log posterior for the whole batch, in this
            case torch.func.vmap will be called, this is typically slower than
            directly writing log_posterior to be per sample.
        init_prec_diag: Initial diagonal precision matrix.
            Can be tree like params or scalar. Defaults to zero.

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
        sample_shape: Shape of the desired samples.

    Returns:
        Sample(s) from Normal distribution.
    """
    sd_diag = tree_map(lambda x: x.sqrt().reciprocal(), state.prec_diag)
    return diag_normal_sample(state.params, sd_diag, sample_shape=sample_shape)
