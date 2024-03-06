from typing import Any
from dataclasses import dataclass
from functools import partial
import torch
from uqlib.types import TensorTree, Transform, LogProbFn, Tensor
from uqlib.utils import per_samplify, tree_size, empirical_fisher


@dataclass
class FullLaplaceState:
    """State encoding a Normal distribution over parameters,
    with a dense precision matrix
    Args:
        mean: Mean of the Normal distribution.
        prec: Precision matrix of the Normal distribution.
        aux: Auxiliary information from the log_posterior call.
    """

    mean: TensorTree
    prec: Tensor
    aux: Any = None


def init(
    params: TensorTree,
    init_prec: Tensor | None = None,
) -> FullLaplaceState:
    """Initialise Normal distribution over parameters,
    with a dense precision matrix.
    Args:
        params: Mean of the Normal distribution.
        init_prec: Initial precision matrix. Defaults to identity.
    Returns:
        Initial FullLaplaceState.
    """

    if init_prec is None:
        num_params = tree_size(params)
        init_prec = torch.zeros((num_params, num_params), requires_grad=False)

    return FullLaplaceState(params, init_prec)


def update(
    state: FullLaplaceState,
    batch: Any,
    log_posterior: LogProbFn,
    per_sample: bool = False,
    inplace: bool = True,
) -> FullLaplaceState:
    """Adds empirical Fisher information matrix of covariance summed over
    given batch.

    Args:
        state: Current state.
        batch: Input data to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
        per_sample: If True, then log_posterior is assumed to return a vector of
        log posteriors for each sample in the batch. If False, then log_posterior
        is assumed to return a scalar log posterior for the whole batch, in this
        case torch.func.vmap will be called, this is typically slower than
        directly writing log_posterior to be per sample.
        inplace: If True, the state is updated in place. Otherwise, a new
            state is returned.

    Returns:
        Updated FullLaplaceState.
    """
    if not per_sample:
        log_posterior = per_samplify(log_posterior)

    fisher, aux = empirical_fisher(log_posterior, state.mean, batch)

    if inplace:
        state.prec += fisher
        state.aux = aux
        return state
    else:
        return FullLaplaceState(state.mean, state.prec + fisher, aux)


def build(
    log_posterior: LogProbFn,
    per_sample: bool = False,
    init_prec: Tensor | None = None,
) -> Transform:
    """Builds a transform for dense empirical Fisher information
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
        init_prec_diag: Initial diagonal of the precision matrix. Defaults to zero.

    Returns:
        Diagonal empirical Fisher information Laplace approximation transform
        (uqlib.types.Transform instance).
    """
    init_fn = partial(init, init_prec=init_prec)
    update_fn = partial(update, log_posterior=log_posterior, per_sample=per_sample)
    return Transform(init_fn, update_fn)


def sample(
    state: FullLaplaceState,
    sample_shape: torch.Size = torch.Size([]),
) -> Tensor:
    """Sample from Normal distribution over parameters.

    Args:
        state: State encoding mean and precision matrix.
        sample_shapa: Shape of the desired samples.

    Returns:
        Sample(s) from the Normal distribution.
    """
    return torch.distributions.MultivariateNormal(
        loc=state.mean, precision_matrix=state.prec
    ).sample(sample_shape)
