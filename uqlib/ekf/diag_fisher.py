from typing import Any
from functools import partial
import torch
from torch.func import jacrev
from optree import tree_map
from dataclasses import dataclass

from uqlib.types import TensorTree, Transform, LogProbFn
from uqlib.utils import diag_normal_sample, flexi_tree_map, per_samplify


@dataclass
class EKFDiagState:
    """State encoding a diagonal Normal distribution over parameters.

    Args:
        params: Mean of the Normal distribution.
        sd_diag: Square-root diagonal of the covariance matrix of the
            Normal distribution.
        log_likelihood: Log likelihood of the data given the parameters.
        aux: Auxiliary information from the log_likelihood call.
    """

    params: TensorTree
    sd_diag: TensorTree
    log_likelihood: float = 0
    aux: Any = None


def init(
    params: TensorTree,
    init_sds: TensorTree | None = None,
) -> EKFDiagState:
    """Initialise diagonal Normal distribution over parameters.

    Args:
        params: Initial mean of the variational distribution.
        init_sds: Initial square-root diagonal of the covariance matrix
            of the variational distribution. Defaults to ones.

    Returns:
        Initial EKFDiagState.
    """
    if init_sds is None:
        init_sds = tree_map(
            lambda x: torch.ones_like(x, requires_grad=x.requires_grad), params
        )

    return EKFDiagState(params, init_sds)


def update(
    state: EKFDiagState,
    batch: Any,
    log_likelihood: LogProbFn,
    lr: float,
    transition_sd: float = 0.0,
    per_sample: bool = False,
    inplace: bool = False,
) -> EKFDiagState:
    """Applies an extended Kalman Filter update to the diagonal Normal distribution.
    The update is first order, i.e. the likelihood is approximated by a

    log p(y | x, p) ≈ log p(y | x, μ) + lr * g(μ)ᵀ(p - μ)
        + lr * 1/2 (p - μ)ᵀ F_d(μ) (p - μ) T⁻¹

    where μ is the mean of the variational distribution, lr is the learning rate
    (likelihood inverse temperature), whilst g(μ) is the gradient and F_d(μ) the
    negative diagonal empirical Fisher of the log-likelihood with respect to the
    parameters.

    Args:
        state: Current state.
        batch: Input data to log_likelihood.
        log_likelihood: Function that takes parameters and input batch and
            returns the log-likelihood value as well as auxiliary information,
            e.g. from the model call.
        lr: Inverse temperature of the update, which behaves like a learning rate.
            see https://arxiv.org/abs/1703.00209 for details.
        transition_sd: Standard deviation of the transition noise, to additively
            inflate the diagonal covariance before the update. Defaults to zero.
        per_sample: If True, then log_likelihood is assumed to return a vector of
            log likelihoods for each sample in the batch. If False, then log_likelihood
            is assumed to return a scalar log likelihood for the whole batch, in this
            case torch.func.vmap will be called, this is typically slower than
            directly writing log_likelihood to be per sample.
        inplace: Whether to update the state parameters in-place.

    Returns:
        Updated EKFDiagState.
    """

    if not per_sample:
        log_likelihood = per_samplify(log_likelihood)

    predict_sd_diag = flexi_tree_map(
        lambda x: (x**2 + transition_sd**2) ** 0.5, state.sd_diag, inplace=inplace
    )
    with torch.no_grad():
        log_liks, aux = log_likelihood(state.params, batch)
        jac, _ = jacrev(log_likelihood, has_aux=True)(state.params, batch)
        grad = tree_map(lambda x: x.mean(0), jac)
        diag_lik_hessian_approx = tree_map(lambda x: -(x**2).mean(0), jac)

    update_sd_diag = flexi_tree_map(
        lambda sig, h: (sig**-2 - lr * h) ** -0.5,
        predict_sd_diag,
        diag_lik_hessian_approx,
        inplace=inplace,
    )
    update_mean = flexi_tree_map(
        lambda mu, sig, g: mu + sig**2 * lr * g,
        state.params,
        update_sd_diag,
        grad,
        inplace=inplace,
    )

    if inplace:
        state.log_likelihood = log_liks.mean().detach()
        state.aux = aux
        return state
    return EKFDiagState(update_mean, update_sd_diag, log_liks.mean().detach(), aux)


def build(
    log_likelihood: LogProbFn,
    lr: float,
    transition_sd: float = 0.0,
    per_sample: bool = False,
    init_sds: TensorTree | None = None,
) -> Transform:
    """Builds a transform for variational inference with a diagonal Normal
    distribution over parameters.

    Args:
        log_likelihood: Function that takes parameters and input batch and
            returns the log-likelihood value as well as auxiliary information,
            e.g. from the model call.
        lr: Inverse temperature of the update, which behaves like a learning rate.
            see https://arxiv.org/abs/1703.00209 for details.
        transition_sd: Standard deviation of the transition noise, to additively
            inflate the diagonal covariance before the update. Defaults to zero.
        per_sample: If True, then log_likelihood is assumed to return a vector of
            log likelihoods for each sample in the batch. If False, then log_likelihood
            is assumed to return a scalar log likelihood for the whole batch, in this
            case torch.func.vmap will be called, this is typically slower than
            directly writing log_likelihood to be per sample.
        init_sds: Initial square-root diagonal of the covariance matrix
            of the variational distribution. Defaults to ones.

    Returns:
        Diagonal EKF transform (uqlib.types.Transform instance).
    """
    init_fn = partial(init, init_sds=init_sds)
    update_fn = partial(
        update,
        log_likelihood=log_likelihood,
        lr=lr,
        transition_sd=transition_sd,
        per_sample=per_sample,
    )
    return Transform(init_fn, update_fn)


def sample(
    state: EKFDiagState, sample_shape: torch.Size = torch.Size([])
) -> TensorTree:
    """Single sample from diagonal Normal distribution over parameters.

    Args:
        state: State encoding mean and standard deviations.
        sample_shape: Shape of the desired samples.

    Returns:
        Sample from Normal distribution.
    """
    return diag_normal_sample(state.params, state.sd_diag, sample_shape=sample_shape)
