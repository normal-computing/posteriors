from typing import Any
from functools import partial
import torch
from torch.func import grad_and_value
from optree import tree_map

from uqlib.types import TensorTree, Transform, LogProbFn
from uqlib.utils import diag_normal_sample, hessian_diag, flexi_tree_map
from uqlib.ekf.diag_fisher import EKFDiagState


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
    inplace: bool = True,
) -> EKFDiagState:
    """Applies an extended Kalman Filter update to the diagonal Normal distribution.
    The update is first order, i.e. the likelihood is approximated by a

    log p(y | x, p) ≈ log p(y | x, μ) + lr * g(μ)ᵀ(p - μ)
        + lr * 1/2 (p - μ)ᵀ H_d(μ) (p - μ) T⁻¹

    where μ is the mean of the variational distribution, lr is the learning rate
    (likelihood inverse temperature), whilst g(μ) is the gradient and H_d(μ) the
    diagonal Hessian of the log-likelihood with respect to the parameters.

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
        inplace: Whether to update the state parameters in-place.

    Returns:
        Updated EKFDiagState.
    """
    predict_sd_diag = flexi_tree_map(
        lambda x: (x**2 + transition_sd**2) ** 0.5, state.sd_diag, inplace=inplace
    )
    with torch.no_grad():
        grad, (log_lik, aux) = grad_and_value(log_likelihood, has_aux=True)(
            state.mean, batch
        )
        diag_hessian, _ = hessian_diag(log_likelihood, has_aux=True)(state.mean, batch)

    update_sd_diag = flexi_tree_map(
        lambda sig, h: (sig**-2 - lr * h) ** -0.5,
        predict_sd_diag,
        diag_hessian,
        inplace=inplace,
    )
    update_mean = flexi_tree_map(
        lambda mu, sig, g: mu + sig**2 * lr * g,
        state.mean,
        update_sd_diag,
        grad,
        inplace=inplace,
    )
    return EKFDiagState(update_mean, update_sd_diag, log_lik.item(), aux)


def build(
    log_likelihood: LogProbFn,
    lr: float,
    transition_sd: float = 0.0,
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
    )
    return Transform(init_fn, update_fn)


def sample(state: EKFDiagState, sample_shape: torch.Size = torch.Size([])):
    """Single sample from diagonal Normal distribution over parameters.

    Args:
        state: State encoding mean and standard deviations.

    Returns:
        Sample from Normal distribution.
    """
    return diag_normal_sample(state.mean, state.sd_diag, sample_shape=sample_shape)
