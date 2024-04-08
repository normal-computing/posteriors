from typing import Any
from functools import partial
import torch
from torch.func import grad_and_value
from dataclasses import dataclass

from posteriors.types import TensorTree, Transform, LogProbFn, TransformState
from posteriors.utils import flexi_tree_map, CatchAuxError


@dataclass
class SGLDState(TransformState):
    """State encoding params for SGLD.

    Args:
        params: Parameters.
        log_posterior: Log posterior evaluation.
        aux: Auxiliary information from the log_posterior call.
    """

    params: TensorTree
    log_posterior: torch.tensor = None
    aux: Any = None


def init(params: TensorTree) -> SGLDState:
    """Initialise SGLD.

    Args:
        params: Parameters for which to initialise.

    Returns:
        Initial SGLDState.
    """

    return SGLDState(params)


def update(
    state: SGLDState,
    batch: Any,
    log_posterior: LogProbFn,
    lr: float,
    beta: float = 0.0,
    temperature: float = 1.0,
    inplace: bool = False,
) -> SGLDState:
    """Updates parameters for SGLD.

    SGLD update rule:
        θ = θ + lr * ∇ log p(θ, batch) + ε,
            ε ~ N(0, lr * (2 - lr * β) * temperature)

    Args:
        state: SGLDState containing params.
        batch: Data batch to be send to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        lr: Learning rate.
        beta: Gradient noise coefficient (estimated variance).
        temperature: Temperature of the sampling distribution.
        inplace: Whether to modify state in place.

    Returns:
        Updated state (which are pointers to the input state tensors if inplace=True).
    """
    with torch.no_grad(), CatchAuxError():
        grads, (log_post, aux) = grad_and_value(log_posterior, has_aux=True)(
            state.params, batch
        )

    def transform_params(p, g):
        return (
            p
            + lr * g
            + (temperature * lr * (2 - temperature * lr * beta)) ** 0.5
            * torch.randn_like(p)
        )

    params = flexi_tree_map(transform_params, state.params, grads, inplace=inplace)

    if inplace:
        state.log_posterior = log_post.detach()
        state.aux = aux
        return state
    return SGLDState(params, log_post.detach(), aux)


def build(
    log_posterior: LogProbFn,
    lr: float,
    beta: float = 0.0,
    temperature: float = 1.0,
) -> Transform:
    """Builds SGLD transform.

    Args:
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        lr: Learning rate.
        beta: Gradient noise coefficient (estimated variance).
        temperature: Temperature of the sampling distribution.

    Returns:
        SGLD transform (posteriors.types.Transform instance).
    """
    update_fn = partial(
        update,
        log_posterior=log_posterior,
        lr=lr,
        beta=beta,
        temperature=temperature,
    )
    return Transform(init, update_fn)
