from typing import NamedTuple, Any
from functools import partial
import torch
from torch.func import grad_and_value
from optree import tree_map

from uqlib.types import TensorTree, Transform, LogProbFn
from uqlib.utils import flexi_tree_map


class SGHMCState(NamedTuple):
    """State enconding momenta for SGHMC.

    Args:
        params: Parameters.
        momenta: Momenta for each parameter.
        log_posterior: Log posterior evaluation.
        aux: Auxiliary information from the log_posterior call.
    """

    params: TensorTree
    momenta: TensorTree
    log_posterior: torch.tensor = torch.tensor(0.0)
    aux: Any = None


def init(params: TensorTree, momenta: TensorTree | None = None) -> SGHMCState:
    """Initialise momenta for SGHMC.

    Args:
        params: Parameters for which to initialise.
        momenta: Initial momenta. Defaults to all zeroes.

    Returns:
        Initial SGHMCState containing momenta.
    """
    if momenta is None:
        momenta = tree_map(torch.zeros_like, params)
    return SGHMCState(params, momenta)


def update(
    state: SGHMCState,
    batch: Any,
    log_posterior: LogProbFn,
    lr: float,
    alpha: float = 0.01,
    beta: float = 0.0,
    temperature: float = 1.0,
    inplace: bool = True,
) -> SGHMCState:
    """Updates parameters and momenta for SGHMC.

    Args:
        state: SGHMCState containing params and momenta.
        batch: Data batch to be send to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        lr: Learning rate.
        alpha: Friction coefficient.
        beta: Gradient noise coefficient (estimated variance).
        temperature: Temperature of the joint parameter + momenta distribution.
        params: Values of parameters, not used for SGHMC update.
        inplace: Whether to modify updates and state in place.

    Returns:
        Updated state
        (which are pointers to the inputted state tensors if inplace=True).
    """

    grads, (log_post, aux) = grad_and_value(log_posterior, has_aux=True)(
        state.params, batch
    )

    def transform_params(p, m):
        return p + lr * m

    def transform_momenta(m, g):
        return (
            m
            + lr * g
            - lr * alpha * m
            + (temperature * lr * (2 * alpha - temperature * lr * beta)) ** 0.5
            * torch.randn_like(m)
        )

    params = flexi_tree_map(
        transform_params, state.params, state.momenta, inplace=inplace
    )
    momenta = flexi_tree_map(transform_momenta, state.momenta, grads, inplace=inplace)

    return SGHMCState(params, momenta, log_post.detach(), aux)


def build(
    log_posterior: LogProbFn,
    lr: float,
    alpha: float = 0.01,
    beta: float = 0.0,
    temperature: float = 1.0,
    momenta: TensorTree | None = None,
) -> Transform:
    """Builds SGHMC transform.

    Args:
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        lr: Learning rate.
        alpha: Friction coefficient.
        beta: Gradient noise coefficient (estimated variance).
        temperature: Temperature of the joint parameter + momenta distribution.
        momenta: Initial momenta. Defaults to all zeroes.

    Returns:
        SGHMC transform (uqlib.types.Transform instance).
    """
    init_fn = partial(init, momenta=momenta)
    update_fn = partial(
        update,
        log_posterior=log_posterior,
        lr=lr,
        alpha=alpha,
        beta=beta,
        temperature=temperature,
    )
    return Transform(init_fn, update_fn)
