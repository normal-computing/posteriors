from typing import NamedTuple, Tuple, Any, Callable
from functools import partial
import torch
from torch.func import grad_and_value
from optree import tree_map, tree_map_

from uqlib.types import TensorTree, Transform
from uqlib.utils import inplacify


class SGHMCState(NamedTuple):
    """State enconding momenta for SGHMC.

    Args:
        params: Parameters.
        momenta: Momenta for each parameter.
        log_posterior: Log posterior evaluation.
    """

    params: TensorTree
    momenta: TensorTree
    log_posterior: float = 0.0


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
    log_posterior: Callable[[TensorTree, Any], float],
    lr: float,
    alpha: float = 0.01,
    beta: float = 0.0,
    temperature: float = 1.0,
    inplace: bool = True,
) -> Tuple[TensorTree, SGHMCState]:
    """Updates parameters and momenta for SGHMC.

    Args:
        state: SGHMCState containing params and momenta.
        batch: Data batch to be send to log_posterior.
        log_posterior: Function taking parameters and batch
            returning the log posterior evaluation.
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

    grads, log_post = grad_and_value(log_posterior)(state.params, batch)

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

    if inplace:
        params = tree_map_(inplacify(transform_params), state.params, state.momenta)
        momenta = tree_map_(inplacify(transform_momenta), state.momenta, grads)

    else:
        params = tree_map(transform_params, state.params, state.momenta)
        momenta = tree_map(transform_momenta, state.momenta, grads)

    return SGHMCState(params, momenta, log_post.item())


def build(
    log_posterior: Callable[[TensorTree, Any], float],
    lr: float,
    alpha: float = 0.01,
    beta: float = 0.0,
    temperature: float = 1.0,
    momenta: TensorTree | None = None,
) -> Transform:
    """Builds SGHMC transform.

    Args:
        log_posterior: Function taking parameters and batch
            returning the log posterior evaluation.
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
