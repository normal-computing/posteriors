from typing import Any, NamedTuple, Tuple
from functools import partial
import torch
from torchopt.base import GradientTransformation
from optree import tree_map, tree_map_


class SGHMCState(NamedTuple):
    """State enconding momenta for SGHMC.

    Args:
        momenta: Momenta for each parameter.
    """

    momenta: Any


def init(params: Any, momenta: Any | None = None) -> SGHMCState:
    """Initialise momenta for SGHMC.

    Args:
        params: Parameters for which to initialise.
        momenta: Initial momenta. Defaults to all zeroes.

    Returns:
        Initial SGHMCState containing momenta.
    """
    if momenta is None:
        momenta = tree_map(torch.zeros_like, params)
    return SGHMCState(momenta)


def update(
    updates: Any,
    state: SGHMCState,
    lr: float,
    alpha: float = 0.01,
    beta: float = 0.0,
    temperature: float = 1.0,
    maximize: bool = True,
    params: Any | None = None,
    inplace: bool = True,
) -> Tuple[Any, SGHMCState]:
    """Updates gradients and momenta for SGHMC.

    Args:
        updates: Gradients to update.
        state: SGHMCState containing momenta.
        lr: Learning rate.
        alpha: Friction coefficient.
        beta: Gradient noise coefficient (estimated variance).
        temperature: Temperature of the joint parameter + momenta distribution.
        maximize: Whether to maximize (ascend) or minimise (descend).
        params: Values of parameters, not used for SGHMC update.
        inplace: Whether to modify updates and state in place.

    Returns:
        Updated gradients and state (which are pointers to the inputted
        updates and state if inplace=True).
    """

    def momenta_updates(m, g):
        return (
            lr * g * (-1) ** ~maximize
            - lr * alpha * m
            + (temperature * lr * (2 * alpha - temperature * lr * beta)) ** 0.5
            * torch.randn_like(m)
        )

    if inplace:

        def transform_momenta_(m, g):
            m += momenta_updates(m, g)

        def update_(u, m):
            u.data = lr * m.data

        tree_map_(transform_momenta_, state.momenta, updates)
        tree_map_(update_, updates, state.momenta)
        return updates, state

    else:

        def transform_momenta(m, g):
            return m + momenta_updates(m, g)

        momenta = tree_map(transform_momenta, state.momenta, updates)
        updates = tree_map(lambda m: lr * m, momenta)

        return updates, SGHMCState(momenta)


def build(
    lr: float,
    alpha: float = 0.01,
    beta: float = 0.0,
    temperature: float = 1.0,
    maximize: bool = True,
    momenta: Any | None = None,
) -> GradientTransformation:
    """Builds SGHMC optimizer.

    Args:
        lr: Learning rate.
        alpha: Friction coefficient.
        beta: Gradient noise coefficient (estimated variance).
        maximize: Whether to maximize (ascend) or minimise (descend).
        momenta: Initial momenta. Defaults to all zeroes.

    Returns:
        SGHMC optimizer (torchopt.base.GradientTransformation instance).
    """
    init_fn = partial(init, momenta=momenta)
    update_fn = partial(
        update,
        lr=lr,
        alpha=alpha,
        beta=beta,
        temperature=temperature,
        maximize=maximize,
    )
    return GradientTransformation(init_fn, update_fn)
