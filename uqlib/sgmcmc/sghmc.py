from typing import Any, NamedTuple, Tuple
from functools import partial
import torch
from torchopt.base import GradientTransformation
from optree import tree_map, tree_map_


class SGHMCState(NamedTuple):
    momenta: Any


def init(params: Any, momenta: Any | None = None) -> SGHMCState:
    if momenta is None:
        momenta = tree_map(torch.zeros_like, params)
    return SGHMCState(momenta)


def update(
    updates: Any,
    state: SGHMCState,
    lr: float,
    alpha: float = 0.01,
    beta: float = 0.0,
    params: Any | None = None,
    inplace: bool = True,
) -> Tuple[Any, SGHMCState]:
    def momenta_updates(m, g):
        return (
            lr * g
            - lr * alpha * m
            + (lr * (2 * alpha - lr * beta)) ** 0.5 * torch.randn_like(m)
        )

    if inplace:

        def transform_momenta_(m, g):
            m += momenta_updates(m, g)

        tree_map_(transform_momenta_, state.momenta, updates)
        tree_map_(lambda _, m: lr * m, updates, state.momenta)
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
    momenta: Any | None = None,
) -> GradientTransformation:
    init_fn = partial(init, momenta=momenta)
    update_fn = partial(update, lr=lr, alpha=alpha, beta=beta)
    return GradientTransformation(init_fn, update_fn)
