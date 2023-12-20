from typing import Any, NamedTuple, Tuple
from functools import partial
import torch
from torchopt.base import GradientTransformation
from optree import tree_map


class SGHMCState(NamedTuple):
    momenta: Any


def init(params: Any, momenta: Any | None = None) -> SGHMCState:
    if momenta is None:
        momenta = tree_map(torch.zeros_like, params)
    return SGHMCState(momenta)


def update(
    grads: Any,
    state: SGHMCState,
    params: Any,
    lr: float,
    alpha: float = 0.01,
    beta: float = 0.0,
    inplace: bool = False,
) -> Tuple[Any, SGHMCState]:
    if inplace:
        raise NotImplementedError("inplace updates not implemented")

    def transform_momenta(g, m):
        return (
            m
            + lr * g
            - lr * alpha * m
            + (lr * (2 * alpha - lr * beta)) ** 0.5 * torch.randn_like(m)
        )

    momenta = tree_map(transform_momenta, grads, state.momenta)

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
