from typing import Any
from functools import partial
import torch
from torch.func import grad_and_value
from optree import tree_map
from dataclasses import dataclass

from posteriors.types import TensorTree, Transform, LogProbFn, TransformState
from posteriors.utils import flexi_tree_map, is_scalar, CatchAuxError


def build(
    log_posterior: LogProbFn,
    lr: float,
    alpha: float = 0.01,
    beta: float = 0.0,
    temperature: float = 1.0,
    momenta: TensorTree | float | None = None,
) -> Transform:
    """Builds SGHMC transform.

    Algorithm from [Chen et al, 2014](https://arxiv.org/abs/1402.4102):

    \\begin{align}
    θ_{t+1} &= θ_t + ε m_t \\\\
    m_{t+1} &= m_t + ε \\nabla \\log p(θ_t, \\text{batch}) - ε α m_t
    + N(0, ε T (2 α - ε β) T \\mathbb{I})\\
    \\end{align}
    
    for learning rate $\\epsilon$ and temperature $T$

    Targets $p_T(θ, m) \\propto \\exp( (\\log p(θ) - \\frac12 m^Tm) / T)$
    with temperature $T$.

    The log posterior and temperature are recommended to be [constructed in tandem](../../log_posteriors.md)
    to ensure robust scaling for a large amount of data.

    Args:
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        lr: Learning rate.
        alpha: Friction coefficient.
        beta: Gradient noise coefficient (estimated variance).
        temperature: Temperature of the joint parameter + momenta distribution.
        momenta: Initial momenta. Can be tree like params or scalar.
            Defaults to random iid samples from N(0, 1).

    Returns:
        SGHMC transform instance.
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


@dataclass
class SGHMCState(TransformState):
    """State encoding params and momenta for SGHMC.

    Args:
        params: Parameters.
        momenta: Momenta for each parameter.
        log_posterior: Log posterior evaluation.
        aux: Auxiliary information from the log_posterior call.
    """

    params: TensorTree
    momenta: TensorTree
    log_posterior: torch.tensor = None
    aux: Any = None


def init(params: TensorTree, momenta: TensorTree | float | None = None) -> SGHMCState:
    """Initialise momenta for SGHMC.

    Args:
        params: Parameters for which to initialise.
        momenta: Initial momenta. Can be tree like params or scalar.
            Defaults to random iid samples from N(0, 1).

    Returns:
        Initial SGHMCState containing momenta.
    """
    if momenta is None:
        momenta = tree_map(
            lambda x: torch.randn_like(x, requires_grad=x.requires_grad),
            params,
        )
    elif is_scalar(momenta):
        momenta = tree_map(
            lambda x: torch.full_like(x, momenta, requires_grad=x.requires_grad),
            params,
        )

    return SGHMCState(params, momenta)


def update(
    state: SGHMCState,
    batch: Any,
    log_posterior: LogProbFn,
    lr: float,
    alpha: float = 0.01,
    beta: float = 0.0,
    temperature: float = 1.0,
    inplace: bool = False,
) -> SGHMCState:
    """Updates parameters and momenta for SGHMC.
    
    Update rule from [Chen et al, 2014](https://arxiv.org/abs/1402.4102):

    \\begin{align}
    θ_{t+1} &= θ_t + ε m_t \\\\
    m_{t+1} &= m_t + ε \\nabla \\log p(θ_t, \\text{batch}) - ε α m_t
    + N(0, ε T (2 α - ε β) T \\mathbb{I})\\
    \\end{align}
    
    for learning rate $\\epsilon$ and temperature $T$

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
        inplace: Whether to modify state in place.

    Returns:
        Updated state
        (which are pointers to the inputted state tensors if inplace=True).
    """
    with torch.no_grad(), CatchAuxError():
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

    if inplace:
        state.log_posterior = log_post.detach()
        state.aux = aux
        return state
    return SGHMCState(params, momenta, log_post.detach(), aux)
