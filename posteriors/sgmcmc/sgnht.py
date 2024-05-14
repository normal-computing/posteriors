from typing import Any
from functools import partial
import torch
from torch.func import grad_and_value
from optree import tree_map
from optree.integration.torch import tree_ravel
from dataclasses import dataclass

from posteriors.types import TensorTree, Transform, LogProbFn, TransformState
from posteriors.tree_utils import flexi_tree_map
from posteriors.utils import is_scalar, CatchAuxError


def build(
    log_posterior: LogProbFn,
    lr: float,
    alpha: float = 0.01,
    beta: float = 0.0,
    temperature: float = 1.0,
    momenta: TensorTree | float | None = None,
    xi: float = None,
) -> Transform:
    """Builds SGNHT transform.

    Algorithm from [Ding et al, 2014](https://proceedings.neurips.cc/paper/2014/file/21fe5b8ba755eeaece7a450849876228-Paper.pdf):

    \\begin{align}
    θ_{t+1} &= θ_t + ε m_t \\\\
    m_{t+1} &= m_t + ε \\nabla \\log p(θ_t, \\text{batch}) - ε ξ_t m_t
    + N(0, ε T (2 α - ε β T) \\mathbb{I})\\\\
    ξ_{t+1} &= ξ_t + ε (m_t^T m_t / d - T)
    \\end{align}
    
    for learning rate $\\epsilon$, temperature $T$ and parameter dimension $d$.

    Targets $p_T(θ, m, ξ) \\propto \\exp( (\\log p(θ) - \\frac12 m^Tm + \\frac{d}{2}(ξ - α)^2) / T)$.

    The log posterior and temperature are recommended to be [constructed in tandem](../../log_posteriors.md)
    to ensure robust scaling for a large amount of data and variable batch size.
    
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
        xi: Initial value for scalar thermostat ξ. Defaults to `alpha`.

    Returns:
        SGNHT transform instance.
    """
    init_fn = partial(init, momenta=momenta, xi=xi or alpha)
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
class SGNHTState(TransformState):
    """State encoding params and momenta for SGNHT.

    Args:
        params: Parameters.
        momenta: Momenta for each parameter.
        log_posterior: Log posterior evaluation.
        aux: Auxiliary information from the log_posterior call.
    """

    params: TensorTree
    momenta: TensorTree
    xi: float
    log_posterior: torch.tensor = None
    aux: Any = None


def init(
    params: TensorTree, momenta: TensorTree | float | None = None, xi: float = 0.01
) -> SGNHTState:
    """Initialise momenta for SGNHT.

    Args:
        params: Parameters for which to initialise.
        momenta: Initial momenta. Can be tree like params or scalar.
            Defaults to random iid samples from N(0, 1).
        xi: Initial value for scalar thermostat ξ.

    Returns:
        Initial SGNHTState containing momenta.
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

    return SGNHTState(params, momenta, xi)


def update(
    state: SGNHTState,
    batch: Any,
    log_posterior: LogProbFn,
    lr: float,
    alpha: float = 0.01,
    beta: float = 0.0,
    temperature: float = 1.0,
    inplace: bool = False,
) -> SGNHTState:
    """Updates parameters, momenta and xi for SGNHT.
    
    Update rule from [Ding et al, 2014](https://proceedings.neurips.cc/paper/2014/file/21fe5b8ba755eeaece7a450849876228-Paper.pdf):

    \\begin{align}
    θ_{t+1} &= θ_t + ε m_t \\
    m_{t+1} &= m_t + ε \\nabla \\log p(θ_t, \\text{batch}) - ε ξ_t m_t
    + N(0, ε T (2 α - ε β T) \\mathbb{I})\\
    ξ_{t+1} &= ξ_t + ε (m_t^T m_t / d - T)
    \\end{align}
    
    for learning rate $\\epsilon$ and temperature $T$

    Args:
        state: SGNHTState containing params, momenta and xi.
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
        Updated SGNHTState
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
            - lr * state.xi * m
            + (temperature * lr * (2 * alpha - temperature * lr * beta)) ** 0.5
            * torch.randn_like(m)
        )

    m_flat, _ = tree_ravel(state.momenta)
    xi_new = state.xi + lr * (torch.mean(m_flat**2) - temperature)

    params = flexi_tree_map(
        transform_params, state.params, state.momenta, inplace=inplace
    )
    momenta = flexi_tree_map(transform_momenta, state.momenta, grads, inplace=inplace)

    if inplace:
        state.xi = xi_new
        state.log_posterior = log_post.detach()
        state.aux = aux
        return state
    return SGNHTState(params, momenta, xi_new, log_post.detach(), aux)
