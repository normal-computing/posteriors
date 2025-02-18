from typing import Any
from functools import partial
import torch
from torch.func import grad_and_value
from optree import tree_map
from optree.integration.torch import tree_ravel
from tensordict import TensorClass, NonTensorData

from posteriors.types import TensorTree, Transform, LogProbFn
from posteriors.tree_utils import flexi_tree_map, tree_insert_, tree_size
from posteriors.utils import is_scalar, CatchAuxError


def build(
    log_posterior: LogProbFn,
    lr: float,
    alpha: float = 0.01,
    sigma: float = 1.0,
    temperature: float = 1.0,
    momenta: TensorTree | float | None = None,
) -> Transform:
    """Builds BAOA-NHT transform.

    Algorithm from [Leimkuhler and Matthews, 2015 - p350](https://link.springer.com/ok/10.1007/978-3-319-16375-8).

    Langevin dynamics with a Nosé-Hoover thermostat, discretized in an BAOA-like manner.
    
    Note that the Nosé-Hoover-Langevin dynamics here are different from the ones in
    [SGNHT](sgnht.md). Here we follow [Leimkuhler and Matthews, 2015 - p345](https://link.springer.com/ok/10.1007/978-3-319-16375-8),
    whereas [SGNHT](sgnht.md) follows [Ding et al, 2014](https://proceedings.neurips.cc/paper/2014/file/21fe5b8ba755eeaece7a450849876228-Paper.pdf)
    with more focus on the handling of stochastic gradients.
    
    \\begin{align}
    m_{t+1/2} &= m_t + ε \\nabla \\log p(θ_t, \\text{batch}), \\\\
    θ_{t+1/2} &= θ_t + (ε / 2) σ^{-2} m_{t+1/2}, \\\\
    ξ_{t+1/2} &= ξ_t + (ε / 2) (σ^{-2} d^{-1} m_t^T m_t - T), \\\\
    \\tilde{m}_{t+1/2} &= e^{-ε/2 ξ_{t+1/2}} m_{t+1/2}, \\\\
    \\tilde{ξ}_{t+1/2} &= e^{-α ε}ξ_{t+1/2} + N(0, d^{-1}T (1 - e^{-2α ε})), \\\\
    m_{t+1} &= e^{-ε/2 \\tilde{ξ}_{t+1/2}} \\tilde{m}_{t+1/2}, \\\\
    ξ_{t+1} &= \\tilde{ξ}_{t+1/2} + (ε / 2) (σ^{-2} d^{-1} m_{t+1}^T m_{t+1} - T), \\\\
    θ_{t+1} &= θ_{t+1/2} + (ε / 2) σ^{-2} m_{t+1},
    \\end{align}
 
    for learning rate $\\epsilon$, temperature $T$, thermostat friction $α$
    and momenta noise variance $σ^2$.

    Targets $p_T(θ, m, ξ) \\propto \\exp( (\\log p(θ) - \\frac{1}{2σ^2} m^Tm - \\frac{d}{2}ξ^2) / T)$.

    The log posterior and temperature are recommended to be [constructed in tandem](../../log_posteriors.md)
    to ensure robust scaling for a large amount of data and variable batch size.
    
    Args:
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        lr: Learning rate.
        alpha: Friction coefficient.
        sigma: Standard deviation of momenta target distribution.
        temperature: Temperature of the joint parameter + momenta distribution.
        momenta: Initial momenta. Can be tree like params or scalar.
            Defaults to random iid samples from N(0, 1).

    Returns:
        BAOA-NHT transform instance.
    """
    init_fn = partial(init, momenta=momenta)
    update_fn = partial(
        update,
        log_posterior=log_posterior,
        lr=lr,
        alpha=alpha,
        sigma=sigma,
        temperature=temperature,
    )
    return Transform(init_fn, update_fn)


class BAOANHTState(TensorClass["frozen"]):
    """State encoding params and momenta for BAOA-NHT.

    Attributes:
        params: Parameters.
        momenta: Momenta for each parameter.
        xi: Scalar thermostat.
        log_posterior: Log posterior evaluation.
        aux: Auxiliary information from the log_posterior call.
    """

    params: TensorTree
    momenta: TensorTree
    xi: torch.Tensor = torch.tensor([])
    log_posterior: torch.Tensor = torch.tensor([])
    aux: NonTensorData = None


def init(
    params: TensorTree,
    momenta: TensorTree | float | None = None,
    xi: float | torch.Tensor = 0.01,
) -> BAOANHTState:
    """Initialise momenta for BAOA-NHT.

    Args:
        params: Parameters for which to initialise.
        momenta: Initial momenta. Can be tree like params or scalar.
            Defaults to random iid samples from N(0, 1).
        xi: Initial value for scalar thermostat ξ.

    Returns:
        Initial BAOANHTState containing params, momenta and xi (thermostat).
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

    return BAOANHTState(params, momenta, torch.tensor(xi))


def update(
    state: BAOANHTState,
    batch: Any,
    log_posterior: LogProbFn,
    lr: float,
    alpha: float = 0.01,
    sigma: float = 1.0,
    temperature: float = 1.0,
    inplace: bool = False,
) -> BAOANHTState:
    """Updates parameters and momenta for BAOA.

    Algorithm from [Leimkuhler and Matthews, 2015 - p271](https://link.springer.com/ok/10.1007/978-3-319-16375-8).

    See [build](baoa.md#posteriors.sgmcmc.baoa.build) for more details.

    Args:
        state: SGHMCState containing params and momenta.
        batch: Data batch to be send to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        lr: Learning rate.
        alpha: Friction coefficient.
        sigma: Standard deviation of momenta target distribution.
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

    prec = sigma**-2

    d = tree_size(state.params)

    def BB_step(m, g):
        return m + lr * g

    def A_step(p, m):
        return p + (lr / 2) * prec * m

    def thermostat_step(xi, m):
        m_flat, _ = tree_ravel(m)
        return xi + (lr / 2) * (prec * torch.mean(m_flat**2) - temperature)

    def scale_momenta(m, xi):
        return torch.exp(-xi * lr / 2) * m

    def O_thermostat_step(xi):
        return (
            torch.exp(-alpha * lr) * xi
            + torch.randn_like(xi)
            * (temperature / d * (1 - torch.exp(-2 * alpha * lr))) ** 0.5
        )

    momenta = flexi_tree_map(BB_step, state.momenta, grads, inplace=inplace)
    params = flexi_tree_map(A_step, state.params, momenta, inplace=inplace)
    xi = thermostat_step(state.xi, momenta)
    momenta = flexi_tree_map(scale_momenta, momenta, xi, inplace=inplace)
    xi = O_thermostat_step(xi)
    momenta = flexi_tree_map(scale_momenta, momenta, xi, inplace=inplace)
    params = flexi_tree_map(A_step, params, momenta, inplace=inplace)

    if inplace:
        tree_insert_(state.xi, xi)
        tree_insert_(state.log_posterior, log_post.detach())
        return state.replace(aux=NonTensorData(aux))
    return BAOANHTState(params, momenta, xi, log_post.detach(), aux)
