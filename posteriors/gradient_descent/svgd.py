from functools import partial
from typing import Callable, Any, NamedTuple

import torchopt
from optree import tree_map
from torch.func import grad_and_value, vmap
from optree.integration.torch import tree_ravel

from posteriors.types import TensorTree, Transform


def _build_stein_variational_gradient_step(
    log_posterior_gradient: Callable[[TensorTree, Any], TensorTree], kernel: Callable
):
    """
    hardcode a function that calculates phi_star according to user defined log_posterior_gradient
    """

    def _phi_star_summand(param, param_, batch):
        grad_log = log_posterior_gradient(param, batch)
        k, grad_k = grad_and_value(kernel, argnums=0)(param, param_)
        return tree_map(lambda gl, gk: -(k * gl) - gk, grad_log, grad_k)

    def step(params: TensorTree, batch):
        phi_star_summand = partial(_phi_star_summand, batch=batch)
        r_params, unravel = tree_ravel(params)
        gradients = vmap(
            lambda param: (
                vmap(lambda param_: phi_star_summand(param, param_))(r_params).mean(
                    axis=0
                )
            )
        )(r_params)
        return unravel(gradients)

    return step


def build(
    log_posterior_gradient: Callable[[TensorTree, Any], TensorTree],
    optimizer: torchopt.base.GradientTransformation,
    kernel: Callable[[TensorTree, Any], float],
) -> Transform:
    """
    TBD
    """
    init_fn = partial(init, optimizer=optimizer)
    step_gradient_fn = _build_stein_variational_gradient_step(
        log_posterior_gradient, kernel
    )
    update_fn = partial(
        update,
        step_function=step_gradient_fn,
        optimizer=optimizer,
    )
    return Transform(init_fn, update_fn)


class SVGDState(NamedTuple):
    """
    TBD
    """

    params: TensorTree
    opt_state: torchopt.typing.OptState


def init(
    params: TensorTree,
    optimizer: torchopt.base.GradientTransformation,
) -> SVGDState:
    """TBD"""
    opt_state = optimizer.init(params)
    return SVGDState(params, opt_state)


def update(
    state: SVGDState,
    batch: Any,
    step_function: Callable,
    optimizer: torchopt.base.GradientTransformation,
    inplace: bool = False,
) -> SVGDState:
    """
    TBD
    """

    step_gradient = step_function(state.params, batch)

    updates, opt_state = optimizer.update(
        step_gradient, state.opt_state, params=state.params, inplace=inplace
    )
    params = torchopt.apply_updates(state.params, updates)

    return SVGDState(params, opt_state)
