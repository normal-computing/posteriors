from functools import partial
from typing import Callable, Any, NamedTuple

from optree import tree_map
from torch.func import grad_and_value, vmap
from optree.integration.torch import tree_ravel

from posteriors.types import TensorTree, Transform


def _build_stein_variational_gradient_step(
    log_posterior: Callable[[TensorTree, Any], TensorTree], kernel: Callable
):
    """
    hardcode a function that calculates phi_star according to user defined log_posterior_gradient
    """

    # def _phi_star_summand(param, param_, batch):
    #     log_prob_grad, _ = grad_and_value(log_posterior, argnums=0)(param, batch)
    #     grad_k, k = grad_and_value(kernel, argnums=0)(param, param_)
    #     return tree_map(lambda gl, gk: (k * gl) + gk, log_prob_grad, grad_k)

    def step(params: TensorTree, batch):
        def _phi_star_summand(param, param_, batch):
            log_prob_grad, _ = grad_and_value(log_posterior, argnums=0)(param, batch)
            grad_k, k = grad_and_value(kernel, argnums=0)(param, param_)
            return tree_map(lambda gl, gk: (k * gl) + gk, log_prob_grad, grad_k)

        phi_star_summand = partial(_phi_star_summand, batch=batch)
        r_params, unravel = tree_ravel(params)

        gradients = tree_map(
            lambda p: vmap(lambda p_: phi_star_summand(p, p_))(r_params).mean(axis=0),
            params,
        )
        return gradients
        # gradients = vmap(
        #     lambda param: (
        #         vmap(lambda param_: phi_star_summand(param, param_))(r_params).mean(
        #             axis=0
        #         )
        #     )
        # )(r_params)
        # return unravel(gradients)

    return step


def build(
    log_posterior: Callable[[TensorTree, Any], TensorTree],
    learning_rate: float,
    kernel: Callable,
) -> Transform:
    """
    TBD
    """
    step_gradient_fn = _build_stein_variational_gradient_step(log_posterior, kernel)
    update_fn = partial(
        update,
        step_function=step_gradient_fn,
        learning_rate=learning_rate,
    )
    return Transform(init, update_fn)


class SVGDState(NamedTuple):
    """
    TBD
    """

    params: TensorTree


def init(
    params: TensorTree,
) -> SVGDState:
    """TBD"""
    return SVGDState(params)


def update(
    state: SVGDState,
    batch: Any,
    step_function: Callable,
    learning_rate: float,
    inplace: bool = False,
) -> SVGDState:
    """
    TBD
    """

    step_gradient = step_function(state.params, batch)
    params = tree_map(lambda p, g: p + learning_rate * g, state.params, step_gradient)
    if inplace:
        state.params = params
    else:
        return SVGDState(params)
