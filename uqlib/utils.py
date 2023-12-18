from functools import reduce
from typing import Callable, Any, Tuple, List, Union
import torch
import torch.nn as nn
from torch.func import grad, jvp, functional_call
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.distributions import Normal


def tree_map(f: Callable, tree: Any, *rest: Tuple[dict, ...]):
    """Applies a function to each value in a pytree or collection of pytrees
    with the same structure (which advances torch.utils._pytree.tree_map).

    E.g. zeroed_dict = tree_map(lambda x: torch.zeros_like(x), dict1)
    or summed_dict = tree_map(lambda x, y: x + y, dict1, dict2)

    Args:
        f: Function to apply to each value. Takes len(rest) + 1 arguments.
        tree: Pytree.
        *rest: Additional pytree (all with the same structure as tree).

    Returns:
        Pytree with the same structure as tree.
    """
    leaves, spec = tree_flatten(tree)
    all_leaves = [leaves] + [tree_flatten(r)[0] for r in rest]
    return tree_unflatten([f(*xs) for xs in zip(*all_leaves)], spec)


def tree_reduce(f: Callable, tree: Any) -> Any:
    """Apply a function of two arguments cumulatively to the items of a pytree.
    See functools.reduce

    E.g. sum_dict = tree_reduce(torch.add, dict1)

    Args:
        f: Function to apply to each value.
        tree: Pytree.

    Returns:
        Reduced output of f.
    """
    return reduce(f, tree_flatten(tree)[0])


def model_to_function(model: torch.nn.Module) -> Callable[[dict, Any], Any]:
    """Converts a model into a function that maps parameters and inputs to outputs.

    Args:
        model: torch.nn.Module with parameters stored in .named_parameters().

    Returns:
        Function that takes a dictionary of parameters as well as any input
        arg or kwargs and returns the output of the model.
    """

    def func_model(p_dict, *args, **kwargs):
        return functional_call(model, p_dict, args=args, kwargs=kwargs)

    return func_model


def hvp(f: Callable, primals: tuple, tangents: tuple):
    """Hessian vector product.

    H_f(primals) @ tangents

    Taken from https://pytorch.org/functorch/nightly/notebooks/jacobians_hessians.html
    Follows API from https://pytorch.org/docs/stable/generated/torch.func.jvp.html

    Args:
        f: A function with scalar output.
        primals: Tuple of e.g. tensor or dict with tensor values to evalute f at.
        tangents: Tuple matching structure of primals.

    Returns:
        Object matching structure of the elements of x and v.
    """
    return jvp(grad(f), primals, tangents)[1]


def hessian_diag(f: Callable) -> Callable:
    """Modify a scalar-valued function that takes a dict (with tensor values) as first
    input to return its Hessian diagonal.

    Inspired by https://github.com/google/jax/issues/3801

    Args:
        f: A scalar-valued function that takes a dict with tensor values in its first
        argument and produces a scalar output.

    Returns:
        A new function that computes the Hessian diagonal.
    """

    def hessian_diag_fn(
        x: Union[torch.Tensor, dict[Any, torch.Tensor]], *args, **kwargs
    ) -> torch.Tensor:
        v = tree_map(lambda v: torch.ones_like(v, requires_grad=False), x)

        def ftemp(xtemp):
            return f(xtemp, *args, **kwargs)

        return hvp(ftemp, (x,), (v,))

    return hessian_diag_fn


def diag_normal_log_prob(x: Any, mean: Any, sd_diag: Any) -> float:
    """Evaluate multivariate normal log probability for a diagonal covariance matrix.

    Args:
        x: Value to evaluate log probability at.
        mean: Mean of the distribution.
        sd_diag: Square-root diagonal of the covariance matrix.

    Returns:
        Log probability.
    """
    log_probs = tree_map(
        lambda v, m, sd: Normal(m, sd).log_prob(v).sum(), x, mean, sd_diag
    )
    log_prob = tree_reduce(torch.add, log_probs)
    return log_prob


def diag_normal_sample(mean: Any, sd_diag: Any, sample_shape=torch.Size([])) -> dict:
    """Single sample from multivariate normal with diagonal covariance matrix.

    Args:
        mean: Mean of the distribution.
        sd_diag: Square-root diagonal of the covariance matrix.

    Returns:
        Sample from normal distribution with the same structure as mean and sd_diag.
    """
    return tree_map(
        lambda m, sd: m + torch.randn(sample_shape + m.shape) * sd, mean, sd_diag
    )


def load_optimizer_param_to_model(model: nn.Module, groups: List[List[torch.Tensor]]):
    """Updates the model parameters in-place with the provided grouped parameters.

    Args:
        model: A torch.nn.Module object
        groups: A list of groups where each group is a list of parameters
    """

    optimizer_params = []
    for group in groups:
        for param in group:
            optimizer_params.append(torch.from_numpy(param))

    for model_param, optimizer_param in zip(list(model.parameters()), optimizer_params):
        model_param.data = optimizer_param
