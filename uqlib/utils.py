from typing import Callable, Any, Tuple, List

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import grad, jvp, functional_call


def dict_map(f: Callable, d: dict, *rest: Tuple[dict, ...]):
    """Applies a function to each value in a dictionary or collection of dictionaries
    with the same keys.

    E.g. zeroed_dict = dict_map(lambda x: torch.zeros_like(x), dict1)
    or summed_dict = dict_map(lambda x, y: x + y, dict1, dict2)

    Args:
        f: Function to apply to each value. Takes len(rest) + 1 arguments.
        d: Dictionary or collection of dictionaries.
        *rest: Additional dictionaries (all with the same keys as d).

    Returns:
        Dictionary with the same keys as d, where each value is the result of applying
        f to the corresponding values in d and *rest.
    """
    return {k: f(d[k], *[r[k] for r in rest]) for k in d.keys()}


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


def forward_multiple(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    parameter_vectors: torch.Tensor,
) -> torch.Tensor:
    """Evaluates multiple forward passes of a model with different parameter vectors.

        Does not use torch.inference_mode() by default
        (although this should be considered in most cases).

        Does not squeeze output, output is guranateed to be 3D
        (even if only one parameter vector or one input is passed).

        Args:
            model: torch.nn.Module
            parameter_vectors: torch.tensor
                Shape: (n_parameter_vectors, dim_parameter)
            inputs: torch.tensor
                Shape: (n_samples, dim_input)

    Returns:
            torch.tensor
                Shape: (n_samples, n_parameter_vectors, dim_output)
    """
    parameter_vectors = torch.atleast_2d(parameter_vectors).to(model.device)

    # This assumes that X is a tensor, is this a fair assumption?
    inputs = torch.atleast_2d(inputs).to(model.device)

    outputs = list()

    orig_params = parameters_to_vector(model.parameters())

    for vec in parameter_vectors:
        vector_to_parameters(vec, model.parameters())
        outputs.append(model(inputs))

    vector_to_parameters(orig_params, model.parameters())
    return torch.stack(outputs).transpose(0, 1)


def hvp(f: Callable, primals: tuple, tangents: tuple):
    """Hessian vector product.

    H_f(primals) @ tangents

    Taken from https://pytorch.org/functorch/nightly/notebooks/jacobians_hessians.html
    Follows API from https://pytorch.org/docs/stable/generated/torch.func.jvp.html

    Args:
        f: A function with scalar output.
        x: Tuple of e.g. tensor or dict with tensor values to evalute f at.
        v: Tuple matching structure of x.

    Returns:
        Object matching structure of the elements of x and v.
    """
    return jvp(grad(f), primals, tangents)[1]


def diagonal_hessian(f: Callable) -> Callable:
    """Modify a scalar-valued function that takes a dict (with tensor values) as first
    input to return its Hessian diagonal.

    Inspired by https://github.com/google/jax/issues/3801

    Args:
        f: A scalar-valued function that takes a dict with tensor values in its first
        argument and produces a scalar output.

    Returns:
        A new function that computes the Hessian diagonal.
    """

    def hessian_diag_fn(x: dict[Any, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            v = torch.ones_like(x)
        elif isinstance(x, dict):
            v = dict_map(lambda v: torch.ones_like(v), x)
        else:
            raise ValueError("x must be a tensor or dict with tensor values")

        def ftemp(xtemp):
            return f(xtemp, *args, **kwargs)

        return hvp(ftemp, (x,), (v,))

    return hessian_diag_fn


def load_optimizer_param_to_model(model: nn.Module, groups: List[List[torch.Tensor]]):
    """Updates the model parameters in-place with the provided optimizer parameters (provided by SGHMC optimizer)

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
