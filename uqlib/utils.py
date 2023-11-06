from typing import Callable, Any

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import grad


def forward_multiple(model, parameter_vectors: torch.tensor, X) -> torch.tensor:
    """Evaluates multiple forward passes of a model with different parameter vectors.

    Does not use torch.inference_mode() by default
    (although this should be considered in most cases).

    Does not squeeze output, output is guranateed to be 3D
    (even if only one parameter vector or one input is passed).

    Args:
        model: torch.nn.Module
        parameter_vectors: torch.tensor
            Shape: (n_parameter_vectors, dim_parameter)
        X: torch.tensor
            Shape: (n_samples, dim_input)

    Returns:
        torch.tensor
            Shape: (n_samples, n_parameter_vectors, dim_output)
    """
    parameter_vectors = torch.atleast_2d(parameter_vectors).to(model.device)

    # This assumes that X is a tensor, is this a fair assumption?
    X = torch.atleast_2d(X).to(model)

    fs = list()

    orig_params = parameters_to_vector(model.parameters())

    for vec in parameter_vectors:
        vector_to_parameters(vec, model.parameters())
        fs.append(model(X))

    vector_to_parameters(orig_params, model.parameters())
    return torch.stack(fs).transpose(0, 1)


def hvp(f, x, v, *args, **kwargs):
    """Hessian vector product.

    H_f(x, args, kwargs) @ v

    Inspired by https://github.com/google/jax/issues/3801

    Args:
        f: A scalar-valued function that takes a tensor in its first argument and
        produces a scalar output.
        x: A tensor.
        v: A tensor with the same shape as x.
        args: Additional positional arguments to pass to f.
        kwargs: Additional keyword arguments to pass to f.

    Returns:
        A tensor with the same shape as x.
    """
    return grad(lambda x: torch.vdot(grad(f)(x, *args, **kwargs), v))(x)


def diagonal_hessian(f: Callable[[torch.tensor, Any], float]) -> Callable:
    """Modify a scalar-valued function to return its Hessian diagonal.

    Inspired by https://github.com/google/jax/issues/3801

    Args:
        f: A scalar-valued function that takes a tensor in its first argument and
        produces a scalar output.

    Returns:
        A new function that computes the Hessian diagonal.
    """

    def hessian_diag_fn(x: torch.tensor, *args, **kwargs) -> torch.Tensor:
        # Ensure the input tensor requires a gradient
        x = x.detach().requires_grad_(True)
        return hvp(f, x, torch.ones_like(x), *args, **kwargs)

    return hessian_diag_fn
