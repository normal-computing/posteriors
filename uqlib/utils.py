from typing import Callable, Any

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import grad


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


def hvp(f: Callable, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
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


def diagonal_hessian(f: Callable) -> Callable:
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
