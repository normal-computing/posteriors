from typing import Callable, Any

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import grad, jvp


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


def hvp(f, primals, tangents):
    """Hessian vector product.

    H_f(primals) @ tangents

    Taken from https://pytorch.org/functorch/nightly/notebooks/jacobians_hessians.html

    Args:
        f: A scalar-valued function that takes a dict as its single argument and
        produces a scalar output.
        x: Tensors.
        v: Tensors matching x.

    Returns:
        A tensor with the same shape as x.
    """
    return jvp(grad(f), primals, tangents)[1]


def diagonal_hessian(f: Callable) -> Callable:
    """Modify a scalar-valued function (that takes a dict with tensor values as first
    input) to return its Hessian diagonal.

    Inspired by https://github.com/google/jax/issues/3801

    Args:
        f: A scalar-valued function that takes a dict with tensor values in its first
        argument and produces a scalar output.

    Returns:
        A new function that computes the Hessian diagonal.
    """

    def hessian_diag_fn(x: dict[Any, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        v = {k: torch.ones_like(v) for k, v in x.items()}
        ftemp = lambda xtemp: f(xtemp, *args, **kwargs)
        return hvp(ftemp, (x,), (v,))

    return hessian_diag_fn
