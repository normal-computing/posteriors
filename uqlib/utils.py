from typing import Callable, Any, List
import torch
import torch.nn as nn
from torch.func import grad, jvp, functional_call
from torch.distributions import Normal
from optree import tree_map, tree_map_, tree_reduce


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

    def hessian_diag_fn(x: Any, *args, **kwargs) -> torch.Tensor:
        v = tree_map(lambda v: torch.ones_like(v, requires_grad=False), x)

        def ftemp(xtemp):
            return f(xtemp, *args, **kwargs)

        return hvp(ftemp, (x,), (v,))

    return hessian_diag_fn


def diag_normal_log_prob(
    x: Any, mean: Any, sd_diag: Any, validate_args: bool = False
) -> float:
    """Evaluate multivariate normal log probability for a diagonal covariance matrix.

    Args:
        x: Value to evaluate log probability at.
        mean: Mean of the distribution.
        sd_diag: Square-root diagonal of the covariance matrix.
        validate_args: Whether to validate arguments, defaults to False as
            torch.func.vmap doesn't like the control flows (if statements).

    Returns:
        Log probability.
    """
    log_probs = tree_map(
        lambda v, m, sd: Normal(m, sd, validate_args=validate_args).log_prob(v).sum(),
        x,
        mean,
        sd_diag,
    )
    log_prob = tree_reduce(torch.add, log_probs)
    return log_prob


def diag_normal_sample(
    mean: Any, sd_diag: Any, sample_shape: torch.Size = torch.Size([])
) -> dict:
    """Single sample from multivariate normal with diagonal covariance matrix.

    Args:
        mean: Mean of the distribution.
        sd_diag: Square-root diagonal of the covariance matrix.

    Returns:
        Sample from normal distribution with the same structure as mean and sd_diag.
    """
    return tree_map(
        lambda m, sd: m + torch.randn(sample_shape + m.shape, device=m.device) * sd,
        mean,
        sd_diag,
    )


def extract_requires_grad(tree: Any) -> Any:
    """Extracts only parameters that require gradients.

    Args:
        tree: A PyTree of tensors.

    Returns:
        A PyTree of tensors that require gradients.
    """
    return tree_map(
        lambda x: x if x.requires_grad else torch.tensor([], device=x.device), tree
    )


def insert_requires_grad(full_tree: Any, sub_tree: Any) -> Any:
    """Inserts sub_tree into full_tree where full_tree tensors requires_grad.
    Both PyTrees must have the same structure.

    Args:
        full_pytree: A PyTree to insert sub_pytree into.
        sub_pytree: A PyTree to insert into full_pytree.

    Returns:
        A PyTree with sub_tree inserted into full_tree.
    """
    return tree_map(
        lambda sub, full: sub if full.requires_grad else full,
        sub_tree,
        full_tree,
    )


def insert_requires_grad_(full_tree: Any, sub_tree: Any) -> Any:
    """Inserts sub_pytree into full_tree in-place where full_tree tensors requires_grad.
    Both PyTrees must have the same structure.

    Args:
        full_pytree: A PyTree to insert sub_tree into.
        sub_pytree: A PyTree to insert into full_tree.

    Returns:
        A pointer to full_tree with sub_tree inserted.
    """

    def insert_(full, sub):
        if full.requires_grad:
            full.data = sub.data

    return tree_map_(insert_, full_tree, sub_tree)


def extract_requires_grad_and_func(
    tree: Any, func: Callable, inplace: bool = False
) -> Any:
    """Extracts only parameters that require gradients and converts a function
    that takes the full parameter tree (in its first argument)
    into one that takes the subtree.

    Args:
        tree: A PyTree of tensors.
        func: A function that takes tree in its first argument.
        inplace: Whether to modify the tree inplace or not whe the new function
            is called.

    Returns:
        A PyTree of tensors that require gradients and a modified func that takes the
        subtree structure rather than full tree in its first argument.
    """
    subtree = extract_requires_grad(tree)

    insert = insert_requires_grad_ if inplace else insert_requires_grad

    def subfunc(subtree, *args, **kwargs):
        return func(insert(tree, subtree), *args, **kwargs)

    return subtree, subfunc


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
