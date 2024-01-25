from typing import Callable, Any, Tuple
import torch
from torch.func import grad, jvp, functional_call
from torch.distributions import Normal
from optree import tree_map, tree_map_, tree_reduce

from uqlib.types import TensorTree


def model_to_function(model: torch.nn.Module) -> Callable[[TensorTree, Any], Any]:
    """Converts a model into a function that maps parameters and inputs to outputs.

    Args:
        model: torch.nn.Module with parameters stored in .named_parameters().

    Returns:
        Function that takes a PyTree of parameters as well as any input
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
    """Modify a scalar-valued function that takes a PyTree (with tensor values) as first
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
    x: TensorTree, mean: TensorTree, sd_diag: TensorTree, validate_args: bool = False
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
    mean: TensorTree, sd_diag: TensorTree, sample_shape: torch.Size = torch.Size([])
) -> dict:
    """Sample from multivariate normal with diagonal covariance matrix.

    Args:
        mean: Mean of the distribution.
        sd_diag: Square-root diagonal of the covariance matrix.
        sample_shape: Shape of the sample.

    Returns:
        Sample(s) from normal distribution with the same structure as mean and sd_diag.
    """
    return tree_map(
        lambda m, sd: m + torch.randn(sample_shape + m.shape, device=m.device) * sd,
        mean,
        sd_diag,
    )


def tree_extract(f: Callable[[torch.tensor], bool], tree: TensorTree) -> TensorTree:
    """Extracts values from a PyTree where f returns True.
    False values are replaced with empty tensors.

    Args:
        f: A function that takes a PyTree element and returns True or False.
        tree: A PyTree.

    Returns:
        A PyTree with the same structure as tree where f returns True.
    """
    return tree_map(lambda x: x if f(x) else torch.tensor([], device=x.device), tree)


def tree_insert(
    f: Callable[[torch.tensor], bool], full_tree: TensorTree, sub_tree: TensorTree
) -> TensorTree:
    """Inserts sub_tree into full_tree where full_tree tensors evaluate f to True.
    Both PyTrees must have the same structure.

    Args:
        f: A function that takes a PyTree element and returns True or False.
        full_pytree: A PyTree to insert sub_pytree into.
        sub_pytree: A PyTree to insert into full_pytree.

    Returns:
        A PyTree with sub_tree inserted into full_tree.
    """
    return tree_map(
        lambda sub, full: sub if f(full) else full,
        sub_tree,
        full_tree,
    )


def tree_insert_(
    f: Callable[[torch.tensor], bool], full_tree: TensorTree, sub_tree: TensorTree
) -> TensorTree:
    """Inserts sub_tree into full_tree in-place where full_tree tensors evaluate
    f to True. Both PyTrees must have the same structure.

    Args:
        f: A function that takes a PyTree element and returns True or False.
        full_pytree: A PyTree to insert sub_pytree into.
        sub_pytree: A PyTree to insert into full_pytree.

    Returns:
        A pointer to full_tree with sub_tree inserted.
    """

    def insert_(full, sub):
        if f(full):
            full.data = sub.data

    return tree_map_(insert_, full_tree, sub_tree)


def extract_requires_grad(tree: TensorTree) -> TensorTree:
    """Extracts only parameters that require gradients.

    Args:
        tree: A PyTree of tensors.

    Returns:
        A PyTree of tensors that require gradients.
    """
    return tree_extract(lambda x: x.requires_grad, tree)


def insert_requires_grad(full_tree: TensorTree, sub_tree: TensorTree) -> TensorTree:
    """Inserts sub_tree into full_tree where full_tree tensors requires_grad.
    Both PyTrees must have the same structure.

    Args:
        full_pytree: A PyTree to insert sub_pytree into.
        sub_pytree: A PyTree to insert into full_pytree.

    Returns:
        A PyTree with sub_tree inserted into full_tree.
    """
    return tree_insert(lambda x: x.requires_grad, full_tree, sub_tree)


def insert_requires_grad_(full_tree: TensorTree, sub_tree: TensorTree) -> TensorTree:
    """Inserts sub_pytree into full_tree in-place where full_tree tensors requires_grad.
    Both PyTrees must have the same structure.

    Args:
        full_pytree: A PyTree to insert sub_tree into.
        sub_pytree: A PyTree to insert into full_tree.

    Returns:
        A pointer to full_tree with sub_tree inserted.
    """
    return tree_insert_(lambda x: x.requires_grad, full_tree, sub_tree)


def extract_requires_grad_and_func(
    tree: TensorTree, func: Callable, inplace: bool = False
) -> Tuple[TensorTree, Callable]:
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


def inplacify(func: Callable) -> Callable:
    """Converts a function that takes a tensor as its first argument
    into one that takes the same arguments but modifies the first arguent
    tensor in-place with the output of the function.

    Args:
        func: A function that takes a tensor as its first argument and a returns
            a modified version of said tensor.

    Returns:
        A function that takes a tensor as its first argument and modifies it
        in-place.
    """

    def func_(tens, *args, **kwargs):
        tens.data = func(tens, *args, **kwargs)
        return tens

    return func_


def tree_map_inplacify_(
    func: Callable,
    tree: TensorTree,
    *rests: TensorTree,
    is_leaf: Callable[[TensorTree], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = "",
) -> TensorTree:
    """Applies a pure function to each tensor in a PyTree in-place.

    Like optree.tree_map_ but takes a pure function as input
    (and takes replaces its first argument with its output in-place)
    rather than a side-effect function.

    Args:
        func: A function that takes a tensor as its first argument and a returns
            a modified version of said tensor.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first
            positional argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same
            structure as ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(x, *xs)`` (not the return value) where ``x`` is the value at the corresponding leaf
        in ``tree`` and ``xs`` is the tuple of values at values at corresponding nodes in ``rests``.
    """
    return tree_map_(
        inplacify(func),
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def flexi_tree_map(
    func: Callable,
    tree: TensorTree,
    *rests: TensorTree,
    inplace: bool = False,
    is_leaf: Callable[[TensorTree], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = "",
) -> TensorTree:
    """Applies a pure function to each tensor in a PyTree, with inplace argument.

    If inplace = True, uses uqlib.tree_map_inplacify_ to modify the tree in-place
        (and return modified tree).
    If inplace = False, uses optree.tree_map to return a new tree.

    Args:
        func: A pure function that takes a tensor as its first argument and a returns
            a modified version of said tensor.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first
            positional argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same
            structure as ``tree`` or has ``tree`` as a prefix.
        inplace (bool, optional): Whether to modify the tree in-place or not.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(x, *xs)`` (not the return value) where ``x`` is the value at the corresponding leaf
        in ``tree`` and ``xs`` is the tuple of values at values at corresponding nodes in ``rests``.
    """
    tm = tree_map_inplacify_ if inplace else tree_map
    return tm(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
