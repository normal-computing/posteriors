from typing import Callable, Tuple
import torch
from optree import tree_map, tree_map_, tree_reduce

from posteriors.types import TensorTree


def tree_size(tree: TensorTree) -> int:
    """Returns the total number of elements in a PyTree.
    Not the number of leaves, but the total number of elements for all tensors in the
    tree.

    Args:
        tree: A PyTree of tensors.

    Returns:
        Number of elements in the PyTree.
    """

    def ensure_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x)

    return tree_reduce(torch.add, tree_map(lambda x: ensure_tensor(x).numel(), tree))


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
        full_tree: A PyTree to insert sub_tree into.
        sub_tree: A PyTree to insert into full_tree.

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
        full_tree: A PyTree to insert sub_tree into.
        sub_tree: A PyTree to insert into full_tree.

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
        full_tree: A PyTree to insert sub_tree into.
        sub_tree: A PyTree to insert into full_tree.

    Returns:
        A PyTree with sub_tree inserted into full_tree.
    """
    return tree_insert(lambda x: x.requires_grad, full_tree, sub_tree)


def insert_requires_grad_(full_tree: TensorTree, sub_tree: TensorTree) -> TensorTree:
    """Inserts sub_tree into full_tree in-place where full_tree tensors requires_grad.
    Both PyTrees must have the same structure.

    Args:
        full_tree: A PyTree to insert sub_tree into.
        sub_tree: A PyTree to insert into full_tree.

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
    into one that takes the same arguments but modifies the first argument
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

    Like [optree.tree_map_](https://optree.readthedocs.io/en/latest/ops.html#optree.tree_map_)
    but takes a pure function as input (and takes replaces its first argument with its
    output in-place) rather than a side-effect function.

    Args:
        func: A function that takes a tensor as its first argument and a returns
            a modified version of said tensor.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first
            positional argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same
            structure as ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be
            called at each flattening step. It should return a boolean, with
            `True` stopping the traversal and the whole subtree being treated as a
            leaf, and `False` indicating the flattening should traverse the
            current object.
        none_is_leaf (bool, optional): Whether to treat `None` as a leaf. If
            `False`, `None` is a non-leaf node with arity 0. Thus `None` is contained in
            the treespec rather than in the leaves list and `None` will be remain in the
            result pytree. (default: `False`)
        namespace (str, optional): The registry namespace used for custom pytree node
            types. (default: :const:`''`, i.e., the global namespace)

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of
            function ``func(x, *xs)`` (not the return value) where ``x`` is the value at
            the corresponding leaf in ``tree`` and ``xs`` is the tuple of values at
            values at corresponding nodes in ``rests``.
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

    ```
    out_tensor = func(tensor, *rest_tensors)
    ```

    where `out_tensor` is of the same shape as `tensor`.
    Therefore

    ```
    out_tree = func(tree, *rests, inplace=True)
    ```

    will return `out_tree` a pointer to the original `tree` with leaves (tensors)
    modified in place.
    If `inplace=False`, `flexi_tree_map` is equivalent to [`optree.tree_map`](https://optree.readthedocs.io/en/latest/ops.html#optree.tree_map)
    and returns a new tree.

    Args:
        func: A pure function that takes a tensor as its first argument and a returns
            a modified version of said tensor.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first
            positional argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same
            structure as ``tree`` or has ``tree`` as a prefix.
        inplace (bool, optional): Whether to modify the tree in-place or not.
        is_leaf (callable, optional): An optionally specified function that will be
            called at each flattening step. It should return a boolean, with `True`
            stopping the traversal and the whole subtree being treated as a leaf, and
            `False` indicating the flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat `None` as a leaf. If `False`,
            `None` is a non-leaf node with arity 0. Thus `None` is contained in the
            treespec rather than in the leaves list and `None` will be remain in the
            result pytree. (default: `False`)
        namespace (str, optional): The registry namespace used for custom pytree node
            types. (default: :const:`''`, i.e., the global namespace)

    Returns:
        Either the original tree modified in-place or a new tree depending on the
            `inplace` argument.
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
