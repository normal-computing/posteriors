from typing import Callable, Any, Tuple
from functools import partial
import torch
from torch.func import grad, jvp, functional_call, jacrev
from torch.distributions import Normal
from optree import tree_map, tree_map_, tree_reduce, tree_flatten
from optree.integration.torch import tree_ravel

from posteriors.types import TensorTree, ForwardFn, Tensor


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


def linearized_forward_diag(
    forward_func: ForwardFn, params: TensorTree, batch: TensorTree, sd_diag: TensorTree
) -> Tuple[TensorTree, Tensor, TensorTree]:
    """Compute the linearized forward mean and its square root covariance, assuming
    posterior covariance over parameters is diagonal.

    f(x | θ) ~ N(f(x | θₘ), J(x | θₘ) @ Σ @ J(x | θₘ)^T)

    where θₘ is the MAP estimate, Σ is the diagonal covariance approximation at the MAP
    and J(x | θₘ) is the Jacobian of the forward function f(x | θₘ) with respect to θₘ.

    Args:
        forward_func: A function that takes params and batch and returns the forward
            values and any auxiliary information. Forward values must be a dim=2 Tensor
            with batch dimension in its first axis.
        params: PyTree of tensors.
        batch: PyTree of tensors.
        sd_diag: PyTree of tensors of same shape as params.

    Returns:
        A tuple of (forward_vals, chol, aux) where forward_vals is the output of the
        forward function (mean), chol is the tensor square root of the covariance matrix
        (non-diagonal) and aux is auxiliary info from the forward function.
    """
    forward_vals, aux = forward_func(params, batch)

    with torch.no_grad():
        jac, _ = jacrev(forward_func, has_aux=True)(params, batch)

    # Convert Jacobian to be flat in parameter dimension
    jac = tree_flatten(jac)[0]
    jac = torch.cat([x.flatten(start_dim=2) for x in jac], dim=2)

    # Flatten the diagonal square root covariance
    sd_diag = tree_flatten(sd_diag)[0]
    sd_diag = torch.cat([x.flatten() for x in sd_diag])

    # Cholesky of J @ Σ @ J^T
    linearised_chol = torch.linalg.cholesky((jac * sd_diag**2) @ jac.transpose(-1, -2))

    return forward_vals, linearised_chol, aux


def hvp(
    f: Callable, primals: tuple, tangents: tuple, has_aux: bool = False
) -> Tuple[float, TensorTree] | Tuple[float, TensorTree, Any]:
    """Hessian vector product.

    H_f(primals) @ tangents

    Taken from https://pytorch.org/functorch/nightly/notebooks/jacobians_hessians.html
    Follows API from https://pytorch.org/docs/stable/generated/torch.func.jvp.html

    Args:
        f: A function with scalar output.
        primals: Tuple of e.g. tensor or dict with tensor values to evalute f at.
        tangents: Tuple matching structure of primals.
        has_aux: Whether f returns auxiliary information.

    Returns:
        Returns a (gradient, hvp_out) tuple containing the gradient of func evaluated at
        primals and the Hessian-vector product. If has_aux is True, then instead
        returns a (gradient, hvp_out, aux) tuple.
    """
    return jvp(grad(f, has_aux=has_aux), primals, tangents, has_aux=has_aux)


def diag_normal_log_prob(
    x: TensorTree,
    mean: float | TensorTree = 0.0,
    sd_diag: float | TensorTree = 1.0,
    normalize: bool = True,
) -> float:
    """Evaluate multivariate normal log probability for a diagonal covariance matrix.

    If either mean or sd_diag are scalars, it will be broadcasted to the same shape as x
    (in a memory efficient manner).

    Args:
        x: Value to evaluate log probability at.
        mean: Mean of the distribution. Defaults to 0.0.
        sd_diag: Square-root diagonal of the covariance matrix. Defaults to 1.0.
        normalize: Whether to use normalized log probability.
            If False the elementwise log prob is -0.5 * ((x - mean) / sd_diag)**2.

    Returns:
        Log probability.
    """
    if tree_size(mean) == 1:
        mean = tree_map(lambda t: torch.tensor(mean, device=t.device), x)
    if tree_size(sd_diag) == 1:
        sd_diag = tree_map(lambda t: torch.tensor(sd_diag, device=t.device), x)

    if normalize:

        def univariate_norm_and_sum(v, m, sd):
            return Normal(m, sd, validate_args=False).log_prob(v).sum()
    else:

        def univariate_norm_and_sum(v, m, sd):
            return (-0.5 * ((v - m) / sd) ** 2).sum()

    log_probs = tree_map(
        univariate_norm_and_sum,
        x,
        mean,
        sd_diag,
    )
    log_prob = tree_reduce(torch.add, log_probs)
    return log_prob


def diag_normal_sample(
    mean: TensorTree,
    sd_diag: float | TensorTree,
    sample_shape: torch.Size = torch.Size([]),
) -> dict:
    """Sample from multivariate normal with diagonal covariance matrix.

    If sd_diag is scalar, it will be broadcasted to the same shape as mean
    (in a memory efficient manner).

    Args:
        mean: Mean of the distribution.
        sd_diag: Square-root diagonal of the covariance matrix.
        sample_shape: Shape of the sample.

    Returns:
        Sample(s) from normal distribution with the same structure as mean and sd_diag.
    """
    if tree_size(sd_diag) == 1:
        sd_diag = tree_map(lambda t: torch.tensor(sd_diag, device=t.device), mean)

    return tree_map(
        lambda m, sd: m + torch.randn(sample_shape + m.shape, device=m.device) * sd,
        mean,
        sd_diag,
    )


def tree_size(tree: TensorTree) -> int:
    """Returns the total number of elements in a PyTree.
    Not the number of leaves, but the total sum of the number of elements in each tensor.

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

    ```
    out_tensor = func(tensor, *rest_tensors)
    ```

    where `out_tensor` is of the same shape as `tensor`.
    Therefore

    ```
    out_tree = func(tree, *rests, inplace=True)
    ```

    will return `out_tree` a pointer to the original `tree` with leaves (tensors) modified in place.
    If `inplace=False`, `flexi_tree_map` is equivalent to `optree.tree_map` and returns a new tree.

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
        Either the original tree modified in-place or a new tree depending on the `inplace`
        argument.
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


def per_samplify(
    f: Callable[[TensorTree, TensorTree], Any],
) -> Callable[[TensorTree, TensorTree], Any]:
    """Converts a function that takes params and batch and averages over the batch in
    its output into one that provides an output for each batch sample
    (i.e. no averaging).

    ```
    output = f(params, batch)
    per_sample_output = per_samplify(f)(params, batch)
    ```

    for more info see https://pytorch.org/tutorials/intermediate/per_sample_grads.html

    Args:
        f: A function that takes params and batch and averages over the batch in its
        output.

    Returns:
        A new function that provides an output for each batch sample.
            `per_sample_output  = per_samplify(f)(params, batch)`
    """

    @partial(torch.vmap, in_dims=(None, 0))
    def f_per_sample(params, batch):
        batch = tree_map(lambda x: x.unsqueeze(0), batch)
        return f(params, batch)

    return f_per_sample


def is_scalar(x: Any) -> bool:
    """Returns True if x is a scalar (int, float, bool) or a tensor with a single element.

    Args:
        x: Any object.

    Returns:
        True if x is a scalar.
    """
    return isinstance(x, (int, float)) or (torch.is_tensor(x) and x.numel() == 1)


def empirical_fisher(
    f: Callable[[TensorTree, TensorTree], Any], params: TensorTree, batch: Any
) -> Tuple[Tensor, Any]:
    """
    Compute the empirical Fisher information matrix of a function f with respect to its
    parameters, defined as:

    F(θ) = ∑ᵢ ∇_θ f_θ(xᵢ, yᵢ)^T ∇_θ f_θ(xᵢ, yᵢ)

    Args:
        f: A function that takes params and batch and returns a 1D vector
            with length equal to batch size.
        params: PyTree of tensors.
        batch: Input data to f, of the form (x, y).

    Returns:
        The empirical Fisher information matrix.
    """
    jac, aux = jacrev(f, has_aux=True)(params, batch)

    # Convert Jacobian to be flat in parameter dimension
    jac = torch.vmap(lambda x: tree_ravel(x)[0])(jac)

    return jac.T @ jac, aux