from typing import Protocol, Any, TypeAlias, Tuple, Callable
from dataclasses import dataclass, asdict
from optree.typing import PyTreeTypeVar
from optree import register_pytree_node_class, tree_flatten, tree_unflatten
from optree import registry
from torch import Tensor

TensorTree: TypeAlias = PyTreeTypeVar("TensorTree", Tensor)  # type: ignore

LogProbFn = Callable[[TensorTree, TensorTree], Tuple[float, TensorTree]]
ForwardFn = Callable[[TensorTree, TensorTree], Tuple[Tensor, TensorTree]]
OuterLogProbFn = Callable[[TensorTree, TensorTree], float]

namespace = registry.__GLOBAL_NAMESPACE


@register_pytree_node_class(namespace=namespace)
class TransformState:
    """A `posteriors` transform state is a `dataclass` containing the required
    information for the posteriors iterative algorithm defined by the `init` and
    `update` functions.

    Inherit the `TransformState` class to add the new `state` class to the optree
    PyNode registry to support functions like `optree.tree_map(lambda x: x**2, state)`.

    ```
    @dataclass
    class AlgorithmState(TransformState):
        params: TensorTree
        algorithm_info: Any
        aux: Any
    ```
    """

    params: TensorTree
    aux: Any

    def __init_subclass__(cls):
        super().__init_subclass__()
        register_pytree_node_class(cls, namespace=namespace)

    def tree_flatten(self):
        return tree_flatten(asdict(self))

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(**tree_unflatten(metadata, children))


class InitFn(Protocol):
    @staticmethod
    def __call__(
        params: TensorTree,
    ) -> TransformState:
        """Initiate a posteriors state with unified API:

        ```
        state = init(params)
        ```

        where params is a PyTree of parameters. The produced `state` is a
        `TransformState` (and `dataclass`) containing the required information for the
        posteriors iterative algorithm defined by the `init` and `update` functions.

        Note that this represents the `init` function as stored in a `Transform`
        returned by an algorithm's `build` function, the internal `init` function in
        the algorithm module can and likely will have additional arguments.

        Args:
            params: PyTree containing initial value of parameters.

        Returns:
            The initial state (dataclass).
        """


class UpdateFn(Protocol):
    @staticmethod
    def __call__(
        state: TransformState,
        batch: Any,
        inplace: bool = False,
    ) -> TransformState:
        """Transform a posteriors state with unified API:

        ```
        state = update(state, batch, inplace=False)
        ```

        where state is a `dataclass` containing the required information for the
        posteriors iterative algorithm defined by the `init` and `update` functions.

        Note that this represents the `update` function as stored in a `Transform`
        returned by an algorithm's `build` function, the internal `update` function in
        the algorithm module can and likely will have additional arguments.

        Args:
            state: The state of the iterative algorithm.
            batch: The data batch.
            inplace: Whether to modify state using inplace operations. Defaults to True.

        Returns:
            The transformed state (dataclass).
        """


@dataclass(frozen=True)
class Transform:
    """A transform contains `init` and `update` functions defining an iterative
        algorithm.

    Within the `Transform` all algorithm specific arguments are predefined, so that the
    `init` and `update` functions have a unified API:
    ```
    state = transform.init(params)
    state = transform.update(state, batch, inplace=False)
    ```

    Note that this represents the `Transform` function is returned by an algorithm's
    `build` function, the internal `init` and `update` functions in the
    algorithm module can and likely will have additional arguments.

    Args:
        init: The init function.
        update: The update function.

    """

    init: InitFn
    update: UpdateFn
