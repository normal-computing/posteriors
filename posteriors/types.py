from typing import (
    Protocol,
    Any,
    Tuple,
    Callable,
    NamedTuple,
)
from optree import PyTree
from torch import Tensor
from tensordict import TensorClass

# TensorTree = PyTree with Tensor leaves.
# See https://optree.readthedocs.io/en/latest/typing.html#optree.PyTree
TensorTree = PyTree[Tensor]

LogProbFn = Callable[[TensorTree, TensorTree], Tuple[float, TensorTree]]
ForwardFn = Callable[[TensorTree, TensorTree], Tuple[Tensor, TensorTree]]
OuterLogProbFn = Callable[[TensorTree, TensorTree], float]

TensorLike = float | int | bool | Tensor
Schedule = Callable[
    [TensorLike], TensorLike
]  # Learning rate schedule (input is step index, output is scalar learning rate)


class InitFn(Protocol):
    def __call__(
        self,
        params: TensorTree,
    ) -> TensorClass:
        """Initiate a posteriors state with unified API:

        ```
        state = init(params)
        ```

        where params is a PyTree of parameters. The produced `state` is a
        `tensordict.TensorClass` containing the required information for the
        posteriors iterative algorithm defined by the `init` and `update` functions.

        Note that this represents the `init` function as stored in a `Transform`
        returned by an algorithm's `build` function, the internal `init` function in
        the algorithm module can and likely will have additional arguments.

        Args:
            params: PyTree containing initial value of parameters.

        Returns:
            The initial state, a `tensordict.tensorclass` with `params` and `aux`
            attributes but possibly other attributes too.
        """
        ...  # pragma: no cover


class UpdateFn(Protocol):
    def __call__(
        self,
        state: TensorClass,
        batch: Any,
        inplace: bool = False,
    ) -> tuple[TensorClass, TensorTree]:
        """Transform a posteriors state with unified API:

        ```
        state, aux = update(state, batch, inplace=False)
        ```

        where state is a `tensordict.TensorClass` containing the required information
        for the posteriors iterative algorithm defined by the `init` and `update`
        functions. `aux` is an arbitrary info object returned by the
        `log_posterior` or `log_likelihood` function.

        Note that this represents the `update` function as stored in a `Transform`
        returned by an algorithm's `build` function, the internal `update` function in
        the algorithm module can and likely will have additional arguments.

        Args:
            state: The `tensordict.tensorclass` state of the iterative algorithm.
            batch: The data batch.
            inplace: Whether to modify state using inplace operations. Defaults to True.

        Returns:
            Tuple of `state` and `aux`.
                `state` is a `tensordict.tensorclass` with `params` attributes
                but possibly other attributes too. Must be of the same type as
                the input state.
                `aux` is an arbitrary info object returned by the
                `log_posterior` or `log_likelihood` function.
        """
        ...  # pragma: no cover


class Transform(NamedTuple):
    """A transform contains `init` and `update` functions defining an iterative
        algorithm.

    Within the `Transform` all algorithm specific arguments are predefined, so that the
    `init` and `update` functions have a unified API:
    ```
    state = transform.init(params)
    state, aux = transform.update(state, batch, inplace=False)
    ```

    Note that this represents the `Transform` function is returned by an algorithm's
    `build` function, the internal `init` and `update` functions in the
    algorithm module can and likely will have additional arguments.

    Attributes:
        init: The init function.
        update: The update function.

    """

    init: InitFn
    update: UpdateFn
