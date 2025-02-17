from typing import (
    Protocol,
    Any,
    TypeAlias,
    Tuple,
    Callable,
    NamedTuple,
)
from optree.typing import PyTreeTypeVar
from torch import Tensor
from tensordict import TensorClass

TensorTree: TypeAlias = PyTreeTypeVar("TensorTree", Tensor)

LogProbFn = Callable[[TensorTree, TensorTree], Tuple[float, TensorTree]]
ForwardFn = Callable[[TensorTree, TensorTree], Tuple[Tensor, TensorTree]]
OuterLogProbFn = Callable[[TensorTree, TensorTree], float]


class InitFn(Protocol):
    @staticmethod
    def __call__(
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
    @staticmethod
    def __call__(
        state: TensorClass,
        batch: Any,
        inplace: bool = False,
    ) -> TensorClass:
        """Transform a posteriors state with unified API:

        ```
        state = update(state, batch, inplace=False)
        ```

        where state is a `tensordict.TensorClass` containing the required information
        for the posteriors iterative algorithm defined by the `init` and `update`
        functions.

        Note that this represents the `update` function as stored in a `Transform`
        returned by an algorithm's `build` function, the internal `update` function in
        the algorithm module can and likely will have additional arguments.

        Args:
            state: The `tensordict.tensorclass` state of the iterative algorithm.
            batch: The data batch.
            inplace: Whether to modify state using inplace operations. Defaults to True.

        Returns:
            The transformed state, a `tensordict.tensorclass` with `params` and `aux`
            attributes but possibly other attributes too. Must be of the same type as
            the input state.
        """
        ...  # pragma: no cover


class Transform(NamedTuple):
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

    Attributes:
        init: The init function.
        update: The update function.

    """

    init: InitFn
    update: UpdateFn
