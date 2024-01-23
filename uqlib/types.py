from typing import NamedTuple, Protocol, Any, TypeAlias
from optree.typing import PyTreeTypeVar
from torch import Tensor

TensorTree: TypeAlias = PyTreeTypeVar("TensorTree", Tensor)

TransformState: TypeAlias = TensorTree


class InitFn(Protocol):
    @staticmethod
    def __call__(
        params: TensorTree,
    ) -> TransformState:
        """Initiate a state.

        Args:
            params: The initial value of parameters.

        Returns:
            The initial state.
        """


class UpdateFn(Protocol):
    @staticmethod
    def __call__(
        state: TransformState,
        batch: Any,
        inplace: bool = False,
    ) -> TransformState:
        """Transform a state.

        Args:
            state: The state of the iterative algorithm.
            batch: The data batch.
            inplace: Whether to modify state using inplace operations. Defaults to True.

        Returns:
            The transformed state.
        """


class Transform(NamedTuple):
    """A transform contains init and update functions defining an iterative algorithm.

    Args:
        init: The init function.
        update: The update function.

    Returns:
        A transform.
    """

    init: InitFn
    update: UpdateFn
