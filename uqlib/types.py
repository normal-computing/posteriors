from typing import NamedTuple, Protocol, Any, TypeAlias, Tuple, Callable
from optree.typing import PyTreeTypeVar
from torch import Tensor

TensorTree: TypeAlias = PyTreeTypeVar("TensorTree", Tensor)

TransformState: TypeAlias = NamedTuple

LogProbFn = Callable[[TensorTree, TensorTree], Tuple[float, TensorTree]]


class InitFn(Protocol):
    @staticmethod
    def __call__(
        params: TensorTree,
    ) -> TransformState:
        """Initiate a uqlib state with unified API:

        ```
        state = init(params)
        ```

        where params is a PyTree of parameters around which we want to
        quantify uncertainty. The produced `state` is a `NamedTuple` containing
        the required information for the uqlib iterative algorithm
        defined by the `init` and `update` functions.

        See also uqlib.types.UpdateFn and uqlib.types.Transform.

        Args:
            params: PyTree containing initial value of parameters.

        Returns:
            The initial state (NamedTuple).
        """


class UpdateFn(Protocol):
    @staticmethod
    def __call__(
        state: TransformState,
        batch: Any,
        inplace: bool = True,
    ) -> TransformState:
        """Transform a uqlib state with unified API:

        ```
        state = update(state, batch, inplace=False)
        ```

        where state is a `NamedTuple` containing the required information for the
        uqlib iterative algorithm defined by the `init` and `update` functions.

        See also uqlib.types.InitFn and uqlib.types.Transform.

        Args:
            state: The state of the iterative algorithm.
            batch: The data batch.
            inplace: Whether to modify state using inplace operations. Defaults to True.

        Returns:
            The transformed state (NamedTuple).
        """


class Transform(NamedTuple):
    """A transform contains init and update functions defining an iterative algorithm.

    See also uqlib.types.InitFn and uqlib.types.UpdateFn.

    Args:
        init: The init function.
        update: The update function.

    Returns:
        A transform (`NamedTuple` containing `init` and `update` functions).
    """

    init: InitFn
    update: UpdateFn
