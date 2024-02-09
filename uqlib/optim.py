from typing import Type, NamedTuple, Any
from functools import partial
import torch

from uqlib.types import TensorTree, Transform, LogProbFn


class OptimState(NamedTuple):
    """State of an optimizer.

    Args:
        params: Parameters to be optimised.
        optimizer: torch.optim optimizer instance.
        loss: Loss value.
        aux: Auxiliary information from the loss function call.
    """

    params: TensorTree
    optimizer: torch.optim.Optimizer
    loss: torch.tensor = torch.tensor(0.0)
    aux: Any = None


def init(
    params: TensorTree,
    optimizer_cls: Type[torch.optim.Optimizer],
    *args: Any,
    **kwargs: Any,
) -> OptimState:
    """Initialise an optimizer.

    Args:
        params: Parameters to be optimised.
        optimizer_cls: Optimizer class from torch.optim.
        *args: Positional arguments to pass to the optimizer class.
        **kwargs: Keyword arguments to pass to the optimizer class.

    Returns:
        Initial OptimState.
    """
    opt_params = [params] if isinstance(params, torch.Tensor) else params

    optimizer = optimizer_cls(opt_params, *args, **kwargs)
    return OptimState(params, optimizer)


def update(
    state: OptimState,
    batch: TensorTree,
    loss_fn: LogProbFn,
    inplace: bool = True,
) -> OptimState:
    """Perform a single update step of the optimizer.

    Args:
        state: Current optimizer state.
        batch: Input data to loss_fn.
        loss_fn: Function that takes the parameters and returns the loss.
            of the form `loss, aux = fn(params, batch)`.
        inplace: Whether to update the parameters in place.
            inplace=False not supported for uqlib.optim

    Returns:
        Updated OptimState.
    """
    if not inplace:
        raise NotImplementedError("inplace=False not supported for uqlib.optim")
    state.optimizer.zero_grad()
    loss, aux = loss_fn(state.params, batch)
    loss.backward()
    state.optimizer.step()
    return OptimState(state.params, state.optimizer, state.loss.detach(), aux)


def build(
    optimizer_cls: Type[torch.optim.Optimizer],
    loss_fn: LogProbFn,
    **kwargs: Any,
) -> Transform:
    """Builds an optimizer transform from torch.optim.

    Example usage:

    ```
    transform = build(torch.optim.Adam, loss_fn, lr=0.1)
    state = transform.init(params)

    for batch in dataloader:
        state = transform.update(state, batch)
    ```

    Arg:
        optimizer_cls: Optimizer class from torch.optim.
        loss_fn: Function that takes the parameters and returns the loss.
            of the form `loss, aux = fn(params, batch)`.
        *args: Positional arguments to pass to the optimizer class.
        **kwargs: Keyword arguments to pass to the optimizer class.

    Returns:
        Optimizer transform (uqlib.types.Transform instance).
    """
    init_fn = partial(init, optimizer_cls=optimizer_cls, **kwargs)
    update_fn = partial(update, loss_fn=loss_fn)
    return Transform(init_fn, update_fn)
