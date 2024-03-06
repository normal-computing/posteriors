from typing import Type, Any
from functools import partial
import torch
from dataclasses import dataclass

from uqlib.types import TensorTree, Transform, LogProbFn


@dataclass
class OptimState:
    """State of an optimizer.

    Args:
        params: Parameters to be optimised.
        optimizer: torch.optim optimizer instance.
        loss: Loss value.
        aux: Auxiliary information from the loss function call.
    """

    params: TensorTree
    optimizer: torch.optim.Optimizer
    loss: torch.tensor = None
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
    state.loss = loss
    state.aux = aux
    return state


def build(
    loss_fn: LogProbFn,
    optimizer: Type[torch.optim.Optimizer],
    **kwargs: Any,
) -> Transform:
    """Builds an optimizer transform from torch.optim.

    Example usage:

    ```
    transform = build(loss_fn, torch.optim.Adam, lr=0.1)
    state = transform.init(params)

    for batch in dataloader:
        state = transform.update(state, batch)
    ```

    Arg:
        loss_fn: Function that takes the parameters and returns the loss.
            of the form `loss, aux = fn(params, batch)`.
        optimizer: Optimizer class from torch.optim.
        **kwargs: Keyword arguments to pass to the optimizer class.

    Returns:
        Optimizer transform (uqlib.types.Transform instance).
    """
    init_fn = partial(init, optimizer_cls=optimizer, **kwargs)
    update_fn = partial(update, loss_fn=loss_fn)
    return Transform(init_fn, update_fn)
