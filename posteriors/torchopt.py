from typing import Any
from functools import partial
import torch
import torchopt
from dataclasses import dataclass

from posteriors.types import TensorTree, Transform, LogProbFn, TransformState
from posteriors.utils import CatchAuxError


def build(
    loss_fn: LogProbFn,
    optimizer: torchopt.base.GradientTransformation,
) -> Transform:
    """Build a [TorchOpt](https://github.com/metaopt/torchopt) optimizer transformation.

    Make sure to use the lower case functional optimizers e.g. `torchopt.adam()`.

    ```
    transform = build(loss_fn, torchopt.adam(lr=0.1))
    state = transform.init(params)

    for batch in dataloader:
        state = transform.update(state, batch)
    ```

    Args:
        loss_fn: Loss function.
        optimizer: TorchOpt functional optimizer.
            Make sure to use lower case e.g. torchopt.adam()

    Returns:
        Torchopt optimizer transform instance.
    """
    init_fn = partial(init, optimizer=optimizer)
    update_fn = partial(update, optimizer=optimizer, loss_fn=loss_fn)
    return Transform(init_fn, update_fn)


@dataclass
class TorchOptState(TransformState):
    """State of a [TorchOpt](https://github.com/metaopt/torchopt) optimizer.

    Contains the parameters, the optimizer state for the TorchOpt optimizer,
    loss value, and auxiliary information.

    Args:
        params: Parameters to be optimized.
        opt_state: TorchOpt optimizer state.
        loss: Loss value.
        aux: Auxiliary information from the loss function call.
    """

    params: TensorTree
    opt_state: torchopt.typing.OptState
    loss: torch.tensor = None
    aux: Any = None


def init(
    params: TensorTree,
    optimizer: torchopt.base.GradientTransformation,
) -> TorchOptState:
    """Initialise a [TorchOpt](https://github.com/metaopt/torchopt) optimizer.

    Make sure to use the lower case functional optimizers e.g. `torchopt.adam()`.

    Args:
        params: Parameters to be optimized.
        optimizer: TorchOpt functional optimizer.
            Make sure to use lower case e.g. torchopt.adam()

    Returns:
        Initial TorchOptState.
    """
    opt_state = optimizer.init(params)
    return TorchOptState(params, opt_state)


def update(
    state: TorchOptState,
    batch: TensorTree,
    loss_fn: LogProbFn,
    optimizer: torchopt.base.GradientTransformation,
    inplace: bool = True,
) -> TorchOptState:
    """Update the [TorchOpt](https://github.com/metaopt/torchopt) optimizer state.

    Make sure to use the lower case functional optimizers e.g. `torchopt.adam()`.

    Args:
        state: Current state.
        batch: Batch of data.
        loss_fn: Loss function.
        optimizer: TorchOpt functional optimizer.
            Make sure to use lower case like torchopt.adam()
        inplace: Whether to update the state in place.

    Returns:
        Updated TorchOptState.
    """
    params = state.params
    opt_state = state.opt_state
    with torch.no_grad(), CatchAuxError():
        grads, (loss, aux) = torch.func.grad_and_value(loss_fn, has_aux=True)(
            params, batch
        )
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates, inplace=inplace)
    if inplace:
        state.loss = loss.detach()
        state.aux = aux
        return state
    return TorchOptState(params, opt_state, loss, aux)
