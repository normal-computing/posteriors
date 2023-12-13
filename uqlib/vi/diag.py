from typing import Callable, Any, NamedTuple, Type
import torch
from torch.utils._pytree import tree_flatten

from uqlib.utils import tree_map, diag_normal_log_prob, diag_normal_sample


class VIDiagState(NamedTuple):
    """State encoding a diagonal Normal variational distribution over parameters.

    Args:
        mean: Mean of the variational distribution.
        log_sd_diag: Log of the square-root diagonal of the covariance matrix of the
            variational distribution.
        optimizer: torch.optim.Optimizer instance for updating the
            variational parameters.
        elbo: Evidence lower bound (higher is better).
    """

    mean: Any
    log_sd_diag: Any
    optimizer: torch.optim.Optimizer
    elbo: float = 0


def init(
    init_mean: Any,
    optimizer_cls: Type[torch.optim.Optimizer],
    init_log_sds: Any = None,
) -> VIDiagState:
    """Initialise diagonal Normal variational distribution over parameters.

    optimizer_cls will be called on flattened variational parameters so hyperparameters
    such as learning rate need to prespecifed with e.g.

    ```
    from functools import partial
    from torch.optim import Adam

    optimizer_cls = partial(Adam, lr=0.01)
    vi_state = init(init_mean, optimizer_cls)
    ```

    Args:
        init_mean: Initial mean of the variational distribution.
        optimizer_cls: Optimizer class for updating the variational parameters.
        init_log_sds: Initial log of the square-root diagonal of the covariance matrix
            of the variational distribution. Defaults to zero.

    Returns:
        Initial DiagVIState.
    """
    if init_log_sds is None:
        init_log_sds = tree_map(
            lambda x: torch.zeros_like(x).requires_grad_(True), init_mean
        )

    mean_leaves = tree_flatten(init_mean)[0]
    init_log_sds_leaves = tree_flatten(init_log_sds)[0]
    optimizer = optimizer_cls(mean_leaves + init_log_sds_leaves, maximize=True)
    return VIDiagState(init_mean, init_log_sds, optimizer)


def update(
    state: VIDiagState,
    log_posterior: Callable[[Any, Any], float],
    batch: Any,
    n_samples: int = 1,
) -> VIDiagState:
    """Updates the variational parameters to minimise the ELBO.

    log_posterior expects to take parameters and input batch and return the scalar log
    posterior:

    ```
    val = log_posterior(params, batch)
    ```

    Args:
        state: Current state.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised).
        batch: Input data to log_posterior.
        n_samples: Number of samples to use for Monte Carlo estimate.

    Returns:
        Updated DiagVIState.
    """
    state.optimizer.zero_grad()
    sd_diag = tree_map(torch.exp, state.log_sd_diag)
    elbo_val = elbo(log_posterior, batch, state.mean, sd_diag, n_samples)
    elbo_val.backward()
    state.optimizer.step()
    return VIDiagState(state.mean, state.log_sd_diag, state.optimizer, elbo_val.item())


def elbo(
    log_posterior: Callable[[Any, Any], float],
    batch: Any,
    mean: dict,
    sd_diag: dict,
    n_samples: int = 1,
) -> float:
    """Returns the evidence lower bound (ELBO) for a diagonal Normal
    variational distribution over the parameters of a model.

    Averages ELBO over the batch. Monte Carlo estimate with n_samples from q.

    ELBO = E_q[log p(y|x, θ) + log p(θ) - log q(θ)]

    log_posterior expects to take parameters and input batch and return the scalar log
    posterior:

    ```
    val = log_posterior(params, batch)
    ```

    Args:
        model: Model.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised).
        batch: Input data to log_posterior.
        mean: Mean of the variational distribution.
        sd_diag: Square-root diagonal of the covariance matrix of the
            variational distribution.
        n_samples: Number of samples to use for Monte Carlo estimate.

    Returns:
        The sampled approximate ELBO averaged over the batch.
    """

    def single_elbo(sampled_params):
        log_p = log_posterior(sampled_params, batch).mean()
        log_q = diag_normal_log_prob(sampled_params, mean, sd_diag)
        return log_p - log_q

    # Maybe we should change this loop to a vmap? If its supported
    elbo = 0
    for _ in range(n_samples):
        elbo += single_elbo(diag_normal_sample(mean, sd_diag)) / n_samples
    return elbo


def sample(state: VIDiagState):
    """Single sample from diagonal Normal distribution over parameters.

    Args:
        state: State encoding mean and log standard deviations.

    Returns:
        Sample from Normal distribution.
    """
    sd_diag = tree_map(torch.exp, state.log_sd_diag)
    return diag_normal_sample(state.mean, sd_diag)
