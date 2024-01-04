from typing import Callable, Any, NamedTuple
import torch
from torch.func import grad_and_value, vmap
from optree import tree_map
import torchopt

from uqlib.utils import diag_normal_log_prob, diag_normal_sample


class VIDiagState(NamedTuple):
    """State encoding a diagonal Normal variational distribution over parameters.

    Args:
        mean: Mean of the variational distribution.
        log_sd_diag: Log of the square-root diagonal of the covariance matrix of the
            variational distribution.
        optimizer_state: torchopt state storing optimizer data for updating the
            variational parameters.
        nelbo: Negative evidence lower bound (lower is better).
    """

    mean: Any
    log_sd_diag: Any
    optimizer_state: tuple
    nelbo: float = 0


def init(
    init_mean: Any,
    optimizer: torchopt.base.GradientTransformation,
    init_log_sds: Any = None,
) -> VIDiagState:
    """Initialise diagonal Normal variational distribution over parameters.

    optimizer.initi will be called on flattened variational parameters so hyperparameters
    such as learning rate need to prespecifed through torchopt's functional API:

    ```
    import torchopt

    optimizer = torchopt.adam(lr=1e-2)
    vi_state = init(init_mean, optimizer)
    ```

    It's assumed maximize=False for the optimizer, so that we minimize the NELBO.

    Args:
        init_mean: Initial mean of the variational distribution.
        optimizer: torchopt functional optimizer for updating the variational
            parameters.
        init_log_sds: Initial log of the square-root diagonal of the covariance matrix
            of the variational distribution. Defaults to zero.

    Returns:
        Initial DiagVIState.
    """
    if init_log_sds is None:
        init_log_sds = tree_map(
            lambda x: torch.zeros_like(x, requires_grad=True), init_mean
        )

    optimizer_state = optimizer.init([init_mean, init_log_sds])
    return VIDiagState(init_mean, init_log_sds, optimizer_state)


def update(
    state: VIDiagState,
    log_posterior: Callable[[Any, Any], float],
    batch: Any,
    optimizer: torchopt.base.GradientTransformation,
    n_samples: int = 1,
    stl: bool = True,
    inplace: bool = True,
) -> VIDiagState:
    """Updates the variational parameters to minimise the ELBO.

    log_posterior expects to take parameters and input batch and return a scalar
    unbiased estimate of the full batch log posterior:

    ```
    val = log_posterior(params, batch)
    ```

    Args:
        state: Current state.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised).
        batch: Input data to log_posterior.
        optimizer: torchopt functional optimizer for updating the variational
            parameters.
        n_samples: Number of samples to use for Monte Carlo estimate.
        stl: Whether to use the `stick-the-landing` estimator
            https://arxiv.org/abs/1703.09194.
        inplace: Whether to update the state parameters in-place.

    Returns:
        Updated DiagVIState.
    """
    sd_diag = tree_map(torch.exp, state.log_sd_diag)
    with torch.no_grad():
        nelbo_grads, nelbo_val = grad_and_value(nelbo, argnums=(0, 1))(
            state.mean, sd_diag, log_posterior, batch, n_samples, stl
        )

    updates, optimizer_state = optimizer.update(
        nelbo_grads, state.optimizer_state, params=[state.mean, state.log_sd_diag]
    )
    mean, log_sd_diag = torchopt.apply_updates(
        (state.mean, state.log_sd_diag), updates, inplace=inplace
    )
    return VIDiagState(mean, log_sd_diag, optimizer_state, nelbo_val.item())


def nelbo(
    mean: dict,
    sd_diag: dict,
    log_posterior: Callable[[Any, Any], float],
    batch: Any,
    n_samples: int = 1,
    stl: bool = True,
) -> float:
    """Returns the evidence lower bound (ELBO) for a diagonal Normal
    variational distribution over the parameters of a model.

    Averages ELBO over the batch. Monte Carlo estimate with n_samples from q.

    ELBO = E_q[log p(y|x, θ) + log p(θ) - log q(θ)]

    log_posterior expects to take parameters and input batch and return a scalar:

    ```
    log_posterior_eval = log_posterior(params, batch)
    ```

    Args:
        mean: Mean of the variational distribution.
        sd_diag: Square-root diagonal of the covariance matrix of the
            variational distribution.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised) for each batch member.
        batch: Input data to log_posterior.
        n_samples: Number of samples to use for Monte Carlo estimate.
        stl: Whether to use the `stick-the-landing` estimator
            https://arxiv.org/abs/1703.09194.

    Returns:
        The sampled approximate ELBO averaged over the batch.
    """
    sampled_params = diag_normal_sample(mean, sd_diag, sample_shape=(n_samples,))
    if stl:
        mean = tree_map(lambda x: x.detach(), mean)
        sd_diag = tree_map(lambda x: x.detach(), sd_diag)

    log_p = vmap(log_posterior, (0, None))(sampled_params, batch)
    log_q = vmap(diag_normal_log_prob, (0, None, None))(sampled_params, mean, sd_diag)
    return -(log_p - log_q).mean()


def sample(state: VIDiagState, sample_shape: torch.Size = torch.Size([])):
    """Single sample from diagonal Normal distribution over parameters.

    Args:
        state: State encoding mean and log standard deviations.

    Returns:
        Sample from Normal distribution.
    """
    sd_diag = tree_map(torch.exp, state.log_sd_diag)
    return diag_normal_sample(state.mean, sd_diag, sample_shape=sample_shape)
