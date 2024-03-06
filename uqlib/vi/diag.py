from typing import Callable, Any, Tuple
from functools import partial
import torch
from torch.func import grad_and_value, vmap
from optree import tree_map
import torchopt
from dataclasses import dataclass

from uqlib.types import TensorTree, Transform, LogProbFn
from uqlib.utils import diag_normal_log_prob, diag_normal_sample


@dataclass
class VIDiagState:
    """State encoding a diagonal Normal variational distribution over parameters.

    Args:
        mean: Mean of the variational distribution.
        log_sd_diag: Log of the square-root diagonal of the covariance matrix of the
            variational distribution.
        optimizer_state: torchopt state storing optimizer data for updating the
            variational parameters.
        nelbo: Negative evidence lower bound (lower is better).
        aux: Auxiliary information from the log_posterior call.
    """

    mean: TensorTree
    log_sd_diag: TensorTree
    optimizer_state: tuple
    nelbo: torch.tensor = None
    aux: Any = None


def init(
    params: TensorTree,
    optimizer: torchopt.base.GradientTransformation,
    init_log_sds: TensorTree | None = None,
) -> VIDiagState:
    """Initialise diagonal Normal variational distribution over parameters.

    optimizer.init will be called on flattened variational parameters so hyperparameters
    such as learning rate need to prespecifed through torchopt's functional API:

    ```
    import torchopt

    optimizer = torchopt.adam(lr=1e-2)
    vi_state = init(init_mean, optimizer)
    ```

    It's assumed maximize=False for the optimizer, so that we minimize the NELBO.

    Args:
        params: Initial mean of the variational distribution.
        optimizer: torchopt functional optimizer for updating the variational
            parameters.
        init_log_sds: Initial log of the square-root diagonal of the covariance matrix
            of the variational distribution. Defaults to zero.

    Returns:
        Initial DiagVIState.
    """
    if init_log_sds is None:
        init_log_sds = tree_map(
            lambda x: torch.zeros_like(x, requires_grad=True), params
        )

    optimizer_state = optimizer.init([params, init_log_sds])
    return VIDiagState(params, init_log_sds, optimizer_state)


def update(
    state: VIDiagState,
    batch: Any,
    log_posterior: LogProbFn,
    optimizer: torchopt.base.GradientTransformation,
    temperature: float = 1.0,
    n_samples: int = 1,
    stl: bool = True,
    inplace: bool = False,
) -> VIDiagState:
    """Updates the variational parameters to minimise the NELBO.

    Args:
        state: Current state.
        batch: Input data to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        optimizer: torchopt functional optimizer for updating the variational
            parameters.
        temperature: Temperature to rescale (divide) log_posterior.
            Defaults to 1.
        n_samples: Number of samples to use for Monte Carlo estimate.
            Defaults to 1.
        stl: Whether to use the `stick-the-landing` estimator
            https://arxiv.org/abs/1703.09194.
            Defaults to True.
        inplace: Whether to modify state in place.

    Returns:
        Updated DiagVIState.
    """

    def nelbo_log_sd(m, lsd):
        sd_diag = tree_map(torch.exp, lsd)
        return nelbo(m, sd_diag, batch, log_posterior, temperature, n_samples, stl)

    with torch.no_grad():
        nelbo_grads, (nelbo_val, aux) = grad_and_value(
            nelbo_log_sd, argnums=(0, 1), has_aux=True
        )(state.mean, state.log_sd_diag)

    updates, optimizer_state = optimizer.update(
        nelbo_grads,
        state.optimizer_state,
        params=[state.mean, state.log_sd_diag],
        inplace=inplace,
    )
    mean, log_sd_diag = torchopt.apply_updates(
        (state.mean, state.log_sd_diag), updates, inplace=inplace
    )

    if inplace:
        state.nelbo = nelbo_val.detach()
        state.aux = aux
        return state
    return VIDiagState(mean, log_sd_diag, optimizer_state, nelbo_val.detach(), aux)


def build(
    log_posterior: Callable[[TensorTree, Any], float],
    optimizer: torchopt.base.GradientTransformation,
    temperature: float = 1.0,
    n_samples: int = 1,
    stl: bool = True,
    init_log_sds: TensorTree | None = None,
) -> Transform:
    """Builds a transform for variational inference with a diagonal Normal
    distribution over parameters.

    Args:
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior (which can be unnormalised).
        optimizer: torchopt functional optimizer for updating the variational
            parameters.
        temperature: Temperature to rescale (divide) log_posterior.
            Defaults to 1.
        n_samples: Number of samples to use for Monte Carlo estimate.
            Defaults to 1.
        stl: Whether to use the `stick-the-landing` estimator
            https://arxiv.org/abs/1703.09194.
            Defaults to True.
        init_log_sds: Initial log of the square-root diagonal of the covariance matrix
            of the variational distribution. Defaults to zero.

    Returns:
        Diagonal VI transform (uqlib.types.Transform instance).
    """
    init_fn = partial(init, optimizer=optimizer, init_log_sds=init_log_sds)
    update_fn = partial(
        update,
        log_posterior=log_posterior,
        optimizer=optimizer,
        temperature=temperature,
        n_samples=n_samples,
        stl=stl,
    )
    return Transform(init_fn, update_fn)


def nelbo(
    mean: dict,
    sd_diag: dict,
    batch: Any,
    log_posterior: LogProbFn,
    temperature: float = 1.0,
    n_samples: int = 1,
    stl: bool = True,
) -> Tuple[float, Any]:
    """Returns the negative evidence lower bound (NELBO) for a diagonal Normal
    variational distribution over the parameters of a model.

    Averages NELBO over the batch. Monte Carlo estimate with n_samples from q.

    NELBO = - (E_q[log p(y|x, θ) + log p(θ) - log q(θ) * temperature])

    log_posterior expects to take parameters and input batch and return a scalar:

    ```
    log_posterior_eval = log_posterior(params, batch)
    ```

    Args:
        mean: Mean of the variational distribution.
        sd_diag: Square-root diagonal of the covariance matrix of the
            variational distribution.
        batch: Input data to log_posterior.
        log_posterior: Function that takes parameters and input batch and
            returns the log posterior value (which can be unnormalised)
            as well as auxiliary information, e.g. from the model call.
        temperature: Temperature to rescale (divide) log_posterior.
            Defaults to 1.
        n_samples: Number of samples to use for Monte Carlo estimate.
            Defaults to 1.
        stl: Whether to use the `stick-the-landing` estimator
            https://arxiv.org/abs/1703.09194.
            Defaults to True.

    Returns:
        The sampled approximate NELBO averaged over the batch.
    """
    sampled_params = diag_normal_sample(mean, sd_diag, sample_shape=(n_samples,))
    if stl:
        mean = tree_map(lambda x: x.detach(), mean)
        sd_diag = tree_map(lambda x: x.detach(), sd_diag)

    log_p, aux = vmap(log_posterior, (0, None), (0, 0))(sampled_params, batch)
    log_q = vmap(diag_normal_log_prob, (0, None, None))(sampled_params, mean, sd_diag)
    return -(log_p - log_q * temperature).mean(), aux


def sample(state: VIDiagState, sample_shape: torch.Size = torch.Size([])):
    """Single sample from diagonal Normal distribution over parameters.

    Args:
        state: State encoding mean and log standard deviations.

    Returns:
        Sample from Normal distribution.
    """
    sd_diag = tree_map(torch.exp, state.log_sd_diag)
    return diag_normal_sample(state.mean, sd_diag, sample_shape=sample_shape)
