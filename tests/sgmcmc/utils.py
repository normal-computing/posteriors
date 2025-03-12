from typing import Callable
import torch
from posteriors.types import LogProbFn, Transform
from tests.scenarios import get_multivariate_normal_log_prob
from tests import utils


def run_test_sgmcmc_gaussian(
    transform_builder: Callable[[LogProbFn], Transform],
    dim: int = 3,
    n_steps: int = 15_000,
    burnin: int = 5_000,
    rtol: float = 1e-3,
):
    # Load log posterior
    log_prob, (mean, cov) = get_multivariate_normal_log_prob(dim)

    # Build transform
    transform = transform_builder(log_prob)

    # Initialise parameters far away
    init_var = 100
    params = torch.randn(dim) * init_var**0.5

    # Run transform
    all_states = utils.run_transform(transform, params, n_steps)

    # Remove burnin
    all_states = all_states[burnin:]

    # Check KL divergence between true and inferred Gaussian
    kl_init = utils.kl_gaussians(torch.zeros(dim), torch.eye(dim) * init_var, mean, cov)
    kl_inferred = utils.kl_gaussians(
        all_states.params.mean(0),
        all_states.params.T.cov(),
        mean,
        cov,
    )
    assert kl_init * rtol > kl_inferred

    # Also check the KL distance between the true and
    # inferred sample Gaussian generally decreases as we get more samples.
    cumulative_mean, cumulative_cov = utils.cumulative_mean_and_cov(all_states.params)
    kl_divs = torch.func.vmap(utils.kl_gaussians, in_dims=(0, 0, None, None))(
        cumulative_mean, cumulative_cov, mean, cov
    )

    start_num_samples = 100
    spacing = 500
    kl_divs_spaced = kl_divs[start_num_samples::spacing]
    spaced_decreasing = kl_divs_spaced[:-1] > kl_divs_spaced[1:]
    assert spaced_decreasing.float().mean() > 0.65  # >65% of spaced KLs are decreasing
