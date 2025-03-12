from typing import Callable
import torch
from posteriors.types import LogProbFn, Transform
from tests import scenarios


def run_test_mcmc_gaussian(
    build_transform: Callable[[LogProbFn], Transform],
    dim: int = 5,
    n_steps: int = 10_000,
    burnin: int = 1_000,
):
    torch.manual_seed(42)

    # Load log posterior
    log_prob, (mean, cov) = scenarios.get_multivariate_normal_log_prob(dim)

    # Build transform
    transform = build_transform(log_prob)

    # Initialise parameters
    params = torch.randn(dim)

    # Run transform
    all_states = scenarios.run_transform(transform, params, n_steps)

    # Remove burnin
    all_states = all_states[burnin:]

    # SGMCMC can be quite noisy, so we'll check the KL distance between the true and
    # inferred sample Gaussian decreases as we get more samples.
    cumulative_mean, cumulative_cov = scenarios.cumulative_mean_and_cov(
        all_states.params
    )
    kl_divs = torch.func.vmap(scenarios.kl_gaussians, in_dims=(0, 0, None, None))(
        cumulative_mean, cumulative_cov, mean, cov
    )
    # Check generally decreasing by checking n_check equally spaced KL divs
    start_num_samples = 10
    n_check = 5
    cum_check_inds = torch.linspace(start_num_samples, n_steps, n_check)
    cum_check_inds = cum_check_inds.int()

    def get_kl_div(ind):
        m = all_states.params[:ind].mean(0)
        c = all_states.params[:ind].T.cov()
        return scenarios.kl_gaussians(m, c, mean, cov)

    kl_divs = torch.stack([get_kl_div(ind) for ind in cum_check_inds])
    assert torch.all(kl_divs[:-1] > kl_divs[1:])
