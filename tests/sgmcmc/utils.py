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
    rtol: float = 1e-2,  # Relative reduction of KL for final distribution compared to initial distribution
):
    # Load log posterior
    log_prob, (mean, cov) = get_multivariate_normal_log_prob(dim)

    # Build transform
    transform = transform_builder(log_prob)

    # Initialise parameters far away
    init_var = 100
    params = torch.randn(dim) * init_var**0.5

    # Run transform
    state = transform.init(params)
    all_states = torch.unsqueeze(state, 0)
    for _ in range(n_steps):
        state, _ = transform.update(state, None)
        all_states = torch.cat((all_states, torch.unsqueeze(state, 0)), dim=0)

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
    assert (
        kl_init * rtol > kl_inferred
    )  # KL[samples || true] < rtol * KL[initial || true]

    # Also check the KL distance between the true and
    # inferred sample Gaussian generally decreases as we get more samples.
    cumulative_mean, cumulative_cov = cumulative_mean_and_cov(all_states.params)
    kl_divs = torch.func.vmap(utils.kl_gaussians, in_dims=(0, 0, None, None))(
        cumulative_mean, cumulative_cov, mean, cov
    )

    start_num_samples = 500  # Omit first few KLs with high variance due to few samples
    spacing = 100
    kl_divs_spaced = kl_divs[start_num_samples::spacing]
    spaced_decreasing = kl_divs_spaced[:-1] > kl_divs_spaced[1:]
    proportion_decreasing = spaced_decreasing.float().mean()
    assert proportion_decreasing > 0.8


def cumulative_mean_and_cov(xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = xs.shape
    out_means = torch.zeros((n, d))
    out_covs = torch.zeros((n, d, d))

    out_means[0] = xs[0]
    out_covs[0] = torch.eye(d)

    for i in range(1, n):
        n = i + 1
        out_means[i] = out_means[i - 1] * n / (n + 1) + xs[i] / (n + 1)

        delta_n = xs[i] - out_means[i - 1]
        out_covs[i] = (
            out_covs[i - 1] * (n - 2) / (n - 1) + torch.outer(delta_n, delta_n) / n
        )

    return out_means, out_covs
