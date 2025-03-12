from functools import partial
import torch
from posteriors.sgmcmc import baoa
from tests import scenarios
from tests.sgmcmc.utils import run_test_sgmcmc_gaussian


def test_baoa():
    torch.manual_seed(42)

    # Set inference parameters
    lr = 1e-1
    alpha = 0.1
    sigma = 1.0
    temperature = 1.0

    # Run MCMC test on Gaussian
    run_test_sgmcmc_gaussian(
        partial(baoa.build, lr=lr, alpha=alpha, sigma=sigma, temperature=temperature),
    )


def test_baoa_inplace_step():
    torch.manual_seed(42)

    # Load log posterior
    dim = 5
    log_prob, _ = scenarios.get_multivariate_normal_log_prob(dim)

    # Set inference parameters
    lr = 1e-2

    # Build transform
    transform = baoa.build(log_prob, lr)

    # Initialise
    params = torch.randn(dim)

    # Verify inplace update
    scenarios.verify_inplace_update(transform, params, None)
