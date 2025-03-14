from functools import partial
import torch
from posteriors.sgmcmc import baoa
from tests import scenarios
from tests.utils import verify_inplace_update
from tests.sgmcmc.utils import run_test_sgmcmc_gaussian


def test_baoa():
    torch.manual_seed(42)

    # Set inference parameters (with torch.optim.SGD parameterization)
    lr = 1e-3
    mu = 0.9
    tau = 0.9

    eps = 1 - tau
    sigma = (lr / (1 - tau)) ** -0.5
    alpha = (1 - mu) / lr

    temperature = 1.0

    # Run MCMC test on Gaussian
    run_test_sgmcmc_gaussian(
        partial(baoa.build, lr=eps, alpha=alpha, sigma=sigma, temperature=temperature),
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
    verify_inplace_update(transform, params, None)
