from functools import partial
import torch
from posteriors.sgmcmc import sgnht
from tests import scenarios
from tests.utils import verify_inplace_update
from tests.sgmcmc.utils import run_test_sgmcmc_gaussian


def test_sgnht():
    torch.manual_seed(42)

    # Set inference parameters
    lr = 1e-2
    alpha = 0.1
    sigma = 1.0
    temperature = 1.0
    beta = 0.0

    # Run MCMC test on Gaussian
    run_test_sgmcmc_gaussian(
        partial(
            sgnht.build,
            lr=lr,
            alpha=alpha,
            sigma=sigma,
            temperature=temperature,
            beta=beta,
        ),
    )


def test_sgnht_inplace_step():
    torch.manual_seed(42)

    # Load log posterior
    dim = 5
    log_prob, _ = scenarios.get_multivariate_normal_log_prob(dim)

    # Set inference parameters
    lr = 1e-2

    # Build transform
    transform = sgnht.build(log_prob, lr)

    # Initialise
    params = torch.randn(dim)

    # Verify inplace update
    verify_inplace_update(transform, params, None)
