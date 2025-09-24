from functools import partial
import torch
from posteriors.sgmcmc import sgld
from tests import scenarios
from tests.utils import verify_inplace_update
from tests.sgmcmc.utils import run_test_sgmcmc_gaussian


def test_sgld():
    torch.manual_seed(42)

    # Set inference parameters
    lr = 1e-2
    beta = 0.0

    # Run MCMC test on Gaussian
    run_test_sgmcmc_gaussian(
        partial(sgld.build, lr=lr, beta=beta),
    )


def test_sgld_inplace_step():
    torch.manual_seed(42)

    # Load log posterior
    dim = 5
    log_prob, _ = scenarios.get_multivariate_normal_log_prob(dim)

    # Set inference parameters
    def lr(step):
        return 1e-2 * (step + 1) ** -0.33

    # Build transform
    transform = sgld.build(log_prob, lr)

    # Initialise
    params = torch.randn(dim)

    # Verify inplace update
    verify_inplace_update(transform, params, None)
