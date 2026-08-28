from functools import partial
import torch
from posteriors.sgmcmc import sglrw
from tests.scenarios import get_multivariate_normal_log_prob
from tests.utils import verify_inplace_update
from tests.sgmcmc.utils import run_test_sgmcmc_gaussian


def test_sglrw():
    torch.manual_seed(42)

    # Set inference parameters
    lr = 1e-2

    # Run MCMC test on Gaussian
    run_test_sgmcmc_gaussian(
        partial(sglrw.build, lr=lr),
    )


def test_sglrw_inplace_step():
    torch.manual_seed(42)

    # Load log posterior
    dim = 5
    log_prob, _ = get_multivariate_normal_log_prob(dim)

    # Set inference parameters
    def lr(step):
        return 1e-2 * (step + 1) ** -0.33

    # Build transform
    transform = sglrw.build(log_prob, lr)

    # Initialise
    params = {"w": torch.randn(2, 2), "b": torch.randn(1)}

    # Verify inplace update
    verify_inplace_update(transform, params, None)
