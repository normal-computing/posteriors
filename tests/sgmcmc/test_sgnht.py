from functools import partial
import torch
from posteriors.sgmcmc import sgnht
from tests import scenarios
from tests.utils import verify_inplace_update
from tests.sgmcmc.utils import run_test_sgmcmc_gaussian


def test_sgnht():
    torch.manual_seed(42)

    # Set inference parameters (with torch.optim.SGD parameterization)
    lr = 1e-2
    mu = 0.9
    tau = 0.9

    eps = 1 - tau
    sigma = (lr / (1 - tau)) ** -0.5
    alpha = (1 - mu) / lr

    beta = 0.0
    momenta = 0.0

    # Run MCMC test on Gaussian
    run_test_sgmcmc_gaussian(
        partial(
            sgnht.build,
            lr=eps,
            alpha=alpha,
            sigma=sigma,
            beta=beta,
            momenta=momenta,
        ),
    )


def test_sgnht_inplace_step():
    torch.manual_seed(42)

    # Load log posterior
    dim = 5
    log_prob, _ = scenarios.get_multivariate_normal_log_prob(dim)

    # Set inference parameters
    def lr(step):
        return 1e-2 * (step + 1) ** -0.33

    # Build transform
    transform = sgnht.build(log_prob, lr)

    # Initialise
    params = torch.randn(dim)

    # Verify inplace update
    verify_inplace_update(transform, params, None)
