from functools import partial
import torch
from optree import tree_map
from optree.integration.torch import tree_ravel

from uqlib.sgmcmc import sghmc

from tests.scenarios import batch_normal_log_prob


def test_sghmc():
    torch.manual_seed(42)
    target_mean = {"a": torch.randn(2, 1) + 10, "b": torch.randn(1, 1) + 10}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)

    target_mean_flat = tree_ravel(target_mean)[0]
    target_cov = torch.diag(tree_ravel(target_sds)[0] ** 2)

    batch = torch.arange(10).reshape(-1, 1)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    n_steps = 10000
    lr = 1e-2
    alpha = 1.0
    beta = 0.0

    params = tree_map(lambda x: torch.zeros_like(x), target_mean)
    init_params_copy = tree_map(lambda x: x.clone(), params)

    sampler = sghmc.build(batch_normal_log_prob_spec, lr=lr, alpha=alpha, beta=beta)

    # Test inplace = False
    sghmc_state = sampler.init(params)
    log_posts = []
    all_params = tree_map(lambda x: x.unsqueeze(0), params)

    for _ in range(n_steps):
        sghmc_state = sampler.update(sghmc_state, batch, inplace=False)

        all_params = tree_map(
            lambda x, y: torch.cat((x, y.unsqueeze(0))), all_params, sghmc_state.params
        )

        log_posts.append(sghmc_state.log_posterior.item())

    burnin = 1000
    all_params_flat = torch.vmap(lambda x: tree_ravel(x)[0])(all_params)
    sampled_mean = all_params_flat[burnin:].mean(0)
    sampled_cov = torch.cov(all_params_flat[burnin:].T)

    assert log_posts[-1] > log_posts[0]
    assert torch.allclose(sampled_mean, target_mean_flat, atol=1e-0, rtol=1e-1)
    assert torch.allclose(sampled_cov, target_cov, atol=1e-0, rtol=1e-1)
    assert tree_map(
        lambda x, y: torch.all(x == y), params, init_params_copy
    )  # Check that the parameters are not updated

    # Test inplace = True
    sghmc_state = sampler.init(params, momenta=0.0)
    log_posts = []
    all_params = tree_map(lambda x: x.unsqueeze(0), params)

    for _ in range(n_steps):
        sghmc_state = sampler.update(sghmc_state, batch, inplace=True)

        all_params = tree_map(
            lambda x, y: torch.cat((x, y.unsqueeze(0))), all_params, sghmc_state.params
        )

        log_posts.append(sghmc_state.log_posterior.item())

    burnin = 1000
    all_params_flat = torch.vmap(lambda x: tree_ravel(x)[0])(all_params)
    sampled_mean = all_params_flat[burnin:].mean(0)
    sampled_cov = torch.cov(all_params_flat[burnin:].T)

    assert log_posts[-1] > log_posts[0]
    assert torch.allclose(sampled_mean, target_mean_flat, atol=1e-0, rtol=1e-1)
    assert torch.allclose(sampled_cov, target_cov, atol=1e-0, rtol=1e-1)
    assert tree_map(
        lambda x, y: torch.all(x != y), params, init_params_copy
    )  # Check that the parameters are updated
