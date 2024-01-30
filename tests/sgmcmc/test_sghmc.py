from functools import partial
from typing import Any
import torch
from optree import tree_map

from uqlib.sgmcmc import sghmc
from uqlib.utils import diag_normal_log_prob


def batch_normal_log_prob(
    p: dict, batch: Any, mean: dict, sd_diag: dict
) -> torch.Tensor:
    return diag_normal_log_prob(p, mean, sd_diag)


def test_sghmc():
    target_mean = {"a": torch.randn(2, 1) + 10, "b": torch.randn(1, 1) + 10}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)
    batch = torch.arange(10).reshape(-1, 1)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    n_steps = 10000
    lr = 1e-2
    alpha = 1.0
    beta = 0.0

    params = tree_map(lambda x: torch.zeros_like(x), target_mean)

    sampler = sghmc.build(batch_normal_log_prob_spec, lr=lr, alpha=alpha, beta=beta)

    sghmc_state = sampler.init(params)

    log_posts = []

    all_params = tree_map(lambda x: x.unsqueeze(0), params)

    for _ in range(n_steps):
        sghmc_state = sampler.update(sghmc_state, batch)

        all_params = tree_map(
            lambda x, y: torch.cat((x, y.unsqueeze(0))), all_params, sghmc_state.params
        )

        log_posts.append(sghmc_state.log_posterior)

    burnin = 1000
    param_means = tree_map(lambda x: x[burnin:].mean(0), all_params)
    param_sds = tree_map(lambda x: x[burnin:].std(0), all_params)

    assert log_posts[-1] > log_posts[0]
    for k in target_mean.keys():
        assert torch.allclose(param_means[k], target_mean[k], atol=1e0, rtol=1e-1)
        assert torch.allclose(param_sds[k], target_sds[k], atol=1e0, rtol=1e-1)

    # Test inplace
    state = sampler.init(params)
    state2 = sampler.update(state, batch, inplace=True)
    for k in state.params.keys():
        assert state.params[k] is state2.params[k]
        assert state.momenta[k] is state2.momenta[k]

    # Test not inplace
    state_inp_false = sampler.update(state, batch, inplace=False)
    for k in state.params.keys():
        assert state.params[k] is not state_inp_false.params[k]
        assert state.momenta[k] is not state_inp_false.momenta[k]
