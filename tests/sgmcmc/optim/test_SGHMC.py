from functools import partial
from typing import Any
import torch
from optree import tree_map

from uqlib.sgmcmc.optim import SGHMC
from uqlib.utils import diag_normal_log_prob


def batch_normal_log_prob(
    p: dict, batch: Any, mean: dict, sd_diag: dict
) -> torch.Tensor:
    return diag_normal_log_prob(p, mean, sd_diag)


def test_sghmc_torch_api():
    target_mean = {"a": torch.randn(2, 1) + 10, "b": torch.randn(1, 1) + 10}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)
    batch = torch.arange(10).reshape(-1, 1)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    batch = torch.arange(3).reshape(-1, 1)

    n_steps = 10000
    lr = 1e-2
    alpha = 1.0
    beta = 0.0

    # Test PyTorch API
    params = {
        "a": torch.zeros_like(target_mean["a"], requires_grad=True),
        "b": target_mean["b"],
    }
    all_params = tree_map(lambda x: x.unsqueeze(0), params)

    param_leaves, tree_spec = torch.utils._pytree.tree_flatten(params)
    sampler = SGHMC(param_leaves, lr=lr, alpha=alpha, beta=beta)

    log_posts_C = []

    for _ in range(n_steps):
        log_post = batch_normal_log_prob_spec(params, batch)
        log_post.backward()
        sampler.step()

        all_params = tree_map(
            lambda x, y: torch.cat((x, y.unsqueeze(0))), all_params, params
        )
        log_posts_C.append(log_post.item())

    assert log_posts_C[-1] > log_posts_C[0]

    burnin = 1000
    param_means = tree_map(lambda x: x[burnin:].mean(0), all_params)
    param_sds = tree_map(lambda x: x[burnin:].std(0), all_params)

    for k in target_mean.keys():
        assert torch.allclose(param_means[k], target_mean[k], atol=1e0, rtol=1e-1)
        assert torch.allclose(param_sds[k], target_sds[k], atol=1e0, rtol=1e-1)
