from functools import partial
from typing import Any
import torch
from torch.func import grad_and_value
import torchopt
from optree import tree_map

from uqlib.sgmcmc import sghmc
from uqlib.utils import diag_normal_log_prob


def batch_normal_log_prob(
    p: dict, batch: Any, mean: dict, sd_diag: dict
) -> torch.Tensor:
    return diag_normal_log_prob(p, mean, sd_diag)


def test_sghmc():
    torch.manual_seed(42)
    target_mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)
    batch = torch.arange(10).reshape(-1, 1)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    batch = torch.arange(3).reshape(-1, 1)

    # Test manual
    torch.manual_seed(42)
    params = tree_map(lambda x: torch.zeros_like(x), target_mean)

    sampler = sghmc.build(lr=1e-2, alpha=0.01, beta=0.0)

    sghmc_state = sampler.init(params)

    n_steps = 1000

    log_posts_manual = []

    for _ in range(n_steps):
        grads, log_post = grad_and_value(batch_normal_log_prob_spec)(params, batch)
        updates, sghmc_state = sampler.update(grads, sghmc_state, params)
        params = torchopt.apply_updates(params, updates)

        log_posts_manual.append(log_post.item())

    assert log_posts_manual[-1] > log_posts_manual[0]

    # Test FuncOptimizer
    torch.manual_seed(42)
    params = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)

    func_sampler = torchopt.FuncOptimizer(sghmc.build(lr=1e-2, alpha=0.01, beta=0.0))
    # func_sampler = torchopt.FuncOptimizer(torchopt.adam())

    n_steps = 1000

    log_posts_FuncO = []

    for _ in range(n_steps):
        log_post = batch_normal_log_prob_spec(params, batch)

        param_leaves, tree_spec = torch.utils._pytree.tree_flatten(params)
        param_leaves = func_sampler.step(log_post, tuple(param_leaves))
        params = torch.utils._pytree.tree_unflatten(param_leaves, tree_spec)

        log_posts_FuncO.append(log_post.item())

    assert log_posts_FuncO[-1] > log_posts_FuncO[0]

    assert torch.allclose(
        torch.tensor(log_posts_manual),
        torch.tensor(log_posts_FuncO),
        atol=1e-6,
    )
