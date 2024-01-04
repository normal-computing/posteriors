from functools import partial
from typing import Any
import torch
from torch.func import grad_and_value
import torchopt
from optree import tree_map

from uqlib.sgmcmc import sghmc
from uqlib.sgmcmc.optim import SGHMC
from uqlib.utils import diag_normal_log_prob


def batch_normal_log_prob(
    p: dict, batch: Any, mean: dict, sd_diag: dict
) -> torch.Tensor:
    return diag_normal_log_prob(p, mean, sd_diag)


def test_sghmc_manual():
    target_mean = {"a": torch.randn(2, 1) + 10, "b": torch.randn(1, 1) + 10}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)
    batch = torch.arange(10).reshape(-1, 1)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    batch = torch.arange(3).reshape(-1, 1)

    n_steps = 5000
    lr = 1e-2
    alpha = 1.0
    beta = 0.0

    params = tree_map(lambda x: torch.zeros_like(x), target_mean)

    sampler = sghmc.build(lr=lr, alpha=alpha, beta=beta)

    sghmc_state = sampler.init(params)

    log_posts_manual = []

    all_params = tree_map(lambda x: x.unsqueeze(0), params)

    for _ in range(n_steps):
        grads, log_post = grad_and_value(batch_normal_log_prob_spec)(params, batch)
        updates, sghmc_state = sampler.update(grads, sghmc_state, inplace=False)
        params = torchopt.apply_updates(params, updates)

        all_params = tree_map(
            lambda x, y: torch.cat((x, y.unsqueeze(0))), all_params, params
        )

        log_posts_manual.append(log_post.item())

    burnin = 1000
    param_means = tree_map(lambda x: x[burnin:].mean(0), all_params)
    param_sds = tree_map(lambda x: x[burnin:].std(0), all_params)

    assert log_posts_manual[-1] > log_posts_manual[0]
    for k in target_mean.keys():
        assert torch.allclose(param_means[k], target_mean[k], atol=1e0, rtol=1e-1)
        assert torch.allclose(param_sds[k], target_sds[k], atol=1e0, rtol=1e-1)


def test_sghmc_funcopt():
    target_mean = {"a": torch.randn(2, 1) + 10, "b": torch.randn(1, 1) + 10}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)
    batch = torch.arange(10).reshape(-1, 1)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    batch = torch.arange(3).reshape(-1, 1)

    n_steps = 5000
    lr = 1e-2
    alpha = 1.0
    beta = 0.0

    # Test FuncOptimizer
    params = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)
    all_params = tree_map(lambda x: x.unsqueeze(0), params)

    func_sampler = torchopt.FuncOptimizer(sghmc.build(lr=lr, alpha=alpha, beta=beta))

    log_posts_FuncO = []

    for _ in range(n_steps):
        log_post = batch_normal_log_prob_spec(params, batch)

        param_leaves, tree_spec = torch.utils._pytree.tree_flatten(params)
        param_leaves = func_sampler.step(log_post, tuple(param_leaves))
        params = torch.utils._pytree.tree_unflatten(param_leaves, tree_spec)

        all_params = tree_map(
            lambda x, y: torch.cat((x, y.unsqueeze(0))), all_params, params
        )

        log_posts_FuncO.append(log_post.item())

    assert log_posts_FuncO[-1] > log_posts_FuncO[0]

    burnin = 1000
    param_means = tree_map(lambda x: x[burnin:].mean(0), all_params)
    param_sds = tree_map(lambda x: x[burnin:].std(0), all_params)

    for k in target_mean.keys():
        assert torch.allclose(param_means[k], target_mean[k], atol=1e0, rtol=1e-1)
        assert torch.allclose(param_sds[k], target_sds[k], atol=1e0, rtol=1e-1)


def test_sghmc_torch_api():
    target_mean = {"a": torch.randn(2, 1) + 10, "b": torch.randn(1, 1) + 10}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)
    batch = torch.arange(10).reshape(-1, 1)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    batch = torch.arange(3).reshape(-1, 1)

    n_steps = 5000
    lr = 1e-2
    alpha = 1.0
    beta = 0.0

    # Test PyTorch API
    params = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)
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
