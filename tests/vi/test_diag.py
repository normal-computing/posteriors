from functools import partial
from typing import Any
import torch

from uqlib import vi
from uqlib.utils import tree_map, diag_normal_log_prob


def batch_normal_log_prob(
    p: dict, batch: Any, mean: dict, sd_diag: dict
) -> torch.Tensor:
    return diag_normal_log_prob(p, mean, sd_diag).repeat(batch.shape[0])


def test_batch_normal_log_prob():
    p = {"a": torch.randn(10, 10), "b": torch.randn(10, 1)}
    mean = tree_map(lambda x: torch.zeros_like(x), p)
    sd_diag = tree_map(lambda x: torch.ones_like(x), p)
    batch = torch.arange(10).reshape(-1, 1)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=mean, sd_diag=sd_diag
    )

    single_batch_evals = torch.cat([batch_normal_log_prob_spec(p, b) for b in batch])
    full_batch_evals = batch_normal_log_prob_spec(p, batch)

    assert torch.allclose(single_batch_evals, full_batch_evals)


def test_elbo():
    target_mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)
    batch = torch.arange(10).reshape(-1, 1)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    target_elbo_100 = vi.diag.elbo(
        batch_normal_log_prob_spec,
        batch,
        mean=target_mean,
        sd_diag=target_sds,
        n_samples=100,
    )

    assert torch.isclose(target_elbo_100, torch.tensor(0.0), atol=1e-6)

    bad_mean = tree_map(lambda x: torch.zeros_like(x), target_mean)
    bad_sds = tree_map(lambda x: torch.ones_like(x), target_mean)

    bad_elbo_100 = vi.diag.elbo(
        batch_normal_log_prob_spec, batch, mean=bad_mean, sd_diag=bad_sds, n_samples=100
    )

    assert bad_elbo_100 < target_elbo_100


def _test_vi_diag(optimizer_cls, stl):
    torch.manual_seed(42)
    target_mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    n_vi_samps_large = 1000

    init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)

    optimizer_cls_spec = partial(optimizer_cls, lr=1e-2)

    state = vi.diag.init(init_mean, optimizer_cls_spec)

    init_sds = tree_map(torch.exp, state.log_sd_diag)

    batch = torch.arange(3).reshape(-1, 1)

    elbo_init = vi.diag.elbo(
        batch_normal_log_prob_spec,
        batch,
        mean=state.mean,
        sd_diag=init_sds,
        n_samples=n_vi_samps_large,
    )

    elbo_target = vi.diag.elbo(
        batch_normal_log_prob_spec,
        batch,
        mean=target_mean,
        sd_diag=target_sds,
        n_samples=n_vi_samps_large,
    )

    assert torch.isclose(elbo_target, torch.tensor(0.0), atol=1e-6)
    assert elbo_init < elbo_target

    n_steps = 1000
    n_vi_samps = 5

    elbos = []

    for _ in range(n_steps):
        state = vi.diag.update(
            state, batch_normal_log_prob_spec, batch, n_vi_samps, stl
        )
        elbos.append(state.elbo)

    last_elbos_mean = torch.tensor(elbos[-10:]).mean()

    assert last_elbos_mean > elbo_init
    assert torch.isclose(last_elbos_mean, elbo_target, atol=1)

    for key in state.mean:
        assert torch.allclose(state.mean[key], target_mean[key], atol=0.5)
        assert torch.allclose(state.log_sd_diag[key].exp(), target_sds[key], atol=0.5)


def test_vi_diag_sgd():
    _test_vi_diag(torch.optim.SGD, False)


def test_vi_diag_adamw():
    _test_vi_diag(torch.optim.AdamW, False)


def test_vi_diag_sgd_stl():
    _test_vi_diag(torch.optim.SGD, True)


def test_vi_diag_adamw_stl():
    _test_vi_diag(torch.optim.AdamW, True)
