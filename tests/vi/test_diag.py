from functools import partial
from typing import Any
import torch
import torchopt
from optree import tree_map

from uqlib import vi
from uqlib.utils import diag_normal_log_prob


def batch_normal_log_prob(
    p: dict, batch: Any, mean: dict, sd_diag: dict
) -> torch.Tensor:
    return diag_normal_log_prob(p, mean, sd_diag)


def test_batch_normal_log_prob():
    p = {"a": torch.randn(10, 10), "b": torch.randn(10, 1)}
    mean = tree_map(lambda x: torch.zeros_like(x), p)
    sd_diag = tree_map(lambda x: torch.ones_like(x), p)
    batch = torch.arange(10).reshape(-1, 1)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=mean, sd_diag=sd_diag
    )

    single_batch_evals = torch.stack(
        [batch_normal_log_prob_spec(p, b) for b in batch]
    ).mean()
    full_batch_evals = batch_normal_log_prob_spec(p, batch)

    assert torch.allclose(single_batch_evals, full_batch_evals)


def test_nelbo():
    target_mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)
    batch = torch.arange(10).reshape(-1, 1)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    target_nelbo_100 = vi.diag.nelbo(
        target_mean,
        target_sds,
        batch_normal_log_prob_spec,
        batch,
        n_samples=100,
    )

    assert torch.isclose(target_nelbo_100, torch.tensor(0.0), atol=1e-6)

    bad_mean = tree_map(lambda x: torch.zeros_like(x), target_mean)
    bad_sds = tree_map(lambda x: torch.ones_like(x), target_mean)

    bad_nelbo_100 = vi.diag.nelbo(
        bad_mean, bad_sds, batch_normal_log_prob_spec, batch, n_samples=100
    )

    assert bad_nelbo_100 > target_nelbo_100


def _test_vi_diag(optimizer_cls, stl):
    torch.manual_seed(42)
    target_mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    n_vi_samps_large = 1000

    init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)

    optimizer = optimizer_cls(lr=1e-2)

    state = vi.diag.init(init_mean, optimizer)

    init_sds = tree_map(torch.exp, state.log_sd_diag)

    batch = torch.arange(3).reshape(-1, 1)

    nelbo_init = vi.diag.nelbo(
        state.mean,
        init_sds,
        batch_normal_log_prob_spec,
        batch,
        n_samples=n_vi_samps_large,
    )

    nelbo_target = vi.diag.nelbo(
        target_mean,
        target_sds,
        batch_normal_log_prob_spec,
        batch,
        n_samples=n_vi_samps_large,
    )

    assert torch.isclose(nelbo_target, torch.tensor(0.0), atol=1e-6)
    assert nelbo_init > nelbo_target

    n_steps = 1000
    n_vi_samps = 5

    transform = vi.diag.build(
        batch_normal_log_prob_spec, optimizer, n_samples=n_vi_samps, stl=stl
    )

    state = transform.init(init_mean)

    nelbos = []

    for _ in range(n_steps):
        state = transform.update(state, batch)
        nelbos.append(state.nelbo)

    last_nelbos_mean = torch.tensor(nelbos[-10:]).mean()

    assert last_nelbos_mean < nelbo_init
    assert torch.isclose(last_nelbos_mean, nelbo_target, atol=1)

    for key in state.mean:
        assert torch.allclose(state.mean[key], target_mean[key], atol=0.5)
        assert torch.allclose(state.log_sd_diag[key].exp(), target_sds[key], atol=0.5)

    # Test inplace
    state_ip = transform.init(init_mean)
    state_ip2 = transform.update(
        state_ip,
        batch,
        inplace=True,
    )

    for key in state_ip2.mean:
        assert torch.allclose(state_ip2.mean[key], state_ip.mean[key], atol=1e-8)
        assert torch.allclose(
            state_ip2.log_sd_diag[key], state_ip.log_sd_diag[key], atol=1e-8
        )

    # Test not inplace
    state_ip_false = transform.update(
        state_ip,
        batch,
        inplace=False,
    )

    for key in state_ip.mean:
        assert not torch.allclose(
            state_ip_false.mean[key], state_ip.mean[key], atol=1e-8
        )
        assert not torch.allclose(
            state_ip_false.log_sd_diag[key], state_ip.log_sd_diag[key], atol=1e-8
        )


def test_vi_diag_sgd():
    _test_vi_diag(torchopt.sgd, False)


def test_vi_diag_adamw():
    _test_vi_diag(torchopt.adamw, False)


def test_vi_diag_sgd_stl():
    _test_vi_diag(torchopt.sgd, True)


def test_vi_diag_adamw_stl():
    _test_vi_diag(torchopt.adamw, True)
