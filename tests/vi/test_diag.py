from functools import partial
import torch
import torchopt
from optree import tree_map

from uqlib import vi

from tests.scenarios import batch_normal_log_prob


def test_nelbo():
    target_mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)
    batch = torch.arange(10).reshape(-1, 1)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    target_nelbo_100, _ = vi.diag.nelbo(
        target_mean,
        target_sds,
        batch,
        batch_normal_log_prob_spec,
        n_samples=100,
    )

    assert torch.isclose(target_nelbo_100, torch.tensor(0.0), atol=1e-6)

    bad_mean = tree_map(lambda x: torch.zeros_like(x), target_mean)
    bad_sds = tree_map(lambda x: torch.ones_like(x), target_mean)

    bad_nelbo_100, _ = vi.diag.nelbo(
        bad_mean, bad_sds, batch, batch_normal_log_prob_spec, n_samples=100
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

    nelbo_init, _ = vi.diag.nelbo(
        state.mean,
        init_sds,
        batch,
        batch_normal_log_prob_spec,
        n_samples=n_vi_samps_large,
    )

    nelbo_target, _ = vi.diag.nelbo(
        target_mean,
        target_sds,
        batch,
        batch_normal_log_prob_spec,
        n_samples=n_vi_samps_large,
    )

    assert torch.isclose(nelbo_target, torch.tensor(0.0), atol=1e-6)
    assert nelbo_init > nelbo_target

    n_steps = 500
    n_vi_samps = 5

    transform = vi.diag.build(
        batch_normal_log_prob_spec, optimizer, n_samples=n_vi_samps, stl=stl
    )

    # Test inplace = False
    init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)
    init_mean_copy = tree_map(lambda x: x.clone(), init_mean)

    state = transform.init(init_mean)
    nelbos = []
    for _ in range(n_steps):
        state = transform.update(state, batch, inplace=False)
        nelbos.append(state.nelbo.item())

    last_nelbos_mean = torch.tensor(nelbos[-10:]).mean()

    assert last_nelbos_mean < nelbo_init
    assert torch.isclose(last_nelbos_mean, nelbo_target, atol=1)

    for key in state.mean:
        assert torch.allclose(state.mean[key], target_mean[key], atol=0.5)
        assert torch.allclose(state.log_sd_diag[key].exp(), target_sds[key], atol=0.5)
        assert torch.allclose(
            init_mean[key], init_mean_copy[key]
        )  # check init_mean was left untouched

    # Test inplace = True
    state = transform.init(init_mean)
    nelbos = []
    for _ in range(n_steps):
        _ = transform.update(state, batch, inplace=True)
        nelbos.append(state.nelbo.item())

    last_nelbos_mean = torch.tensor(nelbos[-10:]).mean()

    assert last_nelbos_mean < nelbo_init
    assert torch.isclose(last_nelbos_mean, nelbo_target, atol=1)

    for key in state.mean:
        assert torch.allclose(state.mean[key], target_mean[key], atol=0.5)
        assert torch.allclose(state.log_sd_diag[key].exp(), target_sds[key], atol=0.5)
        assert torch.allclose(
            state.mean[key], init_mean[key]
        )  # check init_mean was updated in place

    # Test sample
    mean_copy = tree_map(lambda x: x.clone(), state.mean)
    samples = vi.diag.sample(state, (1000,))
    samples_mean = tree_map(lambda x: x.mean(dim=0), samples)
    samples_sd = tree_map(lambda x: x.std(dim=0), samples)
    for key in samples_mean:
        assert torch.allclose(samples_mean[key], state.mean[key], atol=1e-1)
        assert torch.allclose(samples_sd[key], state.log_sd_diag[key].exp(), atol=1e-1)
        assert not torch.allclose(samples_mean[key], mean_copy[key])


def test_vi_diag_sgd():
    _test_vi_diag(torchopt.sgd, False)


def test_vi_diag_adamw():
    _test_vi_diag(torchopt.adamw, False)


def test_vi_diag_sgd_stl():
    _test_vi_diag(torchopt.sgd, True)


def test_vi_diag_adamw_stl():
    _test_vi_diag(torchopt.adamw, True)
