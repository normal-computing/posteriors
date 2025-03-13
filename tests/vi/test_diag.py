import pytest
import torch
import torchopt
from optree import tree_map

from posteriors import vi
from posteriors.utils import diag_normal_log_prob
from tests.utils import verify_inplace_update
from tests.scenarios import get_multivariate_normal_log_prob


def test_nelbo_diag():
    target_mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)
    batch = torch.arange(10).reshape(-1, 1)

    def log_prob(p, batch):
        return diag_normal_log_prob(p, target_mean, target_sds), torch.tensor([])

    batch = torch.arange(10).reshape(-1, 1)
    target_nelbo_100, _ = vi.diag.nelbo(
        target_mean,
        target_sds,
        batch,
        log_prob,
        n_samples=100,
    )
    assert torch.isclose(target_nelbo_100, torch.tensor(0.0), atol=1e-6)

    bad_mean = tree_map(lambda x: torch.zeros_like(x), target_mean)
    bad_sds = tree_map(lambda x: torch.ones_like(x), target_mean)

    bad_nelbo_100, _ = vi.diag.nelbo(bad_mean, bad_sds, batch, log_prob, n_samples=100)
    assert bad_nelbo_100 > target_nelbo_100


@pytest.mark.parametrize("optimizer_cls", [torchopt.adam])
@pytest.mark.parametrize("stl", [True, False])
def test_vi_diag(optimizer_cls, stl, n_vi_samps=2):
    torch.manual_seed(42)
    target_mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)
    batch = torch.arange(10).reshape(-1, 1)

    def log_prob(p, batch):
        return diag_normal_log_prob(p, target_mean, target_sds), torch.tensor([])

    n_vi_samps_large = 1000

    init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)
    init_mean_copy = tree_map(lambda x: x.clone(), init_mean)

    optimizer = optimizer_cls(lr=1e-2)

    state = vi.diag.init(init_mean, optimizer)

    init_sds = tree_map(torch.exp, state.log_sd_diag)

    nelbo_init, _ = vi.diag.nelbo(
        state.params,
        init_sds,
        batch,
        log_prob,
        n_samples=n_vi_samps_large,
    )

    nelbo_target, _ = vi.diag.nelbo(
        target_mean,
        target_sds,
        batch,
        log_prob,
        n_samples=n_vi_samps_large,
    )

    assert torch.isclose(nelbo_target, torch.tensor(0.0), atol=1e-4)
    assert nelbo_init > nelbo_target

    n_steps = 1000

    transform = vi.diag.build(
        log_prob,
        optimizer,
        n_samples=n_vi_samps,
        stl=stl,
    )

    state = transform.init(init_mean)
    nelbos = []
    for _ in range(n_steps):
        state, _ = transform.update(state, batch, inplace=False)
        nelbos.append(state.nelbo.item())

    last_nelbos_mean = torch.tensor(nelbos[-10:]).mean()

    assert last_nelbos_mean < nelbo_init
    assert torch.isclose(last_nelbos_mean, nelbo_target, atol=1)

    state_sds = tree_map(torch.exp, state.log_sd_diag)

    for key in state.params:
        assert torch.allclose(state.params[key], target_mean[key], atol=0.5)
        assert torch.allclose(
            init_mean[key], init_mean_copy[key]
        )  # check init_mean was left untouched
        assert torch.allclose(state_sds[key], target_sds[key], atol=0.5)


def test_vi_diag_inplace():
    torch.manual_seed(42)
    dim = 3
    log_prob, (target_mean, _) = get_multivariate_normal_log_prob(dim=dim)

    # init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)
    init_mean = {
        str(i): torch.zeros_like(x, requires_grad=True)
        for i, x in enumerate(target_mean)
    }
    optimizer = torchopt.sgd(lr=1e-2)
    transform = vi.diag.build(
        log_prob,
        optimizer,
    )
    batch = torch.arange(3).reshape(-1, 1)

    verify_inplace_update(transform, init_mean, batch)


def test_vi_diag_sample():
    torch.manual_seed(42)

    mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}
    sd_diag = tree_map(lambda x: torch.randn_like(x).abs(), mean)
    log_sd_diag = tree_map(torch.log, sd_diag)

    optimizer = torchopt.sgd(lr=1e-2)
    state = vi.diag.init(mean, optimizer, log_sd_diag)

    num_samples = 10000
    samples = vi.diag.sample(state, (num_samples,))

    samples_mean = tree_map(lambda x: x.mean(dim=0), samples)
    samples_sds = tree_map(lambda x: x.std(dim=0), samples)

    for key in samples_mean.keys():
        assert torch.allclose(samples_mean[key], state.params[key], atol=1e-1)
        assert torch.allclose(
            samples_sds[key], torch.exp(state.log_sd_diag[key]), atol=1e-1
        )
