import pytest
import torch
import torchopt
from optree import tree_map

from posteriors import vi
from posteriors.utils import L_from_flat, L_to_flat
from tests.scenarios import get_multivariate_normal_log_prob
from tests.utils import verify_inplace_update


def test_nelbo_dense():
    dim = 3
    log_prob, (target_mean, target_cov) = get_multivariate_normal_log_prob(dim=dim)
    L = torch.linalg.cholesky(target_cov)

    batch = torch.arange(10).reshape(-1, 1)
    target_nelbo_100, _ = vi.dense.nelbo(
        target_mean,
        L_to_flat(L),
        batch,
        log_prob,
        n_samples=100,
    )
    assert torch.isclose(target_nelbo_100, torch.tensor(0.0), atol=1e-6)

    bad_mean = tree_map(lambda x: torch.zeros_like(x), target_mean)
    bad_L = torch.eye(dim)

    bad_nelbo_100, _ = vi.dense.nelbo(
        bad_mean, L_to_flat(bad_L), batch, log_prob, n_samples=100
    )
    assert bad_nelbo_100 > target_nelbo_100


@pytest.mark.parametrize("optimizer_cls", [torchopt.adam])
@pytest.mark.parametrize("stl", [True, False])
def test_vi_dense(optimizer_cls, stl, n_vi_samps=2):
    torch.manual_seed(42)
    dim = 3
    log_prob, (target_mean, target_cov) = get_multivariate_normal_log_prob(dim=dim)
    L = torch.linalg.cholesky(target_cov)
    target_mean = {str(i): x for i, x in enumerate(target_mean)}  # Make PyTree

    n_vi_samps_large = 1000

    init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)
    init_mean_copy = tree_map(lambda x: x.clone(), init_mean)

    optimizer = optimizer_cls(lr=1e-2)

    state = vi.dense.init(init_mean, optimizer)

    init_L_factor = state.L_factor

    batch = torch.arange(3).reshape(-1, 1)

    nelbo_init, _ = vi.dense.nelbo(
        state.params,
        init_L_factor,
        batch,
        log_prob,
        n_samples=n_vi_samps_large,
    )

    nelbo_target, _ = vi.dense.nelbo(
        target_mean,
        L_to_flat(L),
        batch,
        log_prob,
        n_samples=n_vi_samps_large,
    )

    assert torch.isclose(nelbo_target, torch.tensor(0.0), atol=1e-4)
    assert nelbo_init > nelbo_target

    n_steps = 1000

    transform = vi.dense.build(
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

    for key in state.params:
        assert torch.allclose(state.params[key], target_mean[key], atol=0.5)
        assert torch.allclose(
            init_mean[key], init_mean_copy[key]
        )  # check init_mean was left untouched
    state_L = L_from_flat(state.L_factor)
    state_cov = state_L @ state_L.T
    assert torch.allclose(state_cov, target_cov, atol=0.5)


def test_vi_dense_inplace():
    torch.manual_seed(42)
    dim = 3
    log_prob, (target_mean, _) = get_multivariate_normal_log_prob(dim=dim)

    # init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)
    init_mean = {
        str(i): torch.zeros_like(x, requires_grad=True)
        for i, x in enumerate(target_mean)
    }
    optimizer = torchopt.sgd(lr=1e-2)
    transform = vi.dense.build(
        log_prob,
        optimizer,
    )
    batch = torch.arange(3).reshape(-1, 1)

    verify_inplace_update(transform, init_mean, batch)


def test_vi_dense_sample():
    torch.manual_seed(42)

    mean, cov = get_multivariate_normal_log_prob(dim=3)[1]
    L = torch.linalg.cholesky(cov)

    optimizer = torchopt.sgd(lr=1e-2)
    state = vi.dense.init(mean, optimizer, L)

    num_samples = 10000
    samples = vi.dense.sample(state, (num_samples,))

    samples_mean = samples.mean(dim=0)
    samples_cov = torch.cov(samples.T)
    state_L = L_from_flat(state.L_factor)
    state_cov = state_L @ state_L.T

    assert torch.allclose(samples_cov, state_cov, atol=1e-1)
    assert torch.allclose(samples_mean, state.params, atol=1e-1)
