import torch
from torch.distributions import MultivariateNormal
import torchopt
from optree import tree_map
from optree.integration.torch import tree_ravel

from posteriors import vi
from posteriors.tree_utils import tree_size
from posteriors.utils import L_from_flat, L_to_flat


def test_nelbo():
    target_mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}
    num_params = tree_size(target_mean)
    L = torch.randn(num_params, num_params)
    L = torch.tril(L)
    target_cov = L @ L.T

    dist = MultivariateNormal(
        tree_ravel(target_mean)[0], covariance_matrix=target_cov, validate_args=False
    )

    def log_prob(p, b):
        return dist.log_prob(tree_ravel(p)[0]).sum(), torch.Tensor([])

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
    bad_L = torch.tril(torch.eye(num_params))

    bad_nelbo_100, _ = vi.dense.nelbo(
        bad_mean, L_to_flat(bad_L), batch, log_prob, n_samples=100
    )
    assert bad_nelbo_100 > target_nelbo_100


def _test_vi_dense(optimizer_cls, stl, n_vi_samps):
    torch.manual_seed(43)
    target_mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}
    num_params = tree_size(target_mean)
    L = torch.randn(num_params, num_params, requires_grad=True)
    L = torch.tril(L)
    target_cov = L @ L.T

    dist = MultivariateNormal(
        tree_ravel(target_mean)[0], covariance_matrix=target_cov, validate_args=False
    )

    def log_prob(p, b):
        return dist.log_prob(tree_ravel(p)[0]).sum(), torch.Tensor([])

    n_vi_samps_large = 1000

    init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)

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

    assert torch.isclose(nelbo_target, torch.tensor(0.0), atol=1e-6)
    assert nelbo_init > nelbo_target

    n_steps = 1000

    transform = vi.dense.build(
        log_prob,
        optimizer,
        n_samples=n_vi_samps,
        stl=stl,
    )

    # Test inplace = False
    init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)
    init_mean_copy = tree_map(lambda x: x.clone(), init_mean)

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

    # Test inplace = True
    state = transform.init(init_mean)
    nelbos = []
    for _ in range(n_steps):
        transform.update(state, batch, inplace=True)
        nelbos.append(state.nelbo.item())

    last_nelbos_mean = torch.tensor(nelbos[-10:]).mean()

    assert last_nelbos_mean < nelbo_init
    assert torch.isclose(last_nelbos_mean, nelbo_target, atol=1)

    for key in state.params:
        assert torch.allclose(state.params[key], target_mean[key], atol=0.5)
        assert torch.allclose(
            state.params[key], init_mean[key]
        )  # check init_mean was updated in place
    state_L = L_from_flat(state.L_factor)
    state_cov = state_L @ state_L.T
    assert torch.allclose(state_cov, target_cov, atol=0.5)

    # Test sample
    mean_copy = tree_map(lambda x: x.clone(), state.params)
    samples = vi.dense.sample(state, (5000,))
    flat_samples = torch.vmap(lambda s: tree_ravel(s)[0])(samples)
    samples_cov = torch.cov(flat_samples.T)
    samples_mean = tree_map(lambda x: x.mean(dim=0), samples)
    for key in samples_mean:
        assert torch.allclose(samples_mean[key], state.params[key], atol=1e-1)
        assert not torch.allclose(samples_mean[key], mean_copy[key])
    state_L = L_from_flat(state.L_factor)
    state_cov = state_L @ state_L.T
    assert torch.allclose(state_cov, samples_cov, atol=2e-1)


def test_vi_dense_sgd():
    _test_vi_dense(torchopt.sgd, False, 5)


def test_vi_dense_adamw():
    _test_vi_dense(torchopt.adamw, False, 1)


def test_vi_dense_sgd_stl():
    _test_vi_dense(torchopt.sgd, True, 1)


def test_vi_dense_adamw_stl():
    _test_vi_dense(torchopt.adamw, True, 5)
