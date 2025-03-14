import torch
from optree.integration.torch import tree_ravel
from posteriors import ekf
from tests.scenarios import get_multivariate_normal_log_prob
from tests.utils import verify_inplace_update


def test_ekf_diag():
    torch.manual_seed(42)
    dim = 3
    log_prob, (target_mean, target_cov) = get_multivariate_normal_log_prob(dim=dim)

    # init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)
    init_mean = {
        str(i): torch.zeros_like(x, requires_grad=True)
        for i, x in enumerate(target_mean)
    }
    batch = torch.arange(3).reshape(-1, 1)
    n_steps = 1000
    transform = ekf.diag_fisher.build(log_prob, lr=1e-1)

    state = transform.init(init_mean)
    log_liks = []
    for _ in range(n_steps):
        state, _ = transform.update(state, batch, inplace=False)
        log_liks.append(state.log_likelihood.item())

    assert log_liks[0] < log_liks[-1]

    flat_params = tree_ravel(state.params)[0]
    flat_init_mean = tree_ravel(init_mean)[0]
    assert torch.allclose(flat_params, target_mean)
    assert not torch.allclose(flat_params, flat_init_mean)


def test_ekf_diag_inplace():
    torch.manual_seed(42)
    dim = 3
    log_prob, (target_mean, target_cov) = get_multivariate_normal_log_prob(dim=dim)

    # init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)
    init_mean = {
        str(i): torch.zeros_like(x, requires_grad=True)
        for i, x in enumerate(target_mean)
    }
    batch = torch.arange(3).reshape(-1, 1)
    transform = ekf.diag_fisher.build(log_prob, lr=1e-1)

    verify_inplace_update(transform, init_mean, batch)


def test_ekf_diag_sample():
    torch.manual_seed(42)

    mean, cov = get_multivariate_normal_log_prob(dim=3)[1]
    sd_diag = torch.sqrt(torch.diag(cov))

    state = ekf.diag_fisher.init(mean, sd_diag)

    num_samples = 10000
    samples = ekf.diag_fisher.sample(state, (num_samples,))

    flat_samples = torch.vmap(lambda s: tree_ravel(s)[0])(samples)
    samples_cov = torch.cov(flat_samples.T)
    samples_sd = torch.sqrt(torch.diag(samples_cov))

    mean_copy = state.params.clone()
    samples_mean = flat_samples.mean(dim=0)

    assert torch.allclose(samples_sd, state.sd_diag, atol=1e-1)
    assert torch.allclose(samples_mean, state.params, atol=1e-1)
    assert not torch.allclose(samples_mean, mean_copy)
