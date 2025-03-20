from functools import partial
import torch
from posteriors import ekf
from tests.scenarios import get_multivariate_normal_log_prob
from tests.utils import verify_inplace_update
from tests.ekf.utils import run_test_ekf_gaussian


def test_ekf_diag():
    torch.manual_seed(42)
    lr = 1e-1
    run_test_ekf_gaussian(partial(ekf.diag_fisher.build, lr=lr))


def test_ekf_dense_inplace():
    torch.manual_seed(42)
    dim = 3
    log_prob, (target_mean, _) = get_multivariate_normal_log_prob(dim=dim)

    init_mean = {
        str(i): torch.zeros_like(x, requires_grad=True)
        for i, x in enumerate(target_mean)
    }
    batch = torch.arange(3).reshape(-1, 1)
    transform = ekf.dense_fisher.build(log_prob, lr=1e-1)

    verify_inplace_update(transform, init_mean, batch)


def test_ekf_dense_sample():
    torch.manual_seed(42)

    mean, cov = get_multivariate_normal_log_prob(dim=3)[1]

    state = ekf.dense_fisher.init(mean, cov)

    num_samples = 10000
    samples = ekf.dense_fisher.sample(state, (num_samples,))

    samples_mean = samples.mean(dim=0)
    samples_cov = torch.cov(samples.T)
    mean_copy = state.params.clone()

    assert torch.allclose(samples_cov, state.cov, atol=1e-1)
    assert torch.allclose(samples_mean, state.params, atol=1e-1)
    assert not torch.allclose(samples_mean, mean_copy)
