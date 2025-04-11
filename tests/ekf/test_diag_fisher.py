from functools import partial
import torch
from optree.integration.torch import tree_ravel
from posteriors import ekf
from tests.scenarios import get_multivariate_normal_log_prob
from tests.utils import verify_inplace_update
from tests.ekf.utils import run_test_ekf_gaussian


def test_ekf_diag():
    torch.manual_seed(42)
    lr = 1e-1
    run_test_ekf_gaussian(partial(ekf.diag_fisher.build, lr=lr))


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

    def lr(step):
        return 1e-1 * (step + 1) ** -0.33

    transform = ekf.diag_fisher.build(log_prob, lr=lr)

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
