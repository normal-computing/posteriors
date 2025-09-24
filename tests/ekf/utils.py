from typing import Callable
import torch
from optree.integrations.torch import tree_ravel
from tests.scenarios import get_multivariate_normal_log_prob
from posteriors.types import LogProbFn, Transform


def run_test_ekf_gaussian(
    transform_builder: Callable[[LogProbFn], Transform],
    dim: int = 3,
    n_steps: int = 1000,
):
    log_prob, (target_mean, target_cov) = get_multivariate_normal_log_prob(dim=dim)
    transform = transform_builder(log_prob)

    init_mean = {
        str(i): torch.zeros_like(x, requires_grad=True)
        for i, x in enumerate(target_mean)
    }

    batch = torch.arange(3).reshape(-1, 1)

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
