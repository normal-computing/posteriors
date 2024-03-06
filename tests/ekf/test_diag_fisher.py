from functools import partial
import torch
from optree import tree_map

from uqlib import ekf

from tests.scenarios import batch_normal_log_prob


def test_ekf_diag():
    torch.manual_seed(42)
    target_mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)

    batch = torch.arange(3).reshape(-1, 1)

    n_steps = 1000
    transform = ekf.diag_fisher.build(batch_normal_log_prob_spec, lr=1e-3)

    # Test inplace = False
    state = transform.init(init_mean)
    log_liks = []
    for _ in range(n_steps):
        state = transform.update(state, batch, inplace=False)
        log_liks.append(state.log_likelihood.item())

    assert log_liks[0] < log_liks[-1]

    for key in state.params:
        assert torch.allclose(state.params[key], target_mean[key], atol=1e-1)
        assert not torch.allclose(state.params[key], init_mean[key])

    # Test inplace = True
    state = transform.init(init_mean)
    log_liks = []
    for _ in range(n_steps):
        state = transform.update(state, batch, inplace=True)
        log_liks.append(state.log_likelihood.item())

    for key in state.params:
        assert torch.allclose(state.params[key], target_mean[key], atol=1e-1)
        assert torch.allclose(state.params[key], init_mean[key])

    # Test sample
    mean_copy = tree_map(lambda x: x.clone(), state.params)
    samples = ekf.diag_fisher.sample(state, (1000,))
    samples_mean = tree_map(lambda x: x.mean(dim=0), samples)
    samples_sd = tree_map(lambda x: x.std(dim=0), samples)
    for key in samples_mean:
        assert torch.allclose(samples_mean[key], state.params[key], atol=1e-1)
        assert torch.allclose(samples_sd[key], state.sd_diag[key], atol=1e-1)
        assert not torch.allclose(samples_mean[key], mean_copy[key])
