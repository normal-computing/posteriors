from functools import partial
from typing import Any
import torch
from optree import tree_map

from uqlib import ekf
from uqlib.utils import diag_normal_log_prob


def batch_normal_log_prob(
    p: dict, batch: Any, mean: dict, sd_diag: dict
) -> torch.Tensor:
    return diag_normal_log_prob(p, mean, sd_diag), torch.tensor([])


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

    state = transform.init(init_mean)

    log_liks = []

    for _ in range(n_steps):
        state = transform.update(state, batch)
        log_liks.append(state.log_likelihood)

    for key in state.mean:
        assert torch.allclose(state.mean[key], target_mean[key], atol=1e-1)

    # Test inplace
    state_ip = transform.init(init_mean)
    state_ip2 = transform.update(
        state_ip,
        batch,
        inplace=True,
    )

    for key in state_ip2.mean:
        assert torch.allclose(state_ip2.mean[key], state_ip.mean[key], atol=1e-8)
        assert torch.allclose(state_ip2.sd_diag[key], state_ip.sd_diag[key], atol=1e-8)

    # Test not inplace
    state_ip_false = transform.init(
        tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)
    )
    state_ip_false2 = transform.update(
        state_ip_false,
        batch,
        inplace=False,
    )

    for key in state_ip.mean:
        assert not torch.allclose(
            state_ip_false2.mean[key], state_ip_false.mean[key], atol=1e-8
        )
        assert not torch.allclose(
            state_ip_false2.sd_diag[key], state_ip_false.sd_diag[key], atol=1e-8
        )
