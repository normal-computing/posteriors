from functools import partial
from typing import Any
import torch

from uqlib import vi
from uqlib.utils import tree_map, diag_normal_log_prob


def batch_normal_log_prob(
    p: dict, batch: Any, mean: dict, sd_diag: dict
) -> torch.Tensor:
    return diag_normal_log_prob(p, mean, sd_diag).repeat(batch.shape[0])


def test_batch_normal_log_prob():
    p = {"a": torch.randn(10, 10), "b": torch.randn(10, 1)}
    mean = tree_map(lambda x: torch.zeros_like(x), p)
    sd_diag = tree_map(lambda x: torch.ones_like(x), p)
    batch = torch.arange(10).reshape(-1, 1)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=mean, sd_diag=sd_diag
    )

    single_batch_evals = torch.cat([batch_normal_log_prob_spec(p, b) for b in batch])
    full_batch_evals = batch_normal_log_prob_spec(p, batch)

    assert torch.allclose(single_batch_evals, full_batch_evals)


def test_elbo():
    p = {"a": torch.randn(10, 10), "b": torch.randn(10, 1)}
    batch = torch.arange(10).reshape(-1, 1)

    mean = tree_map(lambda x: torch.zeros_like(x), p)
    sd_diag = tree_map(lambda x: torch.ones_like(x), p)

    elbo_100_1 = vi.diag.elbo(
        batch_normal_log_prob, batch, mean=mean, sd_diag=sd_diag, n_samples=100
    )

    elbo_100_2 = vi.diag.elbo(
        batch_normal_log_prob, batch, mean=mean, sd_diag=sd_diag, n_samples=100
    )

    assert torch.isclose(elbo_100_1, elbo_100_2, atol=1e-4)

    bad_mean = tree_map(lambda x: torch.ones_like(x), p)

    bad_elbo_100 = vi.diag.elbo(
        batch_normal_log_prob, batch, mean=bad_mean, sd_diag=sd_diag, n_samples=100
    )

    assert bad_elbo_100 < elbo_100_1


def test_vi_diag():
    target_mean = {"a": torch.randn(2, 3), "b": torch.randn(4, 1)}
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)

    batch_normal_log_prob_spec = partial(
        batch_normal_log_prob, mean=target_mean, sd_diag=target_sds
    )

    n_vi_samps = 1
    stl = False

    init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)

    optimizer_cls = partial(torch.optim.Adam, lr=1e-2)

    state = vi.diag.init(init_mean, optimizer_cls)

    n_steps = 3000

    batch = torch.arange(3).reshape(-1, 1)

    elbos = []

    for _ in range(n_steps):
        state = vi.diag.update(
            state, batch_normal_log_prob_spec, batch, n_vi_samps, stl
        )
        elbos.append(state.elbo)

    import matplotlib.pyplot as plt

    plt.plot(elbos)
    plt.savefig(f"elbos_stl_{stl}.png")
    plt.close()
