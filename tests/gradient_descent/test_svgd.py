from functools import partial
import torch
from optree import tree_map
from posteriors.gradient_descent import svgd
from optree.integration.torch import tree_ravel
from torch.distributions import Normal


def rbf_kernel(x, y, length_scale=1):
    arg = tree_ravel(
        tree_map(lambda x, y: torch.exp(-(1 / length_scale) * ((x - y) ** 2)), x, y)
    )[0]
    return arg.sum()


def flat_log_probability(params, batch, mean, sd_diag, normalize: bool = False):
    if normalize:

        def univariate_norm_and_sum(v, m, sd):
            return Normal(m, sd, validate_args=False).log_prob(v).sum()
    else:

        def univariate_norm_and_sum(v, m, sd):
            return (-0.5 * ((v - m) / sd) ** 2).sum()

    return univariate_norm_and_sum(params, mean, sd_diag)


def test_svgd():
    torch.manual_seed(42)
    target_mean = {"a": torch.randn(2, 1) + 10, "b": torch.randn(1, 1) + 10}
    flat_target_mean = tree_ravel(target_mean)[0]
    target_sds = tree_map(lambda x: torch.randn_like(x).abs(), target_mean)
    flat_target_sds = tree_ravel(target_sds)[0]
    init_mean = tree_map(lambda x: torch.ones_like(x, requires_grad=True), target_mean)

    batch = torch.arange(10).reshape(-1, 1)
    batch_normal_log_prob_spec = partial(
        flat_log_probability, mean=flat_target_mean, sd_diag=flat_target_sds
    )

    n_steps = 1000
    lr = 1e-2
    transform = svgd.build(batch_normal_log_prob_spec, lr, rbf_kernel)
    state = transform.init(init_mean)

    for _ in range(n_steps):
        state = transform.update(state, batch, inplace=False)

    flat_params = tree_ravel(state.params)[0]

    assert torch.allclose(flat_params, flat_target_mean, atol=1e-0, rtol=1e-1)
