import torch
from torchopt import sgd
from optree import tree_map
from optree.integration.torch import tree_ravel
from posteriors import gradient_descent


def rbf_kernel(x, y, length_scale=1):
    arg = tree_ravel(
        tree_map(lambda x, y: torch.exp(-(1 / length_scale) * ((x - y) ** 2)), x, y)
    )[0]
    return arg.sum()


def test_svgd_api():
    torch.manual_seed(42)
    target_mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}

    def log_prob_grad(p, b):
        return 1

    init_mean = tree_map(lambda x: torch.ones_like(x, requires_grad=True), target_mean)
    batch = torch.arange(3).reshape(-1, 1)
    transform = gradient_descent.svgd.build(log_prob_grad, sgd(lr=1e-1), rbf_kernel)

    state = transform.init(init_mean)
    state = transform.update(state, batch, inplace=False)

    # no crushes
    assert True


def dummy_test():
    pass
