from functools import partial
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call
from optree import tree_map

from uqlib.laplace import diag_fisher


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


# Note this won't work if you replace torch.stack with torch.tensor
def normal_log_prior(p: dict):
    return torch.sum(
        torch.stack([Normal(0, 1).log_prob(ptemp).sum() for ptemp in p.values()])
    )


def normal_log_likelihood(y, y_pred):
    return (
        Normal(y_pred, 1, validate_args=False).log_prob(y).sum(dim=-1)
    )  # validate args introduces control flows not yet supported in torch.func.vmap


# def normal_log_likelihood(y, y_pred):
#     return (y - y_pred).square().sum(dim=-1)


def log_posterior_n(params, batch, model, n_data):
    y_pred = functional_call(model, params, batch[0])
    return (
        normal_log_prior(params) + normal_log_likelihood(batch[1], y_pred) * n_data
    ).mean()


def test_diag_fisher():
    torch.manual_seed(42)
    model = TestModel()

    xs = torch.randn(100, 10)
    ys = model(xs)

    dataloader = DataLoader(
        TensorDataset(xs, ys),
        batch_size=20,
    )

    log_posterior = partial(log_posterior_n, model=model, n_data=len(xs))

    params = dict(model.named_parameters())
    laplace_state = diag_fisher.init(params)
    for batch in dataloader:
        laplace_state = diag_fisher.update(laplace_state, log_posterior, batch)

    expected = tree_map(lambda x: torch.zeros_like(x), params)
    for x, y in zip(xs, ys):
        with torch.no_grad():
            g = torch.func.grad(lambda p: log_posterior(p, (x, y)))(params)
        expected = tree_map(lambda x, y: x + y**2, expected, g)

    for key in expected:
        assert torch.allclose(expected[key], laplace_state.prec_diag[key], atol=1e-5)

    # Also check full batch
    laplace_state_fb = diag_fisher.init(params)
    laplace_state_fb = diag_fisher.update(laplace_state_fb, log_posterior, (xs, ys))

    for key in expected:
        assert torch.allclose(expected[key], laplace_state_fb.prec_diag[key], atol=1e-5)
