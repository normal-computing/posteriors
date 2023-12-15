from functools import partial
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

from uqlib import tree_map, hessian_diag
from uqlib.laplace import diag_hessian


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
    return Normal(y_pred, 1).log_prob(y).sum(dim=-1)


def log_posterior_n(params, batch, n_data):
    return normal_log_prior(params) + normal_log_likelihood(*batch) * n_data


def test_diag_hessian():
    model = TestModel()

    xs = torch.randn(100, 10)
    ys = model(xs)

    dataloader = DataLoader(
        TensorDataset(xs, ys),
        batch_size=20,
    )

    log_posterior = partial(log_posterior_n, n_data=len(xs))

    params = dict(model.named_parameters())
    laplace_state = diag_hessian.init(params)
    for batch in dataloader:
        laplace_state = diag_hessian.update(laplace_state, log_posterior, batch)

    expected = tree_map(lambda x: torch.zeros_like(x), params)
    for x, y in zip(xs, ys):
        with torch.no_grad():
            hess = hessian_diag(lambda p: log_posterior(p, (x, y)))(params)
        expected = tree_map(lambda x, y: x - y, expected, hess)

    for key in expected:
        assert torch.allclose(expected[key], laplace_state.prec_diag[key])

    # Also check full batch
    laplace_state_fb = diag_hessian.init(params)
    laplace_state_fb = diag_hessian.update(laplace_state_fb, log_posterior, (xs, ys))

    for key in expected:
        assert torch.allclose(expected[key], laplace_state_fb.prec_diag[key])
