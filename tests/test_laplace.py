import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

from uqlib import laplace


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


def test_fit_diagonal_hessian():
    model = TestModel()

    xs = torch.randn(100, 10)
    ys = model(xs)

    dataloader = DataLoader(
        TensorDataset(xs, ys),
        batch_size=20,
    )

    diag_hess = laplace.fit_diagonal_hessian(
        model, normal_log_prior, normal_log_likelihood, dataloader
    )

    params = dict(model.named_parameters())

    assert all([v.shape == params[k].shape for k, v in diag_hess.items()])
    assert all([torch.all(v > 0) for v in diag_hess.values()])


def test_fit_empirical_fisher():
    model = TestModel()

    xs = torch.randn(100, 10)
    ys = model(xs)

    dataloader = DataLoader(
        TensorDataset(xs, ys),
        batch_size=20,
    )

    diag_hess = laplace.fit_diagonal_empirical_fisher(
        model, normal_log_prior, normal_log_likelihood, dataloader
    )

    params = dict(model.named_parameters())

    assert all([v.shape == params[k].shape for k, v in diag_hess.items()])
    assert all([torch.all(v > 0) for v in diag_hess.values()])
